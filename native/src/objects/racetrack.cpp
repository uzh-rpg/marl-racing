#include "flightlib/objects/racetrack.hpp"

namespace flightlib {

RaceTrack::RaceTrack(const agi::Yaml& gate_node, const int state_buffer_size)
  : state_buffer_size_(state_buffer_size),
    random_sample_(0, state_buffer_size_ - 1) {
  loadRacetrack(gate_node);
  prepareInitialStateBuffer(state_buffer_size);
}

void RaceTrack::loadRacetrack(const agi::Yaml& gate_node) {
  bool valid = true;
  Vector<6> world_box;
  valid &= gate_node["world_box"].getIfDefined(world_box);
  world_box_ = world_box.reshaped(3, 2);
  world_box_adjusted_ = world_box_;

  valid &= gate_node["start_pos"].getIfDefined(start_pos_);
  Vector<4> quat_coeff;
  valid &= gate_node["start_attitude"].getIfDefined(quat_coeff);
  start_att_.coeffs() << quat_coeff[1], quat_coeff[2], quat_coeff[3],
    quat_coeff[0];

  valid &= gate_node["loop"].getIfDefined(is_cyclic_);
  std::cout << "loop track: " << is_cyclic_ << std::endl;

  int num_gates = gate_node["gates"]["N"].as<size_t>();
  for (int i = 0; i < num_gates; i++) {
    std::string gate_id = "Gate" + std::to_string(i + 1);

    // load position and rotation
    Vector<3> posvec;
    valid &= gate_node["gates"][gate_id]["position"].getIfDefined(posvec);
    Vector<4> rotvec;
    valid &= gate_node["gates"][gate_id]["rotation"].getIfDefined(rotvec);

    Scalar gate_size = 1.45;  // default size, assuming square gates
    gate_node["gates"][gate_id]["size"][1].getIfDefined(gate_size);
    gate_size_default_ = gate_size;

    Scalar gate_thickness = 0.275;  // default thickness
    gate_node["gates"][gate_id]["gate_thickness"].getIfDefined(gate_thickness);

    Scalar safety_distance = 0.0;

    Vector<3> position = Vector<3>(posvec.data());
    Quaternion orientation =
      Quaternion(rotvec[0], rotvec[1], rotvec[2], rotvec[3]);

    // in-place construct static gate
    gates_.emplace_back(position, orientation, gate_size, gate_thickness,
                        safety_distance);
    gates_nominal_.emplace_back(position, orientation, gate_size,
                                gate_thickness, safety_distance);
  }

  logger_.info("Number of gates loaded in racetrack: %d", gates_.size());

  if (!valid) {
    logger_.fatal("Loading the gates failed.");
  }

  if (gates_.size() < 1) {
    logger_.error("Number of gate in the race track is insufficient: %d",
                  gates_.size());
  }
  N_ = gates_.size();
  random_gate_ = std::uniform_int_distribution<int>(0, N_ - 1);
}

void RaceTrack::prepareInitialStateBuffer(const int state_buffer_size) {
  initial_state_buffer_idx_.clear();
  initial_state_buffer_idx_.resize(N_, 0);

  initial_state_buffer_.clear();
  for (int gate_idx = 0; gate_idx < N_; ++gate_idx) {
    std::vector<agi::QuadState> states;
    for (int i = 0; i < state_buffer_size; ++i) {
      agi::QuadState state;
      state.setZero();
      state.p = gates_[gate_idx].position();
      state.q(gates_[gate_idx].attitude());
      state.v = state.q() * (2 * Vector<3>::UnitX());
      states.push_back(std::move(state));
    }
    initial_state_buffer_.push_back(std::move(states));
  }
}

int RaceTrack::getInitialState(agi::QuadState& state, const int gate_id) {
  int sample_idx = random_sample_(rg_);
  state = initial_state_buffer_[gate_id % N_][sample_idx];

  if (state.ubat.value() < 14.8) {
    state.resetBattery();
  }
  return (gate_id + 1) % N_;
}

int RaceTrack::getInitialState(agi::QuadState& state) {
  int sample_gate = random_gate_(rg_);
  getInitialState(state, sample_gate);
  return (sample_gate + 1) % N_;
}

int RaceTrack::getShiftedStartingState(agi::QuadState& state, int agent_id,
                                       int num_agents, bool test_env) {
  state.setZero();
  state.resetBattery();
  // 1 m arc-length spacing at same XY radius from Gate 1; face Gate 1.
  const Vector<3> gate_pos = gates_[0].position();
  Vector<3> v0 = start_pos_ - gate_pos;
  v0.z() = 0.0;
  const Scalar radius = v0.norm();
  const Scalar dth = static_cast<Scalar>(1.0) / radius;

  // when more than 4 agents, shift the entire formation back along the arc
  Scalar base_angle_offset = 0.0;
  if (num_agents > 4) {
    int extra_agents = num_agents - 4;
    base_angle_offset = -static_cast<Scalar>(extra_agents) * dth / 2.0;
  }

  Scalar theta = 0.0;
  if (num_agents > 1 || test_env) {
    theta = base_angle_offset + static_cast<Scalar>(agent_id) * dth;
  } else {
    // single agent in non-test env, randomize start position more to be fairer
    const Scalar max_right = static_cast<Scalar>(M_PI) / 4.0;
    const Scalar rand01 = uni_01_dist_(rg_);
    theta = rand01 * max_right;
  }

  const Scalar c = std::cos(theta);
  const Scalar s = std::sin(theta);
  Vector<3> v_rot(c * v0.x() - s * v0.y(), s * v0.x() + c * v0.y(), 0.0);
  state.p = gate_pos + v_rot;
  state.p.z() = start_pos_.z();
  state.q() =
    start_att_ * Quaternion::FromTwoVectors(Vector<3>::UnitX(),
                                            (gate_pos - state.p).normalized());
  return 0;
}

void RaceTrack::addInitialState(const agi::QuadState& state,
                                const int gate_id) {
  agi::QuadState& new_state =
    initial_state_buffer_[gate_id % N_]
                         [initial_state_buffer_idx_[gate_id % N_]];
  new_state = state;
  new_state.p = get(gate_id).position();
  initial_state_buffer_idx_[gate_id % N_] =
    (initial_state_buffer_idx_[gate_id % N_] + 1) % state_buffer_size_;
}

bool RaceTrack::isWorldCollision(const agi::QuadState& state) const {
  return isWorldCollision(state.p);
}

bool RaceTrack::isWorldCollision(const Ref<const Vector<3>> quad_pos) const {
  bool x_valid = quad_pos(0) > world_box_adjusted_(0, 0) &&
                 quad_pos(0) < world_box_adjusted_(0, 1);
  bool y_valid = quad_pos(1) > world_box_adjusted_(1, 0) &&
                 quad_pos(1) < world_box_adjusted_(1, 1);
  bool z_valid = quad_pos(2) > world_box_adjusted_(2, 0) &&
                 quad_pos(2) < world_box_adjusted_(2, 1);
  return !(x_valid && y_valid && z_valid);
}

bool RaceTrack::isGateCollision(const agi::QuadState& state,
                                const float sim_dt) const {
  for (const auto& gate : gates_) {
    if (gate.inCollision(state, sim_dt)) {
      return true;
    }
  }
  return false;
}

void RaceTrack::setGateSize(const Scalar factor) {
  for (int i = 0; i < N_; ++i) {
    gates_[i].setGateSize(factor * gate_size_default_);
    gates_nominal_[i].setGateSize(factor * gate_size_default_);
  }

}

bool RaceTrack::addGatePositionRandomization(const Scalar max_distance) {
  gates_.clear();
  for (int i = 0; i < N_; ++i) {
    gates_.emplace_back(
      gates_nominal_[i].position() + Vector<3>::Random() * max_distance,
      gates_nominal_[i].attitude(), gates_nominal_[i].gateDim(),
      gates_nominal_[i].gateThick(), gates_nominal_[i].safeDist());
  }
  return true;
}

bool RaceTrack::resetGatePositionRandomization() {
  gates_.clear();
  for (int i = 0; i < N_; ++i) {
    gates_.emplace_back(
      gates_nominal_[i].position(), gates_nominal_[i].attitude(),
      gates_nominal_[i].gateDim(), gates_nominal_[i].gateThick(),
      gates_nominal_[i].safeDist());
  }
  return true;
}

bool RaceTrack::setGateSafetyDistRelative(const Scalar dist_safe) {
  for (auto& g : gates_) {
    g.setSafetyDistRelative(dist_safe);
  }
  for (auto& g : gates_nominal_) {
    g.setSafetyDistRelative(dist_safe);
  }
  return true;
}

}  // namespace flightlib
