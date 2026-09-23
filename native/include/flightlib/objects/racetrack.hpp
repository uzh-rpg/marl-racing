#pragma once

#include <algorithm>
#include <memory>
#include <random>

// flightlib
#include "flightlib/objects/particle_system.hpp"
#include "flightlib/objects/simple_gate.hpp"

// agilicious
#include "agilib/math/types.hpp"
#include "agilib/types/quad_state.hpp"
#include "agilib/utils/logger.hpp"
#include "agilib/utils/yaml.hpp"

using namespace agi;

namespace flightlib {

class RaceTrack {
 public:
  RaceTrack(const agi::Yaml& gate_node, const int state_buffer_size);

  const Gate& get(const int gate_idx) const { return gates_[gate_idx % N_]; }

  bool isWorldCollision(const Ref<const Vector<3>>) const;
  bool isWorldCollision(const agi::QuadState& state) const;
  bool isGateCollision(const agi::QuadState& state,
                       const float sim_dt = 0.02) const;
  bool isCyclic() const { return is_cyclic_; };

  void addInitialState(const agi::QuadState& state, const int gate_id);

  int getInitialState(agi::QuadState& state, const int gate_id);
  int getInitialState(agi::QuadState& state);

  int getShiftedStartingState(agi::QuadState& state, int agent_id,
                              int num_agents, bool test_env = false);
  int getNumberOfGates() const { return N_; };

  bool addGatePositionRandomization(const Scalar max_distance = 0.2);
  bool resetGatePositionRandomization();

  bool setGateSafetyDistRelative(const Scalar dist_safe = 0.0);

  void setGateSize(const Scalar factor = 1.0);

  void setWorldBoxExtend(const Vector<3>& margin) {
    world_box_adjusted_.col(0) = world_box_.col(0) - margin;
    world_box_adjusted_.col(1) = world_box_.col(1) + margin;
  }

  Matrix<3, 2> getWorldBox() const { return world_box_; };

  ParticleSystem& getParticleSystem() { return particle_system_; }

  void setParticleSystem(int num_agents) {
    particle_system_ = ParticleSystem(num_agents);
  }

 private:
  void loadRacetrack(const agi::Yaml& gate_node);
  void prepareInitialStateBuffer(const int state_buffer_size);

  Logger logger_{"RaceTrack"};

  // racetrack
  int N_ = 0;
  std::vector<Gate> gates_;
  std::vector<Gate> gates_nominal_;
  Scalar gate_size_default_;
  Matrix<3, 2> world_box_;
  Matrix<3, 2> world_box_adjusted_;

  // Particle system for air downwash simulation
  ParticleSystem particle_system_;

  Vector<3> start_pos_;
  Quaternion start_att_;
  bool is_cyclic_{true};

  // state buffer
  std::vector<std::vector<agi::QuadState>> initial_state_buffer_;
  std::vector<int> initial_state_buffer_idx_;
  const int state_buffer_size_;

  std::uniform_int_distribution<int> random_sample_;
  std::uniform_int_distribution<int> random_gate_;
  std::uniform_real_distribution<Scalar> uni_01_dist_{0.0, 1.0};
  std::random_device rd_;
  std::mt19937 rg_{rd_()};

};

}  // namespace flightlib
