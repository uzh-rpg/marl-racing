#include "flightlib/envs/vision_racing_env/marl_no_camera_env.hpp"

namespace mnce = flightlib::marlnocameraenv;

namespace flightlib {

MarlNoCameraEnv::MarlNoCameraEnv(const agi::Yaml &node, const int env_id)
  : MarlVisionRacingBaseEnv(node, env_id) {
  loadCustom(node_);
}

bool MarlNoCameraEnv::loadCustom(const agi::Yaml &node) {
  bool valid = true;
  obs_dim_ =
    num_agents_ * (mnce::kNObs + mnce::kNObsOtherAgent * mnce::maxNumOpponents);
  single_obs_dim_ = mnce::kNObs + mnce::kNObsOtherAgent * mnce::maxNumOpponents;

  valid &= node_["randomization"]["gate_position_randomization"].getIfDefined(
    gate_position_randomization_);
  yamlCheck(valid, "randomization", "gate_position_randomization");

  valid &=
    node_["environment"]["start_pos_prob"].getIfDefined(reset_start_prob_);
  yamlCheck(valid, "environment", "start_pos_prob");

  logger_.info("MarlNoCameraEnv loaded with %d agents", num_agents_);

  valid &= node_["rewards"]["coeffs"]["ranking_lead_coeff"].getIfDefined(
    ranking_lead_coeff_);
  yamlCheck(valid, "rewards | coeffs", "ranking_lead_coeff");

  valid &= node_["rewards"]["coeffs"]["opponent_collision_coeff"].getIfDefined(
    opponent_collision_coeff_);
  yamlCheck(valid, "rewards | coeffs", "opponent_collision_coeff");

  valid &= node_["environment"]["agent_collision_radius"].getIfDefined(
    collision_radius_);
  yamlCheck(valid, "environment", "agent_collision_radius");

  valid &= node_["environment"]["air_disturbance_modeling"].getIfDefined(
    air_disturbance_modeling_);
  if (!valid) {
    logger_.warn("Air disturbance modeling is turned off per default");
    air_disturbance_modeling_ = false;
    valid = true;
  }

  valid &= node_["environment"]["agent_invisible_prob"].getIfDefined(
    agent_invisible_prob_final_);
  yamlCheck(valid, "environment", "agent_invisible_prob");
  agent_invisible_prob_final_ = agent_invisible_prob_;
  if (test_env_) {
    agent_invisible_prob_ = 0.0;
    agent_invisible_prob_final_ = 0.0;
  }

  if (!node_["environment"]["max_laps"].getIfDefined(max_laps_)) {
    max_laps_ = 3;  // default value
  }
  logger_.info("Max laps for race completion: %d", max_laps_);

  finished_laps_.setZero(static_cast<int>(num_agents_));
  starting_gate_num_.setZero(static_cast<int>(num_agents_));
  race_finished_.setZero(static_cast<int>(num_agents_));

  if (num_agents_ > mnce::maxNumOpponents + 1) {
    logger_.fatal("Number of agents exceeds maximum number of agents");
    return false;
  }

  prop_radius_.setZero(static_cast<int>(num_agents_));
  motor_to_motor_distances_.setZero(static_cast<int>(num_agents_));
  agent_collision_radius_.setZero(static_cast<int>(num_agents_));
  for (int ag = 0; ag < static_cast<int>(num_agents_); ag++) {
    if (agi_quad_nominals_[ag].m_ < 0.4) {
      // kolibri drone
      prop_radius_[ag] = 0.0737 / 2.0;
      motor_to_motor_distances_[ag] = 0.118;
    } else {
      // offboard drone
      prop_radius_[ag] = 0.1295 / 2.0;
      motor_to_motor_distances_[ag] = 0.266;
    }
    agent_collision_radius_[ag] = collision_radius_;
  }

  agents_visible_.setOnes(num_agents_);
  pos_offset_agents_.setZero(num_agents_, 3);

  return reset();
}

void MarlNoCameraEnv::applyPosOffset() {
  for (int ag = 0; ag < static_cast<int>(num_agents_); ag++) {
    agi_quad_state_[ag].p += pos_offset_agents_.row(ag);
  }
}

void MarlNoCameraEnv::setTestMode(bool test_mode, const Vector<3> pos_offset) {
  test_env_ = test_mode;
  for (int ag = 0; ag < static_cast<int>(num_agents_); ag++) {
    pos_offset_agents_.row(ag) = pos_offset;
  }
}

void MarlNoCameraEnv::setTestModeAgent(bool test_mode,
                                       const Vector<3> pos_offset, int id) {
  test_env_ = test_mode;
  pos_offset_agents_.row(id) = pos_offset;
}

bool MarlNoCameraEnv::reset() {

  lap_time_.setZero(num_agents_);
  last_lap_time_.setZero(num_agents_);
  cumulative_lap_time_.setZero(num_agents_);
  finished_laps_.setZero(num_agents_);
  traversal_error_.setZero(num_agents_);
  crashed_agents_.setZero(num_agents_);
  race_finished_.setZero(num_agents_);
  Eigen::VectorXi v =
    Eigen::VectorXi::LinSpaced(num_agents_, 0, num_agents_ - 1);
  ranking_ = v.cast<double>();
  frozen_crash_distance_.setConstant(num_agents_, -1.0);
  all_done_ = false;

  // sample agent visibility
  if (test_env_) {
    agents_visible_.setOnes(num_agents_);
  } else {
    agents_visible_ = (Vector<>::Random(num_agents_).array() * 0.5 + 0.5 >
                       agent_invisible_prob_)
                        .cast<Scalar>();
  }

  // shuffle agent_ids
  std::vector<int> random_agent_ids(num_agents_);
  std::iota(random_agent_ids.begin(), random_agent_ids.end(), 0);
  std::shuffle(random_agent_ids.begin(), random_agent_ids.end(), rg_);

  bool reset_start = uni_01_dist_(rg_) < reset_start_prob_;

  for (int ag = 0; ag < static_cast<int>(num_agents_); ag++) {
    if (reset_start || test_env_) {
      if (test_env_) {
        num_passed_gates_[ag] = racetrack_->getShiftedStartingState(
          agi_quad_state_[ag], ag, static_cast<int>(num_agents_), test_env_);
      } else {
        num_passed_gates_[ag] = racetrack_->getShiftedStartingState(
          agi_quad_state_[ag], random_agent_ids[ag],
          static_cast<int>(num_agents_), test_env_);
      }
      applyPosOffset();
    } else {
      num_passed_gates_[ag] = racetrack_->getInitialState(agi_quad_state_[ag]);
    }
    starting_gate_num_[ag] = num_passed_gates_[ag];

    if (!test_env_) {
      // perturb starting point but "gently" except position
      applyRandomState(1.0, 0.25, 0.2, 0.2, ag);

      // apply dynamics randomization
      applyRandomDrag(ag);
      applyRandomInertia(ag);
      applyRandomMass(ag);
      applyRandomThrust(ag);
      applyRandomTorque(ag);
    }

    // reset agilib quad state
    if (!agi_simulators_[ag]->reset(agi_quad_state_[ag])) {
      logger_.error("Cannot reset agi quadrotor");
    }

    agi_cmd_[ag].t = 0.0 + getRandomDelay() * (!test_env_);

    prev_agi_cmd_[ag] = agi_cmd_[ag];
    prev_agi_cmd_[ag].collective_thrust = 9.81;
    prev_agi_cmd_[ag].omega.setZero();

    prev_agi_quad_state_[ag] = agi_quad_state_[ag];

    flight_mode_[ag] = mvre::kFlying;
  }

  if (!test_env_) {
    racetrack_->addGatePositionRandomization(gate_position_randomization_);
  } else {
    racetrack_->resetGatePositionRandomization();
  }

  global_t_ = 0.0;

  racetrack_->setParticleSystem(static_cast<int>(num_agents_));

  return true;
}

bool MarlNoCameraEnv::reset(Ref<Vector<>> obs) { return reset(); }

bool MarlNoCameraEnv::reset_single_agent(const int agent_id) {
  if (agent_id < 0 || agent_id >= static_cast<int>(num_agents_)) {
    logger_.error("Invalid agent id to reset: %d", agent_id);
    return false;
  }

  lap_time_[agent_id] = 0;
  last_lap_time_[agent_id] = 0;
  cumulative_lap_time_[agent_id] = 0;
  traversal_error_[agent_id] = 0;
  race_finished_[agent_id] = false;
  frozen_crash_distance_[agent_id] = -1.0;

  std::uniform_int_distribution<int> dist(0, static_cast<int>(num_agents_) - 1);
  int random_id = dist(rg_);

  if (uni_01_dist_(rg_) < reset_start_prob_ || test_env_) {
    num_passed_gates_[agent_id] = racetrack_->getShiftedStartingState(
      agi_quad_state_[agent_id], random_id, static_cast<int>(num_agents_),
      test_env_);
    applyPosOffset();
  } else {
    num_passed_gates_[agent_id] =
      racetrack_->getInitialState(agi_quad_state_[agent_id]);
  }

  if (!test_env_) {
    // perturb starting point but "gently" except position
    applyRandomState(1.0, 0.25, 0.2, 0.2, agent_id);

    // apply dynamics randomization
    applyRandomDrag(agent_id);
    applyRandomInertia(agent_id);
    applyRandomMass(agent_id);
    applyRandomThrust(agent_id);
    applyRandomTorque(agent_id);
  }

  if (!agi_simulators_[agent_id]->reset(agi_quad_state_[agent_id])) {
    logger_.error("Cannot reset agi quadrotor");
  };

  // if any agent reaches the end of episode, all are currently reset
  agi_cmd_[agent_id].t = 0.0 + getRandomDelay() * (!test_env_);

  prev_agi_cmd_[agent_id] = agi_cmd_[agent_id];
  prev_agi_cmd_[agent_id].collective_thrust = 9.81;
  prev_agi_cmd_[agent_id].omega.setZero();

  prev_agi_quad_state_[agent_id] = agi_quad_state_[agent_id];

  flight_mode_[agent_id] = mvre::kFlying;

  if (test_env_) {
    agents_visible_(agent_id) = 1.0;
  } else {
    agents_visible_(agent_id) =
      (Vector<>::Random(num_agents_).array() * 0.5 + 0.5 >
       agent_invisible_prob_)
        .cast<Scalar>()(0);
  }

  Scalar max_time = 0.0;
  for (int ag = 0; ag < static_cast<int>(num_agents_); ag++) {
    max_time = std::max(max_time, agi_cmd_[ag].t);
  }
  global_t_ = max_time;

  return true;
}

bool MarlNoCameraEnv::getObs(Ref<Vector<>> obs) {

  if (static_cast<std::size_t>(obs.size()) != single_obs_dim_ * num_agents_) {
    logger_.error("Observation dimension mismatch. %d != %d", obs.size(),
                  single_obs_dim_ * num_agents_);
    return false;
  }

  if (air_disturbance_modeling_) {

    racetrack_->getParticleSystem().update(sim_dt_);

    // ------------ AIR DISTURBANCE: particle emission ------------
    for (int ag = 0; ag < static_cast<int>(num_agents_); ag++) {
      Vector<3> windspeed = Vector<3>::Zero();
      Vector<3> downwash_velocity =
        racetrack_->getParticleSystem().evaluateDownwash(agi_quad_state_[ag].p);

      // randomized downwash scaling between [0.5 1.5]
      Scalar downwash_scale =
        test_env_
          ? 1.0
          : (0.5 + static_cast<Scalar>(uni_01_dist_(rg_)) * (1.0 - 0.5));
      windspeed = downwash_scale * downwash_velocity;

      agi_simulators_[ag]->setWindspeed(windspeed);
    }
  }

  obs.setZero();

  // relative distances/velocities of every agent to each other
  computeRelativeObs();

  // compute ranking of agents based on num_passed_gates_ and gate distance
  computeAgentRanking();

  // Prepare opponent index vector once per observation (avoids per-agent
  // allocations)
  std::vector<int> idx(mnce::maxNumOpponents);
  std::iota(idx.begin(), idx.end(), 0);

  for (int ag = 0; ag < static_cast<int>(num_agents_); ag++) {
    if (air_disturbance_modeling_) {
      // Calculate total thrust from motor commands
      float total_thrust = agi_quad_nominals_[ag]
                             .motorOmegaToThrust(agi_quad_state_[ag].mot)
                             .sum();

      // Spawn particles with enhanced parameters
      racetrack_->getParticleSystem().spawnParticles(
        agi_quad_state_[ag].p,    // position
        agi_quad_state_[ag].R(),  // rotation matrix
        total_thrust,             // total thrust for velocity calculations
        motor_to_motor_distances_[ag],  // motor-to-motor distance
        prop_radius_[ag]                // propeller radius
      );
    }

    // -------------------- science robotics observations --------------------

    const Gate &gate = racetrack_->get(num_passed_gates_[ag]);
    const Gate &gate_n = racetrack_->get(num_passed_gates_[ag] + 1);

    Matrix<4, 3> corners_xyz_1 = gate.corners();
    Matrix<4, 3> corners_xyz_2 = gate_n.corners();

    Matrix<4, 3> corners_xyz =
      corners_xyz_1.rowwise() - agi_quad_state_[ag].p.transpose();
    Matrix<4, 3> relative_corners = corners_xyz_2 - corners_xyz_1;

    obs.segment<mnce::kNObsRelCorner>(ag * single_obs_dim_ +
                                      mnce::kObsRelCorner) =
      Map<Vector<>>(relative_corners.data(), relative_corners.size());
    obs.segment<mnce::kNObsCornerDist>(ag * single_obs_dim_ +
                                       mnce::kObsCornerDist) =
      Map<Vector<>>(corners_xyz.data(), corners_xyz.size());
    obs.segment<mnce::kNObsQuadOri>(ag * single_obs_dim_ + mnce::kObsQuadOri) =
      Map<Vector<>>(agi_quad_state_[ag].R().data(),
                    agi_quad_state_[ag].R().size());
    obs.segment<mnce::kNObsQuadVel>(ag * single_obs_dim_ + mnce::kObsQuadVel) =
      agi_quad_state_[ag].v;

    // observation of other agents relative positions and velocities
    if (test_env_) {
      std::shuffle(idx.begin(), idx.begin() + num_agents_ - 1, rg_);
    } else {
      std::shuffle(idx.begin(), idx.end(), rg_);
    }

    int current_idx = 0;
    for (int i = 0; i < static_cast<int>(num_agents_); ++i) {
      // observe agent i, write in shuffled other agent observation slot
      // idx[current_idx]
      if (i == ag or !static_cast<bool>(agents_visible_(ag))) continue;
      obs.segment<mnce::kNObsOtherAgentRelPos>(
        ag * single_obs_dim_ + mnce::kObsOtherAgent +
        idx[current_idx] * mnce::kNObsOtherAgent + mnce::kObsOtherAgentRelPos) =
        rel_pos_.row(ag).segment<mnce::kNObsOtherAgentRelPos>(
          i * mnce::kNObsOtherAgentRelPos);
      obs.segment<mnce::kNObsOtherAgentRelVel>(
        ag * single_obs_dim_ + mnce::kObsOtherAgent +
        idx[current_idx] * mnce::kNObsOtherAgent + mnce::kObsOtherAgentRelVel) =
        rel_vel_.row(ag).segment<mnce::kNObsOtherAgentRelVel>(
          i * mnce::kNObsOtherAgentRelVel);

      current_idx++;
    }

    // // -------------------- new RSS observations --------------------

  }

  // Compute minimum distances once per observation step (O(N^2) -> once instead
  // of per-agent)
  updateMinAgentDistance();
  return true;
}  // namespace flightlib

void MarlNoCameraEnv::computeRelativeObs() {
  // compute a map of relative distances/velocities of every agent to each
  // other

  rel_pos_.setZero(static_cast<int>(num_agents_),
                   static_cast<int>(num_agents_) * mnce::kNObsOtherAgentRelPos);
  rel_vel_.setZero(static_cast<int>(num_agents_),
                   static_cast<int>(num_agents_) * mnce::kNObsOtherAgentRelVel);

  rel_pos_norm_.setZero(static_cast<int>(num_agents_),
                        static_cast<int>(num_agents_));

  for (int i = 0; i < static_cast<int>(num_agents_); ++i) {
    // Diagonal remains zero by construction
    for (int j = i + 1; j < static_cast<int>(num_agents_); ++j) {
      if (!static_cast<bool>(agents_visible_(i)) ||
          !static_cast<bool>(agents_visible_(j)))
        continue;
        // relative position and velocity with some noise
      const Vector<3> true_relative_position =
        agi_quad_state_[j].p - agi_quad_state_[i].p;
      const Vector<3> relative_position =
        true_relative_position - 0.5 * uni_01_dist_(rg_) * 0.1 * Vector<3>::Ones();
      const Vector<3> relative_velocity =
        agi_quad_state_[j].v - agi_quad_state_[i].v - 0.5 * uni_01_dist_(rg_) * 0.4 *
          Vector<3>::Ones();

      // Fill both directions to avoid recomputation in full matrix
      rel_pos_.block<1, 3>(i, j * 3) = relative_position.transpose();
      rel_pos_.block<1, 3>(j, i * 3) = (-relative_position).transpose();
      rel_vel_.block<1, 3>(i, j * 3) = relative_velocity.transpose();
      rel_vel_.block<1, 3>(j, i * 3) = (-relative_velocity).transpose();
      const Scalar norm_ij = true_relative_position.norm();
      rel_pos_norm_(i, j) = norm_ij;
      rel_pos_norm_(j, i) = norm_ij;  // symmetric matrix
    }
  }
}

void MarlNoCameraEnv::computeAgentRanking() {
  const Vector<> previous_ranking = ranking_;
  ranking_.setZero(num_agents_);
  min_agent_distance_.setConstant(num_agents_, std::numeric_limits<Scalar>::max());
  std::vector<std::pair<int, std::pair<int, Scalar>>> agent_scores(num_agents_);

  // compute scores for each agent
  // finished agents get max score to ensure they stay in front
  for (int ag = 0; ag < static_cast<int>(num_agents_); ++ag) {
    if (race_finished_[ag]) {
      // Finished agents are ordered by elapsed race time: smaller is faster.
      agent_scores[ag] = {ag, {INT_MAX, cumulative_lap_time_[ag]}};
    } else {
      Scalar dist_to_next_gate = racetrack_->get(num_passed_gates_[ag])
                                  .centerDistance(agi_quad_state_[ag]);
      if (test_env_ && crashed_agents_[ag]) {
        // The post-reward ranking call captures the crash position before
        // stepSimulation grounds the drone on the next step.
        if (frozen_crash_distance_[ag] < 0.0) {
          frozen_crash_distance_[ag] = dist_to_next_gate;
        }
        dist_to_next_gate = frozen_crash_distance_[ag];
      }
      agent_scores[ag] = {ag, {num_passed_gates_[ag], dist_to_next_gate}};
    }
  }

  // sort agents based on the passed gates and distance to the next gate
  std::sort(agent_scores.begin(), agent_scores.end(),
            [&previous_ranking](const std::pair<int, std::pair<int, Scalar>> &a,
               const std::pair<int, std::pair<int, Scalar>> &b) {
              if (a.second.first != b.second.first) {
                return a.second.first > b.second.first;
              }
              if (a.second.second != b.second.second) {
                return a.second.second < b.second.second;
              }
              if (previous_ranking[a.first] != previous_ranking[b.first]) {
                return previous_ranking[a.first] < previous_ranking[b.first];
              }
              return a.first < b.first;
            });

  for (int rank = 0; rank < static_cast<int>(num_agents_); ++rank) {
    int agent_id = agent_scores[rank].first;
    // Recompute the complete ordering after every state update. This makes
    // the fastest completed race rank first and preserves exact integer ranks.
    ranking_[agent_id] = rank;

  }
}

bool MarlNoCameraEnv::computeReward(Ref<Vector<>> reward) {
  Matrix<Eigen::Dynamic> per_agent_reward;
  // total reward is extra variable
  per_agent_reward.setZero(static_cast<int>(num_agents_),
                           reward_names_.size() - 1);

  Vector<Eigen::Dynamic> total_reward_all;
  total_reward_all.setZero(static_cast<int>(num_agents_));

  lap_time_.setZero(static_cast<int>(num_agents_));
  traversal_error_.setZero(static_cast<int>(num_agents_));

  for (int ag = 0; ag < static_cast<int>(num_agents_); ag++) {
    Scalar progress_reward_all = 0.0;
    Scalar state_omega_penalty_all = 0.0;
    Scalar ranking_reward_all = 0.0;

    Scalar crashing_reward_all = 0.0;
    Scalar opponent_collision_reward_all = 0.0;

    lap_time_[ag] = 0;

    if (flight_mode_[ag] == mvre::kGatePassed) {
      const Gate &passed_gate = racetrack_->get(num_passed_gates_[ag]);
      const Vector<3> p_G = passed_gate.toGateFrame(agi_quad_state_[ag].p);
      traversal_error_[ag] = p_G.segment<2>(1).norm();

      if ((num_passed_gates_[ag] % racetrack_->getNumberOfGates()) ==
            starting_gate_num_[ag] &&
          num_passed_gates_[ag] > 0) {
        lap_time_[ag] = agi_cmd_[ag].t - last_lap_time_[ag];
        last_lap_time_[ag] = agi_cmd_[ag].t;
        ++finished_laps_[ag];

        // Check if agent finished required number of laps
        if (finished_laps_[ag] >= max_laps_ && test_env_) {
          cumulative_lap_time_[ag] = agi_cmd_[ag].t;
          race_finished_[ag] = true;
        }
      }
      if (!racetrack_->isCyclic() && num_passed_gates_[ag] > 0 &&
          (num_passed_gates_[ag] % (racetrack_->getNumberOfGates() - 1) == 0)) {
        lap_time_[ag] = agi_cmd_[ag].t - last_lap_time_[ag];
        last_lap_time_[ag] = agi_cmd_[ag].t;
      }
      if (!racetrack_->isCyclic() && num_passed_gates_[ag] > 0 &&
          (num_passed_gates_[ag] % (racetrack_->getNumberOfGates() - 1) == 0)) {
        lap_time_[ag] = agi_cmd_[ag].t - last_lap_time_[ag];
        last_lap_time_[ag] = agi_cmd_[ag].t;
      }

      ++num_passed_gates_[ag];
    } else if (flight_mode_[ag] == mvre::kCrashGate) {
      const Gate &passed_gate = racetrack_->get(num_passed_gates_[ag]);
      const Vector<3> p_G = passed_gate.toGateFrame(agi_quad_state_[ag].p);
      traversal_error_[ag] = p_G.segment<2>(1).norm();
    }
    // Progress is measured against the next gate after updating the gate count.

    const auto &curr_gate = racetrack_->get(num_passed_gates_[ag]);
    const Scalar prev_dist = curr_gate.centerDistance(prev_agi_quad_state_[ag]);
    const Scalar curr_dist = curr_gate.centerDistance(agi_quad_state_[ag]);
    const Scalar progress_reward = progress_coeff_ * (prev_dist - curr_dist);

    Scalar additional_ranking_reward =
      (static_cast<float>(num_agents_) - ranking_[ag]) /
        (static_cast<float>(num_agents_) + 1e-10) -
      1.0;
    const Scalar ranking_reward =
      ranking_lead_coeff_ * additional_ranking_reward;

    const Scalar state_omega_penalty =
      smoothness_coeff_ * state_omega_coeff_ *
      (agi_quad_state_[ag].w.segment<2>(0).array() /
       act_max_.row(ag).segment<2>(1).array().transpose())
        .matrix()
        .squaredNorm();
    if (!prev_agi_cmd_[ag].valid() || !prev_agi_quad_state_[ag].valid()) {
      logger_.warn("Previous state or command was invalid.");
      prev_agi_cmd_[ag] = agi_cmd_[ag];
      prev_agi_quad_state_[ag] = agi_quad_state_[ag];
    }

    progress_reward_all += progress_reward;
    state_omega_penalty_all += state_omega_penalty;
    ranking_reward_all += ranking_reward;

    // since flightmode is updated before reward we do terminal rewards here
    if (flight_mode_[ag] == mvre::kCrashWorld) {
      crashing_reward_all =
        -collision_coeff_ * agi_quad_state_[ag].v.norm() - 1.0;
      crashed_agents_[ag] = true;
      if (!test_env_) {
        reset_single_agent(ag);
      }
    } else if (flight_mode_[ag] == mvre::kCrashGate) {
      crashing_reward_all =
        -collision_coeff_ * (traversal_error_[ag] * traversal_error_[ag]) - 1.0;
      crashed_agents_[ag] = true;
      if (!test_env_) {
        reset_single_agent(ag);
      }
    } else if (flight_mode_[ag] == mvre::kCrashOtherAgent) {
      opponent_collision_reward_all =
        -opponent_collision_coeff_ * agi_quad_state_[ag].v.norm() - 1.0;
      crashed_agents_[ag] = true;
      if (!test_env_) {
        reset_single_agent(ag);
      }
    } else if (flight_mode_[ag] == mvre::kEpisodeDone) {
      crashing_reward_all = 0.0;
    } else {
      // penalty if too close to other agent but still alive
      Scalar penalty_radius = 2.0 * agent_collision_radius_[ag];
      if (min_agent_distance_[ag] < penalty_radius) {
        Scalar normalized_dist =
          (min_agent_distance_[ag] - agent_collision_radius_[ag]) /
          agent_collision_radius_[ag];
        normalized_dist = std::max(0.0, normalized_dist);  // Clamp to [0, 1]

        opponent_collision_reward_all =
          (-opponent_collision_coeff_ * agi_quad_state_[ag].v.norm() - 1.0) *
          std::exp(-7.0 * normalized_dist);
      }
    }

    if (!crashed_agents_[ag]) {
      per_agent_reward.row(ag) << progress_reward_all, 0.0, 0.0, state_omega_penalty_all,
        0.0, 0.0, 0.0,
        crashing_reward_all, ranking_reward_all, opponent_collision_reward_all;
    } else {
      per_agent_reward.row(ag) << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        crashing_reward_all, 0.0, opponent_collision_reward_all;
    }

    total_reward_all(ag) = per_agent_reward.row(ag).sum();
  }

  int index = 0;
  for (int ag = 0; ag < static_cast<int>(num_agents_); ag++) {
    for (int j = 0; j < per_agent_reward.cols(); ++j) {
      reward(index) = per_agent_reward.row(ag)(j);
      index++;
    }
    reward(index) = total_reward_all(ag);
    index++;
  }

  // getObs() ranks before updateFlightmode() classifies the current step.
  // Re-rank after gate/crash/finish bookkeeping so exported final_ranking
  // reflects this step's event, especially the 22nd finishing passage.
  computeAgentRanking();

  return true;
}

void MarlNoCameraEnv::updateMinAgentDistance() {
  // crash if quadrotor is too close to another quadrotor, keep alive
  // with random probability
  // agents_visible part is already taken care of in the rel_pos_norm_
  // computation such that you don't get a penalty if the other agent is
  // invisible

  // minimum distance to other agents such that no self collision
  for (int ag = 0; ag < static_cast<int>(num_agents_); ag++) {
    Scalar collision_norm =
      (rel_pos_norm_.row(ag).array() > 1e-5)
        .select(rel_pos_norm_.row(ag), std::numeric_limits<Scalar>::max())
        .minCoeff();
    min_agent_distance_(ag) = collision_norm;
  }
}

bool MarlNoCameraEnv::updateFlightmode() {
  for (int ag = 0; ag < static_cast<int>(num_agents_); ag++) {
    // Check if race is finished for this agent (keep agent on ground)
    if (race_finished_[ag] && test_env_) {
      flight_mode_[ag] = mvre::kFinishedRace;
      continue;
    }

    const Gate &next_gate = racetrack_->get(num_passed_gates_[ag]);
    if (racetrack_->isWorldCollision(agi_quad_state_[ag])) {
      flight_mode_[ag] = mvre::kCrashWorld;
    } else if (racetrack_->isGateCollision(agi_quad_state_[ag], sim_dt_)) {
      flight_mode_[ag] = mvre::kCrashGate;
    } else if (next_gate.hasPassed(agi_quad_state_[ag], sim_dt_)) {
      flight_mode_[ag] = mvre::kGatePassed;
      racetrack_->addInitialState(agi_quad_state_[ag], num_passed_gates_[ag]);
    } else if (min_agent_distance_(ag) < agent_collision_radius_[ag]) {
      if (test_env_ || uni_01_dist_(rg_) > random_keep_alive_) {
        flight_mode_[ag] = mvre::kCrashOtherAgent;
      }
    } else {
      flight_mode_[ag] = mvre::kFlying;
    }
    if (agi_cmd_[ag].t > max_t_ && !test_env_) {
      flight_mode_[ag] = mvre::kEpisodeDone;
    }
  }

  all_done_ = false;
  // double check with global timer for all agents
  if (global_t_ > max_t_) {
    all_done_ = true;
  }

  return true;
}

bool MarlNoCameraEnv::isTerminalState(Scalar &reward) const {

  // all done if global timer is over
  reward = 0;
  return all_done_;

  // currently in vec_env_base only the last agent would get terminal reward

}

void MarlNoCameraEnv::updateExtraInfo(void) {
  for (int ag = 0; ag < static_cast<int>(num_agents_); ag++) {
    extra_info_["flightmode_" + std::to_string(ag)] = flight_mode_[ag];
    extra_info_["traversal_err_" + std::to_string(ag)] = traversal_error_[ag];
    extra_info_["time_" + std::to_string(ag)] = agi_cmd_[ag].t;
    extra_info_["battery_voltage_" + std::to_string(ag)] =
      agi_quad_state_[ag].ubat.value();
    extra_info_["laptime_" + std::to_string(ag)] = lap_time_[ag];
    extra_info_["cumulative_laptime_" + std::to_string(ag)] =
      cumulative_lap_time_[ag];
    extra_info_["finished_laps_" + std::to_string(ag)] = finished_laps_[ag];
    extra_info_["ranking_" + std::to_string(ag)] = ranking_[ag];
    // done could be potentially removed if same as flightmode information
    if (all_done_) {
      extra_info_["done_" + std::to_string(ag)] = 1;
    } else if (crashed_agents_[ag] || flight_mode_[ag] == mvre::kEpisodeDone) {
      extra_info_["done_" + std::to_string(ag)] = 1;
      if (!test_env_) {
        reset_single_agent(ag);
        crashed_agents_[ag] = false;
      }
    } else if (race_finished_[ag]) {
      // Agent finished race, keep it on ground but not marked as crashed
      extra_info_["done_" + std::to_string(ag)] = 1;
    } else {
      extra_info_["done_" + std::to_string(ag)] = 0;
    }
  }
}

Scalar MarlNoCameraEnv::calcCurriculum(const Vector<> counters,
                                       const Vector<> values,
                                       const Scalar query_counter) const {
  if (counters.size() == 0) {
    return 0.0;
  }
  if (counters.size() != values.size()) {
    logger_.error("Curriculum size mismatch.");
    return 0.0;
  }

  if (counters.size() == 1) {
    return values[0];
  }

  if (query_counter <= counters[0]) {
    return values[0];
  } else if (query_counter >= counters[counters.size() - 1]) {
    return values[counters.size() - 1];
  } else {
    for (long i = 0; i < counters.size() - 1; ++i) {
      if (query_counter >= counters[i] && query_counter < counters[i + 1]) {
        return values[i] + (values[i + 1] - values[i]) *
                             (query_counter - counters[i]) /
                             (counters[i + 1] - counters[i]);
      }
    }
  }

  // Just to make the compiler happy, not reachable if input is correct
  return 0.0;
}

void MarlNoCameraEnv::curriculumUpdate(void) {
  curriculum_updates_++;

  smoothness_coeff_ =
    2 * std::clamp(curriculum_updates_ / 100.0 - 0.5, 0.0, 1.0) + 1;

  // random_keep_alive first 1.0 until 200 updates, then decrease
  random_keep_alive_ = calcCurriculum(
    Vector<2>({100, 200}), Vector<2>({0.1, 0.1}), curriculum_updates_);
  agent_invisible_prob_ = calcCurriculum(
    Vector<2>({50, 200}),
    Vector<2>({agent_invisible_prob_final_, agent_invisible_prob_final_}),
    curriculum_updates_);

  const Scalar hor_extend =
    2 * (1 - std::clamp(curriculum_updates_ / 100.0, 0.0, 1.0));
  racetrack_->setWorldBoxExtend({hor_extend, hor_extend, 0.0});

  curriculumGateSize();
  curriculumStartPosProb();
}

}  // namespace flightlib
