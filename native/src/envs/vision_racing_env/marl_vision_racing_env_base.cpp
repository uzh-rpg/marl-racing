#include "flightlib/envs/vision_racing_env/marl_vision_racing_env_base.hpp"

namespace flightlib {

MarlVisionRacingBaseEnv::MarlVisionRacingBaseEnv(const agi::Yaml& node,
                                                 const int env_id)
  : EnvBase(env_id) {
  load(node);
}

bool MarlVisionRacingBaseEnv::load(const agi::Yaml& node) {
  logger_.updateName("MarlVisionRacingBaseEnv" + std::to_string(env_id_));
  logger_.info("Initializing Environment...");

  node_ = node;
  bool valid = true;

  valid &= loadNumAgents(node);
  valid &= loadEnvironment(node);
  valid &= loadRacetrack(node);
  valid &= loadQuadrotor(node);
  valid &= loadRewardComponents(node);
  valid &= loadDynamicsRandomization(node);
  valid &= loadCurriculum(node);
  valid &= loadExtraInfo(node);

  act_dim_ = mvre::kNAct * num_agents_;
  return valid;
}

bool MarlVisionRacingBaseEnv::yamlCheck(const bool valid,
                                        const std::string& category,
                                        const std::string& key) const {
  if (!valid) {
    logger_.fatal(
      "Cannot load configuration. One of the keys %s | %s does not exist",
      category.c_str(), key.c_str());
    return false;
  }
  return true;
}

bool MarlVisionRacingBaseEnv::loadNumAgents(const agi::Yaml& node) {
  bool valid = true;
  valid &= node_["environment"]["num_agents"].getIfDefined(num_agents_);
  yamlCheck(valid, "environment", "num_agents");
  logger_.info("Number of agents: %i", num_agents_);
  return valid;
}

bool MarlVisionRacingBaseEnv::loadEnvironment(const agi::Yaml& node) {
  // load environment
  bool valid = true;
  valid &= node_["environment"]["sim_dt"].getIfDefined(sim_dt_);
  valid &= node_["environment"]["max_t"].getIfDefined(max_t_);
  yamlCheck(valid, "environment", "sim_dt, max_t");

  valid &= node_["main"]["test_env"].getIfDefined(test_env_);
  yamlCheck(valid, "main", "test_env");
  logger_.info("Test env: %s", test_env_ ? "true" : "false");
  return valid;
}

bool MarlVisionRacingBaseEnv::loadRacetrack(const agi::Yaml& node) {
  bool valid = true;
  std::string track_config_path;
  valid &= node["environment"]["tracks"].getIfDefined(track_config_path);

  int init_state_buffer_size;
  valid &= node["environment"]["init_state_buffer_size"].getIfDefined(
    init_state_buffer_size);
  yamlCheck(valid, "environment", "tracks, init_state_buffer_size");

  fs::path gate_file =
    fs::path(getenv("FLIGHTMARE_PATH")) / fs::path(track_config_path);
  if (!file_exists(gate_file)) {
    logger_.fatal("Track config file does not exist");
  }
  logger_.info("Using track file %s", gate_file.c_str());
  agi::Yaml gate_node(gate_file);

  racetrack_ = std::make_shared<RaceTrack>(gate_node, init_state_buffer_size);
  return valid;
}

bool MarlVisionRacingBaseEnv::loadRewardComponents(const agi::Yaml& node) {
  bool valid = true;
  valid &= node_["rewards"]["components"].getIfDefined(reward_names_);
  yamlCheck(valid, "rewards", "components");
  valid &=
    node_["rewards"]["coeffs"]["progress_coeff"].getIfDefined(progress_coeff_);
  valid &= node_["rewards"]["coeffs"]["collision_coeff"].getIfDefined(
    collision_coeff_);
  valid &= node_["rewards"]["coeffs"]["state_omega_penalty"].getIfDefined(
    state_omega_coeff_);
  yamlCheck(valid, "rewards | coeffs",
            "progress_coeff, collision_coeff, state_omega_penalty");
  for (const auto* key : {"gate_pass_coeff", "gate_view_coeff",
                          "gate_behind_coeff", "cmd_omega_xy_coeff",
                          "cmd_omega_z_coeff", "cmd_diff_omega_xy_coeff",
                          "cmd_diff_omega_z_coeff", "cmd_diff_thrust_coeff"}) {
    Scalar value = 0.0;
    node_["rewards"]["coeffs"][key].getIfDefined(value);
    if (value != 0.0) {
      throw std::invalid_argument(std::string("Reward unused by the paper: ") + key);
    }
  }
  rew_dim_ = reward_names_.size();
  return valid;
}

bool MarlVisionRacingBaseEnv::loadExtraInfo(const agi::Yaml& node) {
  bool valid = true;
  std::vector<std::string> info_names;
  valid &= node_["extra_infos"]["components"].getIfDefined(info_names);
  yamlCheck(valid, "extra_infos", "components");

  if (valid) {
    for (std::size_t i = 0; i < num_agents_; i++) {
      for (std::string name : info_names) {
        extra_info_.insert({name + "_" + std::to_string(i), 0.0});
      }
    }
  }
  return valid;
}

bool MarlVisionRacingBaseEnv::loadDynamicsRandomization(const agi::Yaml& node) {
  bool valid = true;
  valid &= node_["randomization"]["dynamics_randomization_init"].getIfDefined(
    dynamics_randomization_);
  valid &= node_["randomization"]["random_mass"].getIfDefined(mass_rd_coeff_);
  valid &=
    node_["randomization"]["random_inertia"].getIfDefined(inertia_rd_coeff_);
  valid &= node_["randomization"]["random_drag"].getIfDefined(drag_rd_coeff_);
  valid &= node_["randomization"]["random_delay"].getIfDefined(delay_rd_coeff_);
  valid &=
    node_["randomization"]["random_thrust_map"].getIfDefined(thrust_rd_coeff_);
  valid &=
    node_["randomization"]["random_torque"].getIfDefined(torque_rd_coeff_);
  valid &= node_["randomization"]["rand_init_pos"].getIfDefined(init_pos_rd_);
  valid &= node_["randomization"]["rand_init_vel"].getIfDefined(init_vel_rd_);
  valid &= node_["randomization"]["rand_init_att"].getIfDefined(init_att_rd_);
  valid &= node_["randomization"]["rand_init_ome"].getIfDefined(init_ome_rd_);
  yamlCheck(valid, "randomization",
            "dynamics_randomization_init, random_mass, random_inertia, "
            "random_drag, random_delay, random_thrust_map, random_torque, "
            "rand_init_[pos|vel|att|ome]_[x,y,z]range");
  return valid;
}

bool MarlVisionRacingBaseEnv::loadCurriculum(const agi::Yaml& node) {
  bool valid = true;
  int gate_points;
  valid &=
    node_["curriculum"]["relative_gate_size_points"].getIfDefined(gate_points);
  if (gate_points > 0) {
    gate_curr_updates_.resize(gate_points);
    gate_curr_sizes_.resize(gate_points);
    valid &=
      node_["curriculum"]["relative_gate_size"].getIfDefined(gate_curr_sizes_);
    valid &= node_["curriculum"]["relative_gate_size_counter"].getIfDefined(
      gate_curr_updates_);
  }
  int start_prob_points;
  valid &= node_["curriculum"]["start_pos_prob_points"].getIfDefined(
    start_prob_points);
  if (start_prob_points > 0) {
    start_pos_prob_curr_updates_.resize(start_prob_points);
    start_pos_prob_curr_values_.resize(start_prob_points);
    valid &= node_["curriculum"]["start_pos_prob_values"].getIfDefined(
      start_pos_prob_curr_values_);
    valid &= node_["curriculum"]["start_pos_prob_counter"].getIfDefined(
      start_pos_prob_curr_updates_);
  }
  return valid;
}

bool MarlVisionRacingBaseEnv::loadQuadrotor(const agi::Yaml& node) {
  const fs::path agi_param_directory =
    std::string(getenv("RACING_PARAMS"));

  std::vector<std::string> filenames;

  bool valid = true;
  valid &= node_["environment"]["simulation_files"].getIfDefined(filenames);

  while (valid && filenames.size() < num_agents_) {
    if (filenames.size() == 0) {
      std::string filename;
      valid &= node_["environment"]["simulation_files"].getIfDefined(filename);
      filenames.emplace_back(filename);
    } else {
      logger_.warn(
        "Not enough simulation files for all agents. "
        "Using default (kolibri_simulation.yaml) for missing files.");
      filenames.push_back("kolibri_simulation.yaml");
    }
  }

  if (!valid || filenames.size() != num_agents_) {
    logger_.error("Simulation files are not set for all agents.");
    return false;
  }

  // initialize matrices
  pi_act_.setZero(num_agents_, mvre::kNAct);
  act_mean_.setZero(num_agents_, mvre::kNAct);
  act_std_.setZero(num_agents_, mvre::kNAct);
  act_min_.setZero(num_agents_, mvre::kNAct);
  act_max_.setZero(num_agents_, mvre::kNAct);

  flight_mode_.setOnes(num_agents_, 1);
  flight_mode_ *= mvre::kFlying;
  num_passed_gates_.setZero(num_agents_, 1);

  for (std::size_t i = 0; i < num_agents_; i++) {
    agi::SimulatorParams param(agi_param_directory / filenames[i],
                               agi_param_directory);

    agi_simulators_.emplace_back(
      std::make_unique<agi::QuadrotorSimulator>(param));

    agi::Quadrotor temp_quad_nominal = agi_simulators_.back()->getQuadrotor();
    agi_quad_nominals_.push_back(temp_quad_nominal);

    if (env_id_ == 0) {
      logger_ << "Quadrotor from "
              << agi_param_directory / fs::path("quads") / filenames[i]
              << std::endl;
      logger_.info("Nominal quadrotor model");
      logger_ << agi_quad_nominals_.back() << std::endl;
    }

    // get maximum values for control signal normalization
    const Scalar mass = agi_quad_nominals_.back().m_;
    std::vector<std::string> temp_derated_thrust_max;
    bool derated = node["environment"]["derated_thrust_max"].getIfDefined(
      temp_derated_thrust_max);

    if (temp_derated_thrust_max.size() == 0) {
      std::string derated_thrust_max;
      derated = node["environment"]["derated_thrust_max"].getIfDefined(
        derated_thrust_max);
      temp_derated_thrust_max.emplace_back(derated_thrust_max);
    }

    // convert to vector of Scalars
    derated_thrust_max_.push_back(std::stof(temp_derated_thrust_max[i]));

    if (derated && temp_derated_thrust_max.size() != num_agents_) {
      logger_.error("Derated thrust max is not set for all agents.");
      return false;
    }

    const Scalar quad_max_force =
      (derated ? derated_thrust_max_[i]
               : agi_quad_nominals_.back().thrust_max_) *
      4;
    const Vector<3> max_omega = agi_quad_nominals_.back().omega_max_;

    std::string action_normalization = "old";
    node["environment"]["action_normalization"].getIfDefined(
      action_normalization);

    // compute the mean and standard deviation of the action
    if (action_normalization == "old") {
      act_mean_.row(i) << quad_max_force / (2.0 * mass), 0.0, 0.0, 0.0;
      act_std_.row(i) << quad_max_force / (2.0 * mass), max_omega[0],
        max_omega[1], max_omega[2];
    } else if (action_normalization == "new") {
      act_mean_.row(i) << 9.81, 0.0, 0.0, 0.0;
      act_std_.row(i) << quad_max_force / mass - 9.81, max_omega[0],
        max_omega[1], max_omega[2];
    } else {
      logger_.fatal("Unknown action normalization %s",
                    action_normalization.c_str());
    }

    // compute min and  max control command
    act_min_.row(i) << 0.0, -max_omega[0], -max_omega[1], -max_omega[2];
    act_max_.row(i) << quad_max_force / mass, max_omega[0], max_omega[1],
      max_omega[2];

    logger_.info("Using %s quadrotor with max thrust %5.2f",
                 derated ? "derated" : "nominal", quad_max_force);

    // fill with default constructors, will be reset later
    agi_quad_state_.emplace_back(agi::QuadState{});
    prev_agi_quad_state_.emplace_back(agi::QuadState{});
    agi_cmd_.emplace_back(agi::Command{});
    prev_agi_cmd_.emplace_back(agi::Command{});
  }

  return true;
}

MarlVisionRacingBaseEnv::~MarlVisionRacingBaseEnv() {}

// TODO double check dimensions of act
bool MarlVisionRacingBaseEnv::stepSimulation(const Ref<const Vector<>> act) {
  if (!act.allFinite() ||
      static_cast<std::size_t>(act.rows()) != mvre::kNAct * num_agents_ ||
      sim_dt_ <= 0) {
    logger_.error("Step Failed.");
    return false;
  }

  for (int ag = 0; ag < static_cast<int>(num_agents_); ag++) {
    if (!agi_quad_state_[ag].valid()) {
      logger_.error("QuadState is not valid in Environment %i", env_id_);
      std::cerr << agi_quad_state_[ag] << std::endl;
    }

    if (test_env_ && crashed_agents_[ag]) {
      // crashed drone is on the floor
      agi_quad_state_[ag].p.z() = racetrack_->getWorldBox()(2, 0) + 0.1;
      agi_quad_state_[ag].v.setZero();
      continue;
    }

    if (test_env_ && race_finished_[ag]) {
      // finished drone is on the floor
      agi_quad_state_[ag].p.z() = racetrack_->getWorldBox()(2, 0) + 0.1;
      agi_quad_state_[ag].v.setZero();
      continue;
    }

    flight_mode_[ag] = mvre::kFlying;

    // compute actual control command
    // using temp variables to make compiler happy
    Vector<4> act_temp = act.segment<mvre::kNAct>(ag * mvre::kNAct);
    Vector<4> act_std_temp = act_std_.row(ag);
    Vector<4> act_mean_temp = act_mean_.row(ag);
    pi_act_.row(ag) = act_temp.cwiseProduct(act_std_temp) + act_mean_temp;

    // clip control action
    pi_act_.row(ag) =
      pi_act_.row(ag).cwiseMax(act_min_.row(ag)).cwiseMin(act_max_.row(ag));

    //  simulate agilicious quadrotor
    agi_cmd_[ag].collective_thrust = pi_act_.row(ag)[0];
    agi_cmd_[ag].omega = pi_act_.row(ag).segment<3>(1);

    if (!agi_cmd_[ag].valid()) {
      agi_cmd_[ag].t = 0;
      agi_cmd_[ag].collective_thrust = 9.81;
      agi_cmd_[ag].omega.setZero();
    }

    if (!agi_simulators_[ag]->run(agi_cmd_[ag], sim_dt_)) {
      logger_.error("Cannot run quadrotor simulator!");
      std::cout << "agent " << ag << " agi_cmd_[ag]: \n"
                << agi_cmd_[ag] << std::endl;
    }
    agi_simulators_[ag]->getState(&agi_quad_state_[ag]);

    agi_cmd_[ag].t += sim_dt_;
  }
  global_t_ += sim_dt_;

  return true;
}

bool MarlVisionRacingBaseEnv::step(const Ref<const Vector<>> act,
                                   Ref<Vector<>> obs, Ref<Vector<>> reward) {

  stepSimulation(act);

  getObs(obs);

  updateFlightmode();

  computeReward(reward);

  prev_agi_quad_state_ = agi_quad_state_;
  prev_agi_cmd_ = agi_cmd_;

  return true;
}

bool MarlVisionRacingBaseEnv::getQuadState(Ref<Vector<>> obs) const {
  for (std::size_t i = 0; i < num_agents_; i++) {
    if (agi_quad_state_[i].t >= 0.0 &&
        (static_cast<std::size_t>(obs.rows()) == agi::QS::SIZE * num_agents_)) {
      obs.segment<agi::QS::SIZE>(i * agi::QS::SIZE) = agi_quad_state_[i].x;
    } else {
      logger_.error("agi_quad_state_[%i].t = %f", i, agi_quad_state_[i].t);
      logger_.error("size of obs: %i", obs.rows());
      logger_.error("Cannot get quadrotor states");
      return false;
    }
  }
  return true;
}

bool MarlVisionRacingBaseEnv::applyRandomMass(const int agent_id) {
  bool valid = true;
  const Scalar mass =
    agi_quad_nominals_[agent_id].m_ *
    (1 + dynamics_randomization_ * uniform_dist_(rg_) * mass_rd_coeff_);
  agi::Quadrotor agi_quad_dynamics = agi_simulators_[agent_id]->getQuadrotor();
  agi_quad_dynamics.m_ = mass;
  valid &= agi_simulators_[agent_id]->updateQuad(agi_quad_dynamics);
  return valid;
}

bool MarlVisionRacingBaseEnv::applyRandomInertia(const int agent_id) {
  bool valid = true;
  const ArrayVector<3> rnd3 = ArrayVector<3>::NullaryExpr(
    [this]() { return this->uniform_dist_(this->rg_); });
  Vector<3> J = agi_quad_nominals_[agent_id].J_.diagonal().array() *
                (ArrayVector<3>::Ones() +
                 dynamics_randomization_ * rnd3 * inertia_rd_coeff_);
  agi::Quadrotor agi_quad_dynamics = agi_simulators_[agent_id]->getQuadrotor();
  agi_quad_dynamics.J_.diagonal() = J;
  agi_quad_dynamics.J_inv_ = agi_quad_dynamics.J_.inverse();
  valid &= agi_simulators_[agent_id]->updateQuad(agi_quad_dynamics);
  return valid;
}

bool MarlVisionRacingBaseEnv::applyRandomDrag(const int agent_id) {
  agi::Quadrotor agi_quad_dynamics = agi_simulators_[agent_id]->getQuadrotor();
  agi_quad_dynamics.scale_drag_ =
    agi_quad_nominals_[agent_id].scale_drag_ *
    (1 + dynamics_randomization_ * uniform_dist_(rg_) * drag_rd_coeff_);
  return agi_simulators_[agent_id]->updateQuad(agi_quad_dynamics);
}

bool MarlVisionRacingBaseEnv::applyRandomThrust(const int agent_id) {
  agi::Quadrotor agi_quad_dynamics = agi_simulators_[agent_id]->getQuadrotor();
  agi_quad_dynamics.scale_thrust_ =
    agi_quad_nominals_[agent_id].scale_thrust_ *
    (1 + dynamics_randomization_ * uniform_dist_(rg_) * thrust_rd_coeff_);
  agi_simulators_[agent_id]->updateQuad(agi_quad_dynamics);
  return true;
}

bool MarlVisionRacingBaseEnv::applyRandomTorque(const int agent_id) {
  agi::Quadrotor agi_quad_dynamics = agi_simulators_[agent_id]->getQuadrotor();
  agi_quad_dynamics.scale_torque_ =
    agi_quad_nominals_[agent_id].scale_torque_ *
    (1 + dynamics_randomization_ * uniform_dist_(rg_) * torque_rd_coeff_);
  agi_simulators_[agent_id]->updateQuad(agi_quad_dynamics);
  return true;
}

Scalar MarlVisionRacingBaseEnv::getRandomDelay() {
  return dynamics_randomization_ * (0.5 * (uniform_dist_(rg_) + 1)) *
         delay_rd_coeff_;
}

bool MarlVisionRacingBaseEnv::applyRandomState(float rand_pos, float rand_vel,
                                               float rand_att, float rand_ome,
                                               const int agent_id) {
  agi_quad_state_[agent_id].p +=
    (dynamics_randomization_ * Vector<3>::Random().array() *
     init_pos_rd_.array() * rand_pos)
      .matrix();

  const Vector<3> rpy = Vector<3>::Random().array() * init_att_rd_.array() *
                        rand_att * M_PI / 180.0;
  agi_quad_state_[agent_id].q((AngleAxis(rpy(0), Vector<3>::UnitX()) *
                               AngleAxis(rpy(1), Vector<3>::UnitY()) *
                               AngleAxis(rpy(2), Vector<3>::UnitZ())) *
                              agi_quad_state_[agent_id].q());
  agi_quad_state_[agent_id].v +=
    (dynamics_randomization_ * Vector<3>::Random().array() *
     init_vel_rd_.array() * rand_vel)
      .matrix();
  agi_quad_state_[agent_id].w +=
    (dynamics_randomization_ * Vector<3>::Random().array() *
     init_ome_rd_.array() * rand_ome)
      .matrix() *
    M_PI / 180.0;
  agi_simulators_[agent_id]->setState(agi_quad_state_[agent_id]);
  return true;
}

bool MarlVisionRacingBaseEnv::curriculumGateSize() {
  if (gate_curr_updates_.size() == 0) {
    return false;
  }

  // Interpolate the correct gate size
  Scalar interpolated_size = 1.0;
  if (curriculum_updates_ <= gate_curr_updates_[0]) {
    interpolated_size = gate_curr_sizes_[0];
  } else if (curriculum_updates_ >=
             gate_curr_updates_[gate_curr_updates_.size() - 1]) {
    interpolated_size = gate_curr_sizes_[gate_curr_updates_.size() - 1];
  } else {
    for (long i = 0; i < gate_curr_updates_.size() - 1; ++i) {
      if (curriculum_updates_ >= gate_curr_updates_[i] &&
          curriculum_updates_ < gate_curr_updates_[i + 1]) {
        interpolated_size =
          gate_curr_sizes_[i] +
          (gate_curr_sizes_[i + 1] - gate_curr_sizes_[i]) *
            (curriculum_updates_ - gate_curr_updates_[i]) /
            (gate_curr_updates_[i + 1] - gate_curr_updates_[i]);
        break;
      }
    }
  }
  racetrack_->setGateSafetyDistRelative(interpolated_size);
  return true;
}

bool MarlVisionRacingBaseEnv::curriculumStartPosProb() {
  if (start_pos_prob_curr_updates_.size() == 0) {
    return false;
  }

  // Interpolate the correct start position probability
  Scalar interpolated_prob = 0.0;
  if (curriculum_updates_ <= start_pos_prob_curr_updates_[0]) {
    interpolated_prob = start_pos_prob_curr_values_[0];
  } else if (curriculum_updates_ >=
             start_pos_prob_curr_updates_[start_pos_prob_curr_updates_.size() -
                                          1]) {
    interpolated_prob =
      start_pos_prob_curr_values_[start_pos_prob_curr_values_.size() - 1];
  } else {
    for (long i = 0; i < start_pos_prob_curr_updates_.size() - 1; ++i) {
      if (curriculum_updates_ >= start_pos_prob_curr_updates_[i] &&
          curriculum_updates_ < start_pos_prob_curr_updates_[i + 1]) {
        interpolated_prob =
          start_pos_prob_curr_values_[i] +
          (start_pos_prob_curr_values_[i + 1] -
           start_pos_prob_curr_values_[i]) *
            (curriculum_updates_ - start_pos_prob_curr_updates_[i]) /
            (start_pos_prob_curr_updates_[i + 1] -
             start_pos_prob_curr_updates_[i]);
        break;
      }
    }
  }
  reset_start_prob_ = interpolated_prob;
  return true;
}

}  // namespace flightlib
