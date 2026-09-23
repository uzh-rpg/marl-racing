#pragma once

// std lib
#include <random>

// yaml cpp
#include <agilib/utils/file_utils.hpp>
#include <agilib/utils/yaml.hpp>

// flightlib
#include "flightlib/common/utils.hpp"
#include "flightlib/envs/env_base.hpp"
#include "flightlib/objects/racetrack.hpp"
#include "flightlib/objects/simple_gate.hpp"

// agilicious
#include "agilib/math/types.hpp"
#include "agilib/simulator/quadrotor_simulator.hpp"
#include "agilib/simulator/simulator_params.hpp"
#include "agilib/types/command.hpp"
#include "agilib/types/quad_state.hpp"
#include "agilib/types/quadrotor.hpp"

namespace flightlib {

namespace marlvisionracingenv {
// this is the racing state for one single drone
enum RacingState : int {
  // action dimension
  kNAct = 4,

  // flight modes
  kEpisodeDone = -1,
  kFlying = 0,
  kGatePassed = 1,
  kCrashWorld = 2,
  kCrashGate = 3,
  kCrashOtherAgent = 4,
  kFinishedRace = 5,
};

}  // namespace marlvisionracingenv

namespace mvre = marlvisionracingenv;
class MarlVisionRacingBaseEnv : public EnvBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MarlVisionRacingBaseEnv(const agi::Yaml &node, const int env_id);
  virtual ~MarlVisionRacingBaseEnv();

  // - public OpenAI-gym-style functionst
  // The step function is provided in the base class. It executes
  // 1. stepSimulation(act) [should not be changed]
  // 2. getObs(obs)
  // 3. updateFlightmode()
  // 4. computeReward(reward)
  // An inherited function should only implement these function,
  // not step itself.
  bool step(const Ref<const Vector<>> act, Ref<Vector<>> obs,
            Ref<Vector<>> reward) override final;
  bool stepSimulation(const Ref<const Vector<>> act);
  virtual bool getObs(Ref<Vector<>> obs) override = 0;
  virtual bool computeReward(Ref<Vector<>> reward) = 0;
  virtual bool updateFlightmode() = 0;
  virtual bool isTerminalState(Scalar &reward) const override = 0;
  virtual bool curriculumGateSize();
  virtual bool curriculumStartPosProb();

  // Any inherited function must provide its own implementation
  virtual bool reset(Ref<Vector<>> obs) override = 0;
  virtual void curriculumUpdate() override = 0;
  virtual void updateExtraInfo(void) override = 0;

  // Any inherited function can provide its own implementation
  // Environment Loading: the idea is to keep it modular and a user can
  // add custom loading functions by first calling the parent "load"
  // in their own load Environment.
  virtual bool load(const agi::Yaml &env_cfg) override;
  virtual bool loadEnvironment(const agi::Yaml &env_cfg);
  virtual bool loadQuadrotor(const agi::Yaml &node);
  virtual bool loadRacetrack(const agi::Yaml &node);
  virtual bool loadDynamicsRandomization(const agi::Yaml &node);
  virtual bool loadCurriculum(const agi::Yaml &node);
  virtual bool loadRewardComponents(const agi::Yaml &node);
  virtual bool loadExtraInfo(const agi::Yaml &env_cfg);

  // Any inherited function can provide its own implementation
  // Domain Randomization
  virtual bool applyRandomMass(int agent_id);
  virtual bool applyRandomInertia(int agent_id);
  virtual bool applyRandomDrag(int agent_id);
  virtual bool applyRandomThrust(int agent_id);
  virtual bool applyRandomTorque(int agent_id);
  virtual Scalar getRandomDelay();
  virtual bool applyRandomState(float rand_pos = 1.0, float rand_vel = 1.0,
                                float rand_att = 1.0, float rand_ome = 1.0,
                                const int agent_id = 0);

  // The inherited classes should use the provided implementation
  bool getQuadState(Ref<Vector<>> state) const override final;

  // MARL methods
  bool loadNumAgents(const agi::Yaml &node);

  const std::vector<std::string> &getRewardNames() const {
    return reward_names_;
  }

 protected:
  std::size_t num_agents_ = 1;
  int single_obs_dim_ = 1;

  bool yamlCheck(const bool valid, const std::string &category,
                 const std::string &key) const;
  std::shared_ptr<RaceTrack> racetrack_;

  Logger logger_{"MarlVisionRacingBaseEnv"};

  std::vector<std::shared_ptr<agi::QuadrotorSimulator>> agi_simulators_;
  std::vector<agi::Quadrotor> agi_quad_nominals_;

  Eigen::Matrix<int, Eigen::Dynamic, 1> flight_mode_;
  Eigen::Matrix<int, Eigen::Dynamic, 1> num_passed_gates_;
  Eigen::Matrix<bool, Eigen::Dynamic, 1> crashed_agents_;
  bool test_env_{false};
  int max_laps_{3};
  Eigen::Matrix<bool, Eigen::Dynamic, 1> race_finished_;

  // quadrotor states and control command
  std::vector<Scalar> derated_thrust_max_;
  std::vector<agi::QuadState> agi_quad_state_;
  std::vector<agi::QuadState> prev_agi_quad_state_;
  std::vector<agi::Command> agi_cmd_;
  std::vector<agi::Command> prev_agi_cmd_;
  Scalar global_t_{0.0};

  // robot observations and actions
  Matrix<Eigen::Dynamic, mvre::kNAct> pi_act_;
  Matrix<Eigen::Dynamic, mvre::kNAct> act_mean_;
  Matrix<Eigen::Dynamic, mvre::kNAct> act_std_;
  Matrix<Eigen::Dynamic, mvre::kNAct> act_min_;
  Matrix<Eigen::Dynamic, mvre::kNAct> act_max_;

  // rewards
  Scalar progress_coeff_;
  Scalar state_omega_coeff_;
  Scalar collision_coeff_;
  std::vector<std::string> reward_names_;

  // curriculum
  Vector<> gate_curr_updates_;
  Vector<> gate_curr_sizes_;
  Scalar reset_start_prob_ = 0.25;  // default value
  Vector<> start_pos_prob_curr_updates_;
  Vector<> start_pos_prob_curr_values_;

  // Domain Randomization
  Scalar mass_rd_coeff_{0.0};
  Scalar drag_rd_coeff_{0.0};
  Scalar thrust_rd_coeff_{0.0};
  Scalar torque_rd_coeff_{0.0};
  Scalar inertia_rd_coeff_{0.0};
  Scalar delay_rd_coeff_{0.0};
  Scalar dynamics_randomization_{1.0};
  Vector<3> init_pos_rd_{0, 0, 0};
  Vector<3> init_vel_rd_{0, 0, 0};
  Vector<3> init_att_rd_{0, 0, 0};
  Vector<3> init_ome_rd_{0, 0, 0};
};

}  // namespace flightlib
