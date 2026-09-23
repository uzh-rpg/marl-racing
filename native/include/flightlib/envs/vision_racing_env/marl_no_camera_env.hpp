#pragma once

#include "agilib/math/types.hpp"
#include "flightlib/envs/vision_racing_env/marl_vision_racing_env_base.hpp"

namespace flightlib {

namespace marlnocameraenv {
enum MarlNoCameraState : int {
  // indices
  kObsRelCorner = 0,
  kObsCornerDist = 12,
  kObsQuadOri = 24,
  kObsQuadVel = 33,
  kObsOtherAgent = 36,

  // dimensions
  kNObs = 36,
  kNObsRelCorner = 12,
  kNObsCornerDist = 12,
  kNObsQuadOri = 9,
  kNObsQuadVel = 3,

  // observation of other agents per agent,
  // maximum number of agents defined in base env (10)
  // default observation includes 10 agents but sets unused agents to zero
  maxNumOpponents = 10,
  kNObsOtherAgent = 6,
  kNObsOtherAgentRelPos = 3,
  kNObsOtherAgentRelVel = 3,

  // index within other agent obs
  kObsOtherAgentRelPos = 0,
  kObsOtherAgentRelVel = 3,

};

}  // namespace marlnocameraenv

class MarlNoCameraEnv final : public MarlVisionRacingBaseEnv {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MarlNoCameraEnv(const agi::Yaml &cfg, const int env_id);

  bool loadCustom(const agi::Yaml &node);
  void applyPosOffset();
  void setTestMode(bool test_mode, const Vector<3> pos_offset);
  void setTestModeAgent(bool test_mode, const Vector<3> pos_offset, int id);
  bool reset(Ref<Vector<>> obs) override;
  bool reset();
  bool reset_single_agent(int agent_id);

  bool getObs(Ref<Vector<>> obs) override;
  bool computeReward(Ref<Vector<>> reward) override;
  bool updateFlightmode() override;
  bool isTerminalState(Scalar &reward) const override;

  void computeRelativeObs();
  void computeAgentRanking();
  void updateMinAgentDistance();

  // Any inherited function must provide its own implementation
  void curriculumUpdate() override;
  void updateExtraInfo(void) override;

  inline int getMaxNumAgents() const {
    return marlnocameraenv::maxNumOpponents;
  };
  inline int getOtherAgentObsDim() const {
    return marlnocameraenv::kNObsOtherAgent;
  };

  Scalar calcCurriculum(const Vector<> counters, const Vector<> values,
                        const Scalar query_counter) const;

 private:
  Matrix<Eigen::Dynamic, 3> pos_offset_agents_;

  bool all_done_ = false;

  Vector<> traversal_error_;
  Vector<> lap_time_;
  Vector<> last_lap_time_;
  Vector<> cumulative_lap_time_;
  Vector<> frozen_crash_distance_;

  Scalar smoothness_coeff_ = 1;

  Scalar gate_position_randomization_ = 0.0;
  Eigen::Matrix<int, Eigen::Dynamic, 1> starting_gate_num_;
  Eigen::Matrix<int, Eigen::Dynamic, 1> finished_laps_;

  Matrix<> rel_pos_;
  Matrix<> rel_vel_;
  Matrix<> rel_pos_norm_;
  Vector<> ranking_;
  Vector<> min_agent_distance_;

  Vector<> prop_radius_;  // offboard drone 0.1295 / 2.0

  Vector<> motor_to_motor_distances_;  // offboard drone 0.266

  // collision radius factor times motor_to_motor_distance / 2.0
  Vector<> agent_collision_radius_;

  Scalar random_keep_alive_{1.0};
  Scalar collision_radius_{0.2};
  Scalar agent_invisible_prob_{0.01};
  Scalar agent_invisible_prob_final_{0.01};

  Vector<> agents_visible_;

  Scalar ranking_lead_coeff_;
  Scalar opponent_collision_coeff_{-2.0};
  bool air_disturbance_modeling_{false};
};

}  // namespace flightlib
