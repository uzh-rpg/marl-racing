#pragma once

// yaml cpp
#include <agilib/utils/yaml.hpp>

#include "flightlib/envs/vec_env_base.hpp"
#include "flightlib/envs/vision_racing_env/marl_no_camera_env.hpp"

namespace flightlib {

class MarlNoCameraVecEnv : public VecEnvBase<MarlNoCameraEnv> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MarlNoCameraVecEnv(const std::string& cfg);
  ~MarlNoCameraVecEnv();

  void setTestMode(bool test_mode);
  void setTestModeOffset(bool test_mode,
                         const Ref<MatrixRowMajor<>> pos_offset);
  void setTestModeOffsetAgent(bool test_mode,
                              const Ref<MatrixRowMajor<>> pos_offset, int id);

  inline int getMaxNumAgents() const {
    return this->envs_[0]->getMaxNumAgents();
  }
  inline int getOtherAgentObsDim() const {
    return this->envs_[0]->getOtherAgentObsDim();
  }
};

}  // namespace flightlib
