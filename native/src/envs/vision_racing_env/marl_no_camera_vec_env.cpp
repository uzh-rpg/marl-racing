#include "flightlib/envs/vision_racing_env/marl_no_camera_vec_env.hpp"

namespace flightlib {

MarlNoCameraVecEnv::MarlNoCameraVecEnv(const std::string& cfg)
  : VecEnvBase(cfg) {}

MarlNoCameraVecEnv::~MarlNoCameraVecEnv() {}

void MarlNoCameraVecEnv::setTestMode(bool test_mode) {
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < this->num_envs_; i++) {
    this->envs_[i]->setTestMode(test_mode, Vector<3>::Zero());
  }
}

void MarlNoCameraVecEnv::setTestModeOffset(
  bool test_mode, const Ref<MatrixRowMajor<>> pos_offset) {
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < this->num_envs_; i++) {
    this->envs_[i]->setTestMode(test_mode, pos_offset.row(i));
  }
}

void MarlNoCameraVecEnv::setTestModeOffsetAgent(
  bool test_mode, const Ref<MatrixRowMajor<>> pos_offset, int id) {
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < this->num_envs_; i++) {
    this->envs_[i]->setTestModeAgent(test_mode, pos_offset.row(i), id);
  }
}

}  // namespace flightlib
