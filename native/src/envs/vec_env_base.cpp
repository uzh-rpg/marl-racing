#include "flightlib/envs/vec_env_base.hpp"

namespace flightlib {

template<typename EnvBaseName>
VecEnvBase<EnvBaseName>::VecEnvBase(const std::string& cfg) {
  if (!(file_exists(cfg))) {
    load(agi::Yaml(cfg));
    std::cout << "Loading Envirovronment from File" << std::endl;
  } else {
    logger_.warn("Loading YAML from file %s", cfg.c_str());
    load(agi::Yaml(fs::path(cfg)));
  }
}

template<typename EnvBaseName>
bool VecEnvBase<EnvBaseName>::load(const agi::Yaml& node) {
  if (!node["main"]["seed"].isDefined() ||
      !node["main"]["num_envs"].isDefined() ||
      !node["main"]["num_threads"].isDefined()) {
    logger_.warn("Cannot load main configurations. Using default parameters.");
    seed_ = 0;
    num_envs_ = 1;
    num_threads_ = 1;
  } else {
    logger_.info("Load main configuration.");
    seed_ = node["main"]["seed"].as<int>();
    num_envs_ = node["main"]["num_envs"].as<int>();
    num_threads_ = node["main"]["num_threads"].as<int>();

    // set threads
    if (num_threads_ >= 0) omp_set_num_threads(num_threads_);

    // create & setup environments
    for (int env_id = 0; env_id < num_envs_; env_id++) {
      envs_.push_back(std::make_unique<EnvBaseName>(node, env_id));
    }

    obs_dim_ = envs_[0]->getObsDim();
    act_dim_ = envs_[0]->getActDim();
    rew_dim_ = envs_[0]->getRewDim();

    // generate reward names
    // compute it once to get reward names. actual value is not used
    envs_[0]->updateExtraInfo();
    for (auto& re : envs_[0]->extra_info_) {
      extra_info_names_.push_back(re.first);
    }
    logger_.info("%d vectorized environments created. ", num_envs_);
    std::cout << "Vectorized Environment:\n"
              << "obs dim    =            [" << obs_dim_ << "]\n"
              << "act dim    =            [" << act_dim_ << "]\n"
              << "rew dim    =            [" << rew_dim_ << "]\n"
              << "einfo dim  =            [" << envs_[0]->extra_info_.size()
              << "]\n"
              << "num_envs   =            [" << num_envs_ << "]\n"
              << "num_thread =            [" << num_threads_ << "]\n"
              << "seed       =            [" << seed_ << "]\n";
  }
  return true;
}

template<typename EnvBaseName>
VecEnvBase<EnvBaseName>::~VecEnvBase() {}

template<typename EnvBaseName>
bool VecEnvBase<EnvBaseName>::reset(Ref<MatrixRowMajor<>> obs) {

  bool valid = true;
#pragma omp parallel for schedule(dynamic) reduction(& : valid)
  for (int i = 0; i < num_envs_; i++) {
    valid &= envs_[i]->reset(obs.row(i));
  }

  return valid;
}

template<typename EnvBaseName>
bool VecEnvBase<EnvBaseName>::step(const Ref<const MatrixRowMajor<>> act,
                                   Ref<MatrixRowMajor<>> obs,
                                   Ref<MatrixRowMajor<>> reward,
                                   Ref<BoolVector<>> done,
                                   Ref<MatrixRowMajor<>> extra_info) {
  
  bool valid = true;
#pragma omp parallel for schedule(dynamic) reduction(& : valid)
  for (int i = 0; i < num_envs_; i++) {
    // get individual rewards
    valid &= envs_[i]->step(act.row(i), obs.row(i), reward.row(i));

    Scalar terminal_reward = 0;
    done[i] = envs_[i]->isTerminalState(terminal_reward);

    envs_[i]->updateExtraInfo();
    for (int j = 0; j < extra_info.cols(); j++)
      extra_info(i, j) = envs_[i]->extra_info_[extra_info_names_[j]];

    if (done[i]) {
      valid &= envs_[i]->reset(obs.row(i));
      reward(i, reward.cols() - 1) = terminal_reward;
    }
  }
  // Check if anything contains nans
  if (!obs.allFinite()) {
    logger_.error("Observation contains Inf or Nan");
  }
  if (!reward.allFinite()) {
    logger_.error("Reward contains Inf or Nan");
  }
  if (!act.allFinite()) {
    logger_.error("Actions contains Inf or Nan");
  }
  return valid;
}

template<typename EnvBaseName>
void VecEnvBase<EnvBaseName>::setSeed(const int seed) {
  int seed_inc = seed;
  for (int i = 0; i < num_envs_; i++) envs_[i]->setSeed(seed_inc++);
}

template<typename EnvBaseName>
bool VecEnvBase<EnvBaseName>::getQuadState(
  Ref<MatrixRowMajor<>> quad_state) const {
  bool valid = true;
#pragma omp parallel for schedule(dynamic) reduction(& : valid)
  for (int i = 0; i < this->num_envs_; i++) {
    valid &= this->envs_[i]->getQuadState(quad_state.row(i));
  }
  return valid;
}

template<typename EnvBaseName>
void VecEnvBase<EnvBaseName>::curriculumUpdate(void) {
  for (int i = 0; i < num_envs_; i++) envs_[i]->curriculumUpdate();
}

template<typename EnvBaseName>
void VecEnvBase<EnvBaseName>::curriculumUpdate(int iteration) {
  for (int i = 0; i < num_envs_; i++) {
    for (int j = envs_[i]->getCurriculumUpdatesCounter(); j < iteration; j++) {
      envs_[i]->curriculumUpdate();
    }
    envs_[i]->resetCurriculumUpdatesCounter(iteration);
  }
}

// IMPORTANT. Otherwise
// Linker errors because of the separation between the
// declaration and definition of the template class.
// Segmentation fault (core dumped)!
template class VecEnvBase<MarlNoCameraEnv>;

}  // namespace flightlib
