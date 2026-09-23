//
// This is inspired by RaiGym, thanks.
// https://raisim.com/
//
#pragma once

// std
#include <memory>

// openmp
#include <omp.h>

#include <agilib/utils/yaml.hpp>

// flightlib
#include "agilib/math/types.hpp"
#include "agilib/utils/logger.hpp"
#include "flightlib/envs/env_base.hpp"

// Template stuff to make the compiler happy
#include "flightlib/envs/vision_racing_env/marl_no_camera_env.hpp"

namespace flightlib {

template<typename EnvBaseName>
class VecEnvBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VecEnvBase(const std::string& node_file);
  virtual ~VecEnvBase() = 0;

  virtual bool load(const agi::Yaml& node);

  // - public OpenAI-gym style functions for vectorized environment
  virtual bool reset(Ref<MatrixRowMajor<>> obs);
  virtual bool step(const Ref<const MatrixRowMajor<>> act,
                    Ref<MatrixRowMajor<>> obs, Ref<MatrixRowMajor<>> reward,
                    Ref<BoolVector<>> done, Ref<MatrixRowMajor<>> extra_info);

  // public set functions
  virtual void setSeed(const int seed);

  // public get functions

  // - auxiliary functions
  virtual void curriculumUpdate();
  void curriculumUpdate(int iteration);

  // public functions
  bool getQuadState(Ref<MatrixRowMajor<>> quadstate) const;
  inline int getObsDim(void) const { return obs_dim_; };
  inline int getActDim(void) const { return act_dim_; };
  inline int getRewDim(void) const { return rew_dim_; };
  inline std::vector<std::string>& getExtraInfoNames() {
    return extra_info_names_;
  };
  inline int getNumOfEnvs(void) const { return envs_.size(); };
  inline std::vector<std::string> getRewardNames(void) {
    return this->envs_[0]->getRewardNames();
  };

 protected:
  // create objects
  Logger logger_{"VecEnvBase"};
  std::vector<std::unique_ptr<EnvBaseName>> envs_;
  std::vector<std::string> extra_info_names_;

  // auxiliary variables
  int seed_;
  int num_envs_;
  int obs_dim_;
  int act_dim_;
  int rew_dim_;
  int num_threads_;

};

}  // namespace flightlib
