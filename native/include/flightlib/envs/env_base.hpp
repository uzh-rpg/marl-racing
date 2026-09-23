//
// This is inspired by RaiGym, thanks.
// https://raisim.com/
//
#pragma once

// standard library
#include <cmath>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "agilib/math/types.hpp"
#include "agilib/utils/file_utils.hpp"
#include "agilib/utils/yaml.hpp"

using namespace agi;

namespace flightlib {

class EnvBase {
 public:
  EnvBase(const int env_id) : env_id_(env_id) {};
  virtual ~EnvBase() = 0;

  // (pure virtual) public methods (has to be implemented by child classes)
  virtual bool load(const agi::Yaml& node) = 0;
  virtual bool reset(Ref<Vector<>> obs) = 0;
  virtual bool step(const Ref<const Vector<>> act, Ref<Vector<>> obs,
                    Ref<Vector<>> reward) = 0;
  virtual bool getObs(Ref<Vector<>> obs) = 0;
  virtual bool getQuadState(Ref<Vector<>> act) const = 0;

  virtual void curriculumUpdate() = 0;
  void resetCurriculumUpdatesCounter(int iteration = 0);
  int getCurriculumUpdatesCounter();
  virtual void updateExtraInfo() = 0;
  virtual bool isTerminalState(Scalar& reward) const = 0;

  // auxilirary functions
  inline void setSeed(const int seed) { rg_.seed(seed); };
  inline int getObsDim() const { return obs_dim_; };
  inline int getActDim() const { return act_dim_; };
  inline int getRewDim() const { return rew_dim_; };

  // public variables
  std::unordered_map<std::string, float> extra_info_;

 protected:
  int env_id_;
  agi::Yaml node_;

  // observation and action dimenstions (for Reinforcement learning)
  int obs_dim_;
  int act_dim_;
  int rew_dim_;

  // control time step
  Scalar sim_dt_;
  Scalar max_t_;

  size_t curriculum_updates_ = 0;

  // random variable generator
  std::uniform_real_distribution<Scalar> uniform_dist_{-1.0, 1.0};
  std::uniform_real_distribution<Scalar> uni_01_dist_{0.0, 1.0};
  std::random_device rd_;
  mutable std::mt19937 rg_{rd_()};
};

}  // namespace flightlib
