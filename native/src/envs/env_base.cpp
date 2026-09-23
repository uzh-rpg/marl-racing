#include "flightlib/envs/env_base.hpp"

namespace flightlib {

EnvBase::~EnvBase() {}

void EnvBase::resetCurriculumUpdatesCounter(int iteration) {
  curriculum_updates_ = iteration;
}

int EnvBase::getCurriculumUpdatesCounter() { return curriculum_updates_; }

}  // namespace flightlib
