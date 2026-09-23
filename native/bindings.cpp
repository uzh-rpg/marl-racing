#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "flightlib/envs/vision_racing_env/marl_no_camera_vec_env.hpp"
namespace py = pybind11;
using namespace flightlib;
PYBIND11_MODULE(flightgym, m) {
  py::class_<MarlNoCameraVecEnv>(m, "MarlNoCameraVecEnv_v0")
    .def(py::init<const std::string&>())
    .def("reset",
         static_cast<bool (MarlNoCameraVecEnv::*)(Ref<MatrixRowMajor<>>)>(
           &MarlNoCameraVecEnv::reset),
         "reset")
    .def("step", &MarlNoCameraVecEnv::step)
    .def("setSeed", &MarlNoCameraVecEnv::setSeed)
    .def("curriculumUpdate", static_cast<void (MarlNoCameraVecEnv::*)()>(
                               &MarlNoCameraVecEnv::curriculumUpdate))
    .def("curriculumUpdate", static_cast<void (MarlNoCameraVecEnv::*)(int)>(
                               &MarlNoCameraVecEnv::curriculumUpdate))
    .def("setTestMode", &MarlNoCameraVecEnv::setTestMode)
    .def("setTestModeOffset", &MarlNoCameraVecEnv::setTestModeOffset)
    .def("setTestModeOffsetAgent", &MarlNoCameraVecEnv::setTestModeOffsetAgent)
    .def("getQuadState", &MarlNoCameraVecEnv::getQuadState)
    .def("getNumOfEnvs", &MarlNoCameraVecEnv::getNumOfEnvs)
    .def("getObsDim", &MarlNoCameraVecEnv::getObsDim)
    .def("getActDim", &MarlNoCameraVecEnv::getActDim)
    .def("getRewDim", &MarlNoCameraVecEnv::getRewDim)
    .def("getRewardNames", &MarlNoCameraVecEnv::getRewardNames)
    .def("getMaxNumAgents", &MarlNoCameraVecEnv::getMaxNumAgents)
    .def("getOtherAgentObsDim", &MarlNoCameraVecEnv::getOtherAgentObsDim)
    .def("getExtraInfoNames", &MarlNoCameraVecEnv::getExtraInfoNames)
    .def("__repr__", [](const MarlNoCameraVecEnv& a) {
      return "RPG State-based Multi-Agent Drone Vectorized Racing Environment.";
    });
}
