#pragma once

#include "agilib/types/quad_state.hpp"
#include "agilib/math/types.hpp"

using namespace agi;

namespace flightlib {

class Gate {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 public:
  Gate(const Vector<3>& pos, const Quaternion& att,
       const Scalar& gate_size = 1.45, const Scalar& gate_thickness = 0.275,
       const Scalar& dist_safe = 0.0);

  Vector<3> toGateFrame(const Ref<const Vector<3>>) const;

  void setSafetyDistRelative(const Scalar dist_safe);
  bool hasPassed(const agi::QuadState& pos, const float sim_dt = 0.02,
                 float* const l_inf_distance = nullptr) const;
  bool inCollision(const agi::QuadState& pos, const float sim_dt = 0.02) const;

  Scalar centerDistance(const agi::QuadState& state) const;

  Vector<3> position() const;
  Quaternion attitude() const;

  Matrix<4, 3> corners() const;

  Scalar safeDist() const;
  Scalar gateDim() const;
  Scalar gateThick() const;

  void setGateSize(const Scalar size);

 private:
  Vector<3> pos_{NAN, NAN, NAN};
  Quaternion att_{NAN, NAN, NAN, NAN};
  Quaternion att_inv_{NAN, NAN, NAN, NAN};
  Scalar gate_size_ = NAN;    // size of the free space in the gate
  Scalar gate_thick_ = NAN;   // thickness of the rim (27.5cm)
  Scalar dist_safe_ = NAN;    // additional safety distance

  // determines the gate order as
  // TL - TR - BR - BL
  const Matrix<4, 3> corners_ =
    (Matrix<4, 3>() << 0, 1, 1, 0, -1, 1, 0, -1, -1, 0, 1, -1).finished();

};

}  // namespace flightlib
