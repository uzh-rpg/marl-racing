#include "flightlib/objects/simple_gate.hpp"

#include "agilib/math/math.hpp"

namespace flightlib {

Gate::Gate(const Vector<3>& gate_pos, const Quaternion& gate_att,
           const Scalar& gate_size, const Scalar& gate_thickness,
           const Scalar& dist_safe)
  : pos_(gate_pos),
    att_(gate_att),
    att_inv_(att_.inverse()),
    gate_size_(gate_size),
    gate_thick_(gate_thickness),
    dist_safe_(dist_safe) {}

Vector<3> Gate::toGateFrame(const Ref<const Vector<3>> pos) const {
  return att_inv_ * (pos - position());
}

bool Gate::hasPassed(const agi::QuadState& state, const float sim_dt,
                     float* const l_inf_distance) const {
  // sets the l_inf_distance value only if a gate has been passed!

  // transform the point to the gate frame
  Vector<3> p_G = toGateFrame(state.p);
  const Vector<3> v_G = att_inv_ * state.v;

  const bool wrong_direction = (v_G.x() < 0.0);

  // Fancy version that takes the correct sim_dt into account!
  const float intersect_dt = -p_G.x() / v_G.x();
  if (intersect_dt <= 0 && intersect_dt >= -2 * sim_dt) {
    const ArrayVector<3> intersect_point = p_G + v_G * intersect_dt;
    const float l_inf_dist =
      intersect_point.segment<2>(1).array().abs().maxCoeff();
    if (l_inf_distance != nullptr) {
      *l_inf_distance = l_inf_dist;
    }
    if (l_inf_dist < 0.5 * gate_size_ - dist_safe_ && !wrong_direction) {
      return true;
    } else {
      return false;
    }
  }
  return false;
}

bool Gate::inCollision(const agi::QuadState& state, const float sim_dt) const {
  // transform the point to the gate frame
  Vector<3> p_G = toGateFrame(state.p);
  const Vector<3> v_G = att_inv_ * state.v;

  // Fancy version that takes the correct sim_dt into account!
  const float intersect_dt = -p_G.x() / v_G.x();
  if (intersect_dt <= 0 && intersect_dt >= - 2*sim_dt) {
    const ArrayVector<3> intersect_point = p_G + v_G * intersect_dt;
    const bool inside_outer_frame =
      (intersect_point.segment<2>(1).array().abs().maxCoeff() <
       0.5 * gate_size_ + gate_thick_);
    const bool outside_inner_frame =
      (intersect_point.segment<2>(1).array().abs().maxCoeff() >
       0.5 * gate_size_ - dist_safe_);
    const bool wrong_direction = (v_G.x() < 0.0);
    return (inside_outer_frame && (outside_inner_frame || wrong_direction));
  }
  return false;
}

void Gate::setSafetyDistRelative(const Scalar dist_safe) {
  // For a gate with free space (size) of G, the new free space
  // for collision checks is dist_safe*G
  dist_safe_ = 0.5 * (1 - std::max(dist_safe, 0.0)) * gate_size_;
}

Scalar Gate::centerDistance(const agi::QuadState& state) const {
  return (state.p - pos_).norm();
}

Matrix<4, 3> Gate::corners() const {
  Matrix<4, 3> corners;
  for (int i = 0; i < 4; ++i) {
    corners.row(i) = position() + attitude() * corners_.row(i) * gate_size_ *
                                    0.5;
  }
  return corners;
}

Vector<3> Gate::position() const { return pos_; }

Quaternion Gate::attitude() const { return att_; }

Scalar Gate::safeDist() const { return dist_safe_; }

Scalar Gate::gateDim() const { return gate_size_; }

Scalar Gate::gateThick() const { return gate_thick_; }

void Gate::setGateSize(const Scalar size) {
  gate_size_ = size;
}

}  // namespace flightlib
