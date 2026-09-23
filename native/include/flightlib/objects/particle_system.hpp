#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <random>

#include "agilib/math/types.hpp"

using namespace agi;

class ParticleSystem {
 public:
  // Configuration constants
  static constexpr float ParticleLifespan = 100.0f;    // t_part in MATLAB
  static constexpr int NumberOfParticlesToSpawn = 96;  // n_per_step in MATLAB
  static constexpr int ParticleDimension = 8;          // d_part in MATLAB
  static constexpr float decayFactorWeight = 0.975f;
  static constexpr float decayFactorVel = 0.982f;
  static constexpr float linearWidthIncrease = 0.0025f;
  static constexpr float z0_factor = 3.0f;  // s0 = 3*l in MATLAB

  // Downwash model parameters (valid for offboard?)
  static constexpr float Bd = 10.11f;
  static constexpr float S = 0.07668f;
  static constexpr float s0 = -5.817f;
  static constexpr float rho = 1.204f;  // air density

  ParticleSystem()
    : currentIndex_(0), buffer_size_(0), rng_(std::random_device{}()){};
  explicit ParticleSystem(int num_agents)
    : currentIndex_(0), rng_(std::random_device{}()) {
    buffer_size_ = static_cast<int>(ParticleLifespan * num_agents *
                                    NumberOfParticlesToSpawn);
    particles_.resize(buffer_size_, ParticleDimension);
    particles_.setZero();
  }

  void spawnParticles(const Vector<3>& pos, const Matrix<3, 3>& rot,
                      const float thrust, const float motor_to_motor_distance,
                      const float prop_radius) {
    const int num_to_spawn = NumberOfParticlesToSpawn;

    // Calculate induced velocity (vi) from thrust
    const float A_prop = M_PI * prop_radius * prop_radius;
    const float vi = std::sqrt(thrust / (2.0f * rho * A_prop * 4.0f));

    // Initial sampling distance (s0 = 3*l in MATLAB)
    const float s0_distance = z0_factor * motor_to_motor_distance;

    // Create radial distribution
    const int num_radial = 8;
    const float max_radius = std::sqrt(3.0f * motor_to_motor_distance / 4.0f);

    // Uniform distribution for perturbations
    std::uniform_real_distribution<float> uniform_dist(-0.5f, 0.5f);

    for (int i = 0; i < num_to_spawn; ++i) {
      const int idx = (currentIndex_ + i) % buffer_size_;

      // Radial and angular distribution
      int radial_idx = i % num_radial;
      int angular_idx = i / num_radial;
      int angular_divisions = num_to_spawn / num_radial;

      // Squared radial distribution
      float r_base =
        (static_cast<float>(radial_idx) / (num_radial - 1)) * max_radius;
      r_base = r_base * r_base / max_radius;  // Square the distribution

      // Add perturbation
      float r = r_base + 0.05f * uniform_dist(rng_);

      // Angular distribution with perturbation
      float phi_base =
        static_cast<float>(angular_idx) / angular_divisions * 2.0f * M_PI;
      float phi = phi_base + 0.25f * uniform_dist(rng_);

      // Position in body frame (matching MATLAB)
      float x_local = r * std::cos(phi + 5.0f * r);
      float y_local = r * std::sin(phi + 5.0f * r);
      float z_local = -motor_to_motor_distance / 4.0f;  // z = -l/4 in MATLAB

      Vector<3> local_pos(x_local, y_local, z_local);

      // Transform to world frame
      Vector<3> world_pos = pos + rot * local_pos;
      particles_.row(idx).head<3>() = world_pos;

      // Calculate downwash velocity
      float vz =
        calculateDownwashVelocity(vi, s0_distance, r, motor_to_motor_distance);

      vz = std::min(vz, 1.25f * vi);

      // Calculate r_12_tilde for conical velocity
      float s_tilde = s0_distance / motor_to_motor_distance;
      float r_12_tilde_val = S * (s_tilde - s0);

      // Conical velocity components
      float vx_local = S * vz * std::cos(phi + 5.0f * r) * 0.875f *
                       (r / r_12_tilde_val + 0.15f);
      float vy_local = S * vz * std::sin(phi + 5.0f * r) * 0.875f *
                       (r / r_12_tilde_val + 0.15f);

      // Add random perturbation
      vx_local += 0.05f * uniform_dist(rng_);
      vy_local += 0.05f * uniform_dist(rng_);
      vz += 0.05f * uniform_dist(rng_);

      // Transform velocity to world frame (note: -vz for downward)
      Vector<3> local_vel(vx_local, vy_local, -vz);
      Vector<3> world_vel = rot * local_vel;

      particles_.row(idx).segment<3>(3) = world_vel;

      // Initialize width and weight
      particles_.row(idx)(6) = motor_to_motor_distance / 8.0f;  // l/8
      particles_.row(idx)(7) = 1.0f;
    }

    currentIndex_ = (currentIndex_ + num_to_spawn) % buffer_size_;
  }

  void update(float dt) {
    // Update position: pos += vel * dt
    particles_.leftCols<3>() += particles_.middleCols<3>(3) * dt;

    // Update velocity: vel *= decayFactorVel
    particles_.middleCols<3>(3).array() *= decayFactorVel;

    // Update width: width += linearWidthIncrease
    particles_.col(6).array() += linearWidthIncrease;

    // Update weight: weight *= decayFactorWeight
    particles_.col(7).array() *= decayFactorWeight;
  }

  Vector<3> evaluateDownwash(const Vector<3>& point) const {
    // Calculate squared distances from point to all particles
    Eigen::ArrayXd distSq =
      (particles_.leftCols<3>().rowwise() - point.transpose().array())
        .square()
        .rowwise()
        .sum();

    // Calculate influence using particle widths and weights
    Eigen::ArrayXd sigma_sq = 2.0 * particles_.col(6).array().square();
    Eigen::ArrayXd influence =
      (-distSq / sigma_sq).exp() * particles_.col(7).array();

    // Calculate weighted velocity sum
    double influence_sum = influence.sum() + 1e-2;

    Vector<3> weighted_velocity =
      (particles_.middleCols<3>(3).array().colwise() * influence)
        .colwise()
        .sum()
        .transpose() /
      influence_sum;

    return weighted_velocity;
  }

 private:
  Array<Eigen::Dynamic, ParticleDimension> particles_;
  int currentIndex_;
  int buffer_size_;
  std::mt19937 rng_;  // Random number generator for perturbations

  // Downwash velocity
  float calculateDownwashVelocity(float vi, float s, float r,
                                  float motor_to_motor_distance) const {
    float s_tilde = s / motor_to_motor_distance;
    float r_tilde = r / motor_to_motor_distance;

    // r_12_tilde function
    float r_12_tilde_val = S * (s_tilde - s0);

    // centerline_profile function
    float centerline = Bd / (s_tilde - s0);

    // radial_profile function
    float zeta_val = r_tilde / r_12_tilde_val;
    float radial =
      1.0f /
      std::pow(1.0f + (std::sqrt(2.0f) - 1.0f) * zeta_val * zeta_val, 2.0f);

    return vi * centerline * radial;
  }
};