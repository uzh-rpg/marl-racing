#include "flightlib/common/utils.hpp"

namespace flightlib {

bool file_exists(const std::string& s) {
  if (s.length() < 256)
    return file_exists(std::filesystem::path(s));
  else
    return false;
}

bool file_exists(const std::filesystem::path& p) {
  return (std::filesystem::exists(p));
}

}  // namespace flightlib
