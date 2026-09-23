#pragma once

#include <filesystem>

namespace flightlib {

bool file_exists(const std::string& s);
bool file_exists(const std::filesystem::path& p);

}  // namespace flightlib
