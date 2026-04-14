#include "LinearAlgebraFixtures.unit.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>
void MathTest::SetUp() {}
void MathTest::TearDown() {}

std::vector<std::float64_t> load_bin(const std::string &path,
                                     size_t expected_size) {
  std::ifstream file(path, std::ios::binary);

  if (!file) {
    throw std::runtime_error("Failed to open file: " + path);
  }

  std::vector<std::float64_t> data(expected_size);

  file.read(reinterpret_cast<char *>(data.data()),
            expected_size * sizeof(std::float64_t));

  if (!file) {
    throw std::runtime_error("Error reading file: " + path);
  }

  return data;
}