// C++ Headers
#include <atomic>

// Library Headers
#include "layers.hpp"

namespace {
std::atomic<tf::layer::layer_id> next_layer_id{1};
} // namespace

Layer::Layer()
    : id_(next_layer_id.fetch_add(1, std::memory_order_relaxed)) {}
