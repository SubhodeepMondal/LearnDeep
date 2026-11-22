#include "loss.hpp"

#include <atomic>

namespace {
std::atomic<tf::loss::loss_id> next_loss_id{1};
} // namespace

Loss::Loss() : id_(next_loss_id.fetch_add(1, std::memory_order_relaxed)) {}
