#include "optimizers.hpp"
#include <core/graph/graph_framework.hpp>

Optimizer::Optimizer() { this->optimizer_graph = new Graph(); }

void Optimizer::executeOptimizer() { this->optimizer_graph->compute(); }