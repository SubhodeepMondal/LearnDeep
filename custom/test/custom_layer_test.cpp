// Out-of-framework test: only the public API header and the built framework
// libraries are used here. No framework test headers or internal layer headers.
#include <api/tensor.h>

#include <iostream>
#include <string_view>
#include <vector>

// Forces the custom layer plugin to be linked and loaded.
extern "C" void ld_custom_layers_load();

namespace {

int failures = 0;

void check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    ++failures;
  }
}

} // namespace

int main() {
  ld_custom_layers_load();

  tf::layer::layer custom("test_scale");
  check(custom.valid(), "out-of-tree layer was registered and created");
  check(custom.type() == "test_scale", "layer reports its registered name");
  check(custom.id() != tf::layer::invalid_layer_id,
        "layer has a stable, non-invalid id");

  const std::vector<tf::tensor> scale = custom.getParameter("scale");
  check(scale.size() == 1, "custom layer exposes its 'scale' parameter");
  if (scale.size() == 1 && scale.front().getPtr())
    check(scale.front().getData()[0] == 1.0,
          "custom layer factory supplied the default scale");

  const tf::layer::layer missing("no_such_custom_layer");
  check(!missing.valid(), "unknown layer type yields an inert handle");
  check(missing.id() == tf::layer::invalid_layer_id,
        "unknown layer type has the invalid id");

  if (failures != 0) {
    std::cerr << failures << " custom check(s) failed\n";
    return 1;
  }

  std::cout << "custom layer checks passed\n";
  return 0;
}
