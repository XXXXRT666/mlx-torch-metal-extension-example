// This example intentionally uses one native module and one metallib for both
// MLX and PyTorch MPS. The MLX entry point is exposed through nanobind, while
// the PyTorch entry point is registered through the dispatcher.

#include <ATen/ATen.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <torch/library.h>

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "mlx/backend/metal/device.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

namespace {

constexpr const char *kKernelName = "silu_and_mul_f16";

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void *>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get the extension module path.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

void validate_last_dim(int64_t last_dim) {
  if (last_dim <= 0 || last_dim % 2 != 0) {
    throw std::invalid_argument(
        "input must have a positive, even-sized last dimension");
  }
}

class SiluAndMul : public mx::UnaryPrimitive {
public:
  explicit SiluAndMul(mx::Stream stream) : mx::UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<mx::array> &, mx::array &) override {
    throw std::runtime_error("silu_and_mul is only implemented for the GPU");
  }

  void eval_gpu(const std::vector<mx::array> &inputs,
                mx::array &output) override {
    const auto &input = inputs[0];
    output.set_data(mx::allocator::malloc(output.nbytes()));
    if (output.size() == 0) {
      return;
    }

    const uint32_t d = input.shape(-1) / 2;
    const uint32_t num_tokens = input.size() / input.shape(-1);
    const uint32_t num_chunks = (d + 7) / 8;

    auto &device = mx::metal::device(stream().device);
    auto *library =
        device.get_library(MLX_METAL_LIBRARY_NAME, current_binary_dir());
    auto *kernel = device.get_kernel(kKernelName, library);

    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_output_array(output, 0);
    encoder.set_input_array(input, 1);
    encoder.set_bytes(d, 2);

    const size_t group_width =
        std::min(static_cast<size_t>(num_chunks),
                 static_cast<size_t>(kernel->maxTotalThreadsPerThreadgroup()));
    encoder.dispatch_threads(MTL::Size(num_chunks, num_tokens, 1),
                             MTL::Size(group_width, 1, 1));
  }

  const char *name() const override { return "SiluAndMul"; }

  bool is_equivalent(const mx::Primitive &) const override { return true; }
};

mx::array mlx_silu_and_mul(const mx::array &input,
                           mx::StreamOrDevice stream_or_device = {}) {
  if (input.ndim() == 0) {
    throw std::invalid_argument("input must have at least one dimension");
  }
  validate_last_dim(input.shape(-1));
  if (input.dtype() != mx::float16) {
    throw std::invalid_argument("input must have dtype mlx.core.float16");
  }

  auto stream = mx::to_stream(stream_or_device);
  auto contiguous_input = mx::contiguous(input, false, stream);
  auto output_shape = input.shape();
  output_shape.back() /= 2;
  return mx::array(output_shape, input.dtype(),
                   std::make_shared<SiluAndMul>(stream), {contiguous_input});
}

at::native::mps::PrecompiledMetalShaderLibrary &torch_metal_library() {
  // Keep the library alive until process teardown. This also avoids depending
  // on MLX's Metal command queue for the PyTorch entry point.
  static auto *library = new at::native::mps::PrecompiledMetalShaderLibrary(
      current_binary_dir() + "/" + MLX_METAL_LIBRARY_NAME + ".metallib");
  return *library;
}

at::Tensor torch_silu_and_mul(const at::Tensor &input) {
  TORCH_CHECK(input.is_mps(), "input must be a torch MPS tensor");
  TORCH_CHECK(input.scalar_type() == at::kHalf,
              "input must have dtype torch.float16");
  TORCH_CHECK(input.dim() > 0, "input must have at least one dimension");

  const int64_t last_dim = input.size(-1);
  TORCH_CHECK(last_dim > 0 && last_dim % 2 == 0,
              "input must have a positive, even-sized last dimension");

  auto contiguous_input = input.contiguous();
  auto output_shape = input.sizes().vec();
  output_shape.back() /= 2;
  auto output = at::empty(output_shape, input.options());
  if (output.numel() == 0) {
    return output;
  }

  const uint32_t d = static_cast<uint32_t>(last_dim / 2);
  const uint64_t num_tokens = input.numel() / last_dim;
  const uint64_t num_chunks = (d + 7) / 8;

  auto kernel = torch_metal_library().getKernelFunction(kKernelName);
  const uint64_t group_width =
      std::min(num_chunks, kernel->getMaxThreadsPerThreadgroup());
  const std::array<uint64_t, 2> threads{num_chunks, num_tokens};
  const std::array<uint64_t, 2> group_size{group_width, 1};

  // runCommandBlock encodes on PyTorch's current MPS stream. The Torch and MLX
  // paths therefore share kernel code without sharing command queues.
  kernel->runCommandBlock([&]() {
    kernel->startEncoding();
    kernel->setArg(0, output);
    kernel->setArg(1, contiguous_input);
    kernel->setArg(2, d);
    kernel->dispatch(threads, group_size);
  });

  return output;
}

} // namespace

TORCH_LIBRARY(mlx_torch_metal, m) {
  m.def("silu_and_mul(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(mlx_torch_metal, MPS, m) {
  m.impl("silu_and_mul", &torch_silu_and_mul);
}

NB_MODULE(MLX_EXTENSION_NAME, m) {
  m.doc() = "One Metal kernel with native MLX and PyTorch MPS entry points.";
  m.def("silu_and_mul", &mlx_silu_and_mul, "input"_a, nb::kw_only(),
        "stream"_a = nb::none(),
        nb::sig(
            "def silu_and_mul(input: mlx.core.array, *, stream: "
            "mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | "
            "None = None) -> mlx.core.array"),
        "Compute silu(x) * y after splitting the input's last dimension.");
}
