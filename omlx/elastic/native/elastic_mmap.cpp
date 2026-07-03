// SPDX-License-Identifier: Apache-2.0
//
// _elastic_mmap: mmap-elastic (file-backed, OS-reclaimable) weight loading
// for MLX, llama.cpp-style. One MAP_SHARED|PROT_READ mapping per safetensors
// shard, wrapped zero-copy in a Metal buffer via mlx::allocator::make_buffer
// (newBufferWithBytesNoCopy), with per-tensor arrays as byte-offset views
// sharing that buffer.
//
// Because the mapping is PROT_READ and file-backed, its pages stay CLEAN:
// under memory pressure the OS can evict them at zero cost and refault them
// from the checkpoint file, instead of swapping dirty anonymous pages. This
// is what makes an "elastic" model: physical footprint shrinks under
// pressure and recovers on demand.
//
// Constraints (see omlx/elastic/README.md):
//  - Weights are READ-ONLY. Any weight-mutating path (in-place update,
//    fused LoRA, requantization) must materialize a private copy first.
//  - Tensor byte offsets must be itemsize-aligned for zero-copy views;
//    omlx.elastic.realign rewrites shards so 100% of tensors qualify.
//
// Validated against the mlx 0.31.2 wheel (public C++ API only). The
// nanobind build flags MUST match the wheel: v2.12.0, NB_STATIC,
// STABLE_ABI, NB_DOMAIN mlx (see CMakeLists.txt).

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mlx/allocator.h"
#include "mlx/array.h"

namespace nb = nanobind;
namespace mx = mlx::core;

namespace {

mx::Dtype dtype_from_str(const std::string& s) {
  if (s == "BF16") return mx::bfloat16;
  if (s == "F16") return mx::float16;
  if (s == "F32") return mx::float32;
  if (s == "U32") return mx::uint32;
  if (s == "U16") return mx::uint16;
  if (s == "U8") return mx::uint8;
  if (s == "I8") return mx::int8;
  if (s == "I16") return mx::int16;
  if (s == "I32") return mx::int32;
  if (s == "I64") return mx::int64;
  if (s == "BOOL") return mx::bool_;
  throw std::runtime_error("[elastic_mmap] unsupported dtype " + s);
}

} // namespace

// Map the whole file (page-aligned by construction) and wrap it zero-copy in
// a Metal buffer. Returns a uint8 array shaped [npages, page_size].
// Throws if the zero-copy wrap fails (we do NOT want the silent copy
// fallback of the array(void*, ...) constructor for this path).
mx::array wrap_file(const std::string& path, bool willneed) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("[elastic_mmap] cannot open " + path);
  }
  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    throw std::runtime_error("[elastic_mmap] fstat failed " + path);
  }
  size_t sz = static_cast<size_t>(st.st_size);
  size_t pg = static_cast<size_t>(getpagesize()); // 16384 on Apple Silicon
  size_t map_len = (sz + pg - 1) & ~(pg - 1);

  // MAP_SHARED + PROT_READ: pages stay clean + file-backed => the OS can
  // evict them at zero cost and refault from the file (llama.cpp's trick).
  void* ptr = mmap(nullptr, map_len, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) {
    throw std::runtime_error("[elastic_mmap] mmap failed " + path);
  }
  if (willneed) {
    madvise(ptr, map_len, MADV_WILLNEED);
  }

  // newBufferWithBytesNoCopy over the mapping. make_buffer also inserts the
  // buffer into MLX's residency set; keep the process wired limit clamped
  // (omlx.elastic.state.block_wiring) so the set never wires these pages.
  auto buf = mx::allocator::make_buffer(ptr, map_len);
  if (buf.ptr() == nullptr) {
    munmap(ptr, map_len);
    throw std::runtime_error(
        "[elastic_mmap] newBufferWithBytesNoCopy rejected the mapping (ptr " +
        std::to_string(reinterpret_cast<uintptr_t>(ptr)) + " len " +
        std::to_string(map_len) + ")");
  }

  auto deleter = [map_len](mx::allocator::Buffer b) {
    void* p = b.raw_ptr();
    // Releases the MTLBuffer wrapper (created with a null deallocator, so
    // this does not free the pages) and removes it from the residency set...
    mx::allocator::release(b);
    // ...then drop the mapping itself.
    munmap(p, map_len);
  };

  if (map_len / pg > static_cast<size_t>(INT32_MAX)) {
    throw std::runtime_error("[elastic_mmap] file too large");
  }
  auto npages = static_cast<int32_t>(map_len / pg);
  return mx::array(
      buf, mx::Shape{npages, static_cast<int32_t>(pg)}, mx::uint8, deleter);
}

// A typed, row-contiguous view into `parent`'s buffer at `byte_offset`.
// Shares the parent's Data (and thus the munmap custody); no copy. The
// parent mapping stays alive as long as any view is alive — callers do not
// need to hold a reference to the parent.
mx::array make_view(
    const mx::array& parent,
    size_t byte_offset,
    const std::vector<int32_t>& shape,
    const std::string& dtype_str) {
  auto dtype = dtype_from_str(dtype_str);
  size_t itemsize = dtype.size();
  if (byte_offset % itemsize != 0) {
    throw std::runtime_error(
        "[elastic_mmap] tensor byte offset not aligned to itemsize; caller "
        "must fall back to a copy for this tensor (or realign the shard)");
  }

  size_t nelem = 1;
  for (auto d : shape) {
    nelem *= static_cast<size_t>(d);
  }

  // Row-major strides.
  mx::Strides strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  mx::array::Flags flags{};
  flags.contiguous = true;
  flags.row_contiguous = true;
  int64_t max_dim = 1;
  for (auto d : shape) {
    max_dim = std::max<int64_t>(max_dim, d);
  }
  flags.col_contiguous =
      (nelem <= 1 || static_cast<int64_t>(nelem) == max_dim);

  // Construct an `available` array backed (temporarily, with a no-op
  // deleter) by the parent's buffer, then re-point it at the parent's
  // shared Data with the right strides/flags/offset. copy_shared_buffer's
  // offset parameter is in elements of *this* array's dtype.
  mx::array child(
      parent.buffer(),
      mx::Shape(shape.begin(), shape.end()),
      dtype,
      [](mx::allocator::Buffer) {});
  child.copy_shared_buffer(
      parent, strides, flags, nelem, byte_offset / itemsize);
  return child;
}

NB_MODULE(_elastic_mmap, m) {
  m.def(
      "wrap_file",
      &wrap_file,
      nb::arg("path"),
      nb::arg("willneed") = true,
      "mmap a file (MAP_SHARED|PROT_READ) and wrap it zero-copy as a "
      "[npages, 16384] uint8 mx.array via newBufferWithBytesNoCopy.");
  m.def(
      "make_view",
      &make_view,
      nb::arg("parent"),
      nb::arg("byte_offset"),
      nb::arg("shape"),
      nb::arg("dtype"),
      "Typed zero-copy view into the parent mapping at byte_offset.");
}
