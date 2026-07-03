# SPDX-License-Identifier: Apache-2.0
# Build the _elastic_mmap nanobind extension against the *installed* mlx
# wheel. Reproducible entry point: scripts/build-elastic-ext.sh
#
# CRITICAL: the nanobind configuration must match the mlx wheel exactly
# (version v2.12.0 for mlx 0.31.x, NB_STATIC, STABLE_ABI, NB_DOMAIN mlx),
# otherwise mx::array objects cannot pass between this module and mlx.core
# (nanobind type casters are looked up per-domain).

cmake_minimum_required(VERSION 3.25)
project(elastic_mmap LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)

include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v2.12.0
  GIT_SHALLOW TRUE
  EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(nanobind)

# MLX from the installed wheel (headers + libmlx.dylib + cmake config).
execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import mlx, pathlib; print(pathlib.Path(mlx.__file__).parent)"
  OUTPUT_VARIABLE MLX_PKG_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(MLX CONFIG REQUIRED PATHS "${MLX_PKG_DIR}/share/cmake/MLX")

nanobind_add_module(
  _elastic_mmap
  NB_STATIC
  STABLE_ABI
  NB_DOMAIN
  mlx
  elastic_mmap.cpp)

target_link_libraries(_elastic_mmap PRIVATE mlx)
