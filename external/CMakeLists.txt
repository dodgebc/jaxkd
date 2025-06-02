cmake_minimum_required(VERSION 3.18...3.27)
project(jaxcukd LANGUAGES C CXX CUDA)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

# Get location of XLA FFI headers from current python environment
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import jaxlib; print(jaxlib.__path__[0] + '/include')"
  OUTPUT_VARIABLE JAXLIB_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES all-major)

# Add cudaKDTree subdirectory
add_subdirectory(cudaKDTree)

# Add bindings for cudaKDTree
add_library(jaxcukd SHARED bindings.cu)
set_target_properties(jaxcukd PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(jaxcukd PRIVATE cudaKDTree ${JAXLIB_INCLUDE_DIR})
target_link_libraries(jaxcukd PRIVATE cudaKDTree)
