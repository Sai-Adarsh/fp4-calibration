cmake_minimum_required(VERSION 3.24)
project(fp4_calibration LANGUAGES CXX)

option(QUANT_ENABLE_NVML "Sample NVIDIA device memory of llama-bench" OFF)
find_package(OpenSSL 1.1 REQUIRED COMPONENTS Crypto)
add_executable(quant_calibrate main.cpp gguf.cpp)
target_compile_features(quant_calibrate PRIVATE cxx_std_20)
target_link_libraries(quant_calibrate PRIVATE OpenSSL::Crypto)
if(QUANT_ENABLE_NVML)
  find_path(NVML_INCLUDE_DIR nvml.h REQUIRED)
  find_library(NVML_LIBRARY nvidia-ml REQUIRED)
  target_include_directories(quant_calibrate PRIVATE ${NVML_INCLUDE_DIR})
  target_link_libraries(quant_calibrate PRIVATE ${NVML_LIBRARY})
  target_compile_definitions(quant_calibrate PRIVATE QUANT_HAS_NVML)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(fake_llama_tool tests/fake_llama.cpp)
  target_compile_features(fake_llama_tool PRIVATE cxx_std_20)
  add_test(NAME quant_calibrate_cli
    COMMAND "${CMAKE_COMMAND}"
      "-DQUANT=$<TARGET_FILE:quant_calibrate>"
      "-DFAKE_LLAMA=$<TARGET_FILE:fake_llama_tool>"
      "-DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cli.cmake")
endif()
