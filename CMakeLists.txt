cmake_minimum_required(VERSION 3.21)
project(FieldSim LANGUAGES CXX HIP)

# Export compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include_directories("/opt/rocm/include" "${CMAKE_SOURCE_DIR}/src/include")
add_compile_definitions("__HIP_PLATFORM_AMD__")
set(CMAKE_MODULE_PATH "/opt/rocm/lib/cmake/hip/" ${CMAKE_MODULE_PATH})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(ROCM_PATH "/opt/rocm")

# Tell CMake to use ROCm's HIP
find_package(HIP REQUIRED)

add_subdirectory(src)
add_subdirectory(tests)

enable_testing()
