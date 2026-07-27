# Compiles the buf-generated C++ client (RM-P0-CORE-07) to prove the Core message
# schemas build against the protobuf C++ runtime. Generated sources come from
# `scripts/gen_proto.sh --langs` (codegen/cpp/generated, gitignored).
#
#   cmake -B build -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
#   cmake --build build
#
# libprotobuf 7.35.* (matching buf plugin protocolbuffers/cpp:v35.1) is provided by
# the conda dev env / CI (CONTRIBUTING.md). Protobuf's CMake config pulls in abseil.
cmake_minimum_required(VERSION 3.16)
project(astro_mine_core_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf CONFIG REQUIRED)

file(GLOB_RECURSE GEN_SRCS CONFIGURE_DEPENDS generated/*.pb.cc)
if(NOT GEN_SRCS)
  message(FATAL_ERROR "No generated C++ sources found. Run: ./scripts/gen_proto.sh --langs")
endif()

add_library(astro_mine_core_proto STATIC ${GEN_SRCS})
target_include_directories(astro_mine_core_proto PUBLIC generated)
target_link_libraries(astro_mine_core_proto PUBLIC protobuf::libprotobuf)
