cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(test test.cpp)

add_library(game_ai SHARED GameAI.cpp nhl94.cpp data.cpp memory.cpp utils.cpp)

target_link_libraries(game_ai "${TORCH_LIBRARIES}")

set_property(TARGET test PROPERTY CXX_STANDARD 17)