cmake_minimum_required(VERSION 3.10)
project(qwen3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)

find_package(Boost REQUIRED COMPONENTS regex)

file(GLOB SRCS src/*.cpp)

add_executable(qwen3 ${SRCS})

target_include_directories(qwen3 PRIVATE 
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
                    ${CMAKE_CURRENT_SOURCE_DIR}/src
                    )
target_link_libraries(qwen3 PRIVATE Boost::regex)
