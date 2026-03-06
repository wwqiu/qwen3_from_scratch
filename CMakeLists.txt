cmake_minimum_required(VERSION 3.10)
project(qwen3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED COMPONENTS regex)

add_executable(qwen3
    main.cpp
    tokenizer.cpp
)

target_include_directories(qwen3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qwen3 PRIVATE Boost::regex)
