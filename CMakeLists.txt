cmake_minimum_required(VERSION 3.10)
project(qwen3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)

find_package(Boost REQUIRED COMPONENTS regex)

set(CORE_SRCS
    src/qwen3.cpp
    src/tokenizer.cpp
)

add_executable(qwen3
    src/main.cpp
    ${CORE_SRCS}
)

add_executable(qwen3_chat
    src/qwen3_chat.cpp
    ${CORE_SRCS}
)

target_include_directories(qwen3 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(qwen3_chat PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(qwen3 PRIVATE Boost::regex)
target_link_libraries(qwen3_chat PRIVATE Boost::regex)
