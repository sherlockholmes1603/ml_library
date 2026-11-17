cmake_minimum_required(VERSION 3.10)
project(MyMLPlusPlus VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable AddressSanitizer for memory leak detection
add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
add_link_options(-fsanitize=address)

# 1. Find Eigen3 package
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(include)

set(SOURCES
    src/tensor.cpp
    src/graph.cpp
    src/linear.cpp
    src/optimizer.cpp
    src/linear_regression.cpp
    src/logistic_regression.cpp
    # Add new .cpp files here as you create them 
)

add_library(ml_lib ${SOURCES})
target_include_directories(ml_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ml_lib PUBLIC Eigen3::Eigen)

add_executable(ml_test_app apps/main.cpp)
target_link_libraries(ml_test_app ml_lib)

# Include test directories
add_subdirectory(tests)