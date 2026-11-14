cmake_minimum_required(VERSION 3.10)

# Use FetchContent to download and build GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Tests executable
add_executable(
    ml_lib_tests
    test_tensor.cpp
    test_autograd.cpp
    test_layers.cpp
    test_optimizer.cpp
)

# Link to libraries and GTest
target_link_libraries(
    ml_lib_tests
    ml_lib
    gtest
    gtest_main
)

# Enable testing
enable_testing()
add_test(NAME all_tests COMMAND ml_lib_tests)
