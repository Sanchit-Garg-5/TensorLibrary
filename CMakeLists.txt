cmake_minimum_required(VERSION 3.10)
project(MiniTensorLibrary)

# Force modern C++ standard
set(CMAKE_CXX_STANDARD 17)
#enforces that the C++ standard specified by
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP package on the system
find_package(OpenMP REQUIRED)

# Build the executable from tensor_test.cpp
add_executable(tensor_test tensor_test.cpp)

# Link OpenMP flags/libraries to the test executable
target_link_libraries(tensor_test PUBLIC OpenMP::OpenMP_CXX)