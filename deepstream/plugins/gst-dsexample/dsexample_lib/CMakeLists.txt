cmake_minimum_required(VERSION 3.16)
project(dsexample_lib VERSION 1.0 LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Source files
set(SRCS dsexample_lib.c)

# Create static library
add_library(dsexample STATIC ${SRCS})

# Set library properties (matching original Makefile behavior)
set_target_properties(dsexample PROPERTIES
    # Enable debug symbols (equivalent to -ggdb)
    CMAKE_BUILD_TYPE Debug
    
    # Position independent code (equivalent to -fPIC)
    POSITION_INDEPENDENT_CODE ON
    
    # Output name
    OUTPUT_NAME "dsexample"
    ARCHIVE_OUTPUT_NAME "dsexample"
)

# Compiler flags (matching original Makefile)
target_compile_options(dsexample PRIVATE
    -ggdb        # Debug symbols
    -fPIC        # Position independent code
)

# Print build information
message(STATUS "Building dsexample static library")
message(STATUS "Source files: ${SRCS}")
message(STATUS "Output: libdsexample.a")

# The library will be built as libdsexample.a automatically by CMake