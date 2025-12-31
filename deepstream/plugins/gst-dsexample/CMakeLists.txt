cmake_minimum_required(VERSION 3.16)
project(nvdsgst_dsexample VERSION 8.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options (matching Makefile defaults)
option(WITH_OPENCV "Build with OpenCV support" OFF)  # Default to OFF like Makefile
option(USE_OPTIMIZED_DSEXAMPLE "Use optimized dsexample plugin" OFF)

# Get CUDA version from environment (required)
if(DEFINED ENV{CUDA_VER})
    set(CUDA_VER $ENV{CUDA_VER})
    message(STATUS "Using CUDA_VER from environment: ${CUDA_VER}")
else()
    message(STATUS "CUDA_VER is not set. Please set CUDA_VER environment variable or use default CUDA value.")
    set(CUDA_VER "12.8")
endif()

# DeepStream version
set(NVDS_VERSION "8.0")

# Detect target device
execute_process(
    COMMAND gcc -dumpmachine
    COMMAND cut -f1 -d-
    OUTPUT_VARIABLE TARGET_DEVICE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set installation directories
set(GST_INSTALL_DIR "/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib/gst-plugins/" 
    CACHE PATH "GStreamer plugins installation directory")
set(LIB_INSTALL_DIR "/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib/" 
    CACHE PATH "DeepStream library installation directory")

# Find required packages
find_package(PkgConfig REQUIRED)

# Find GStreamer packages
pkg_check_modules(GSTREAMER REQUIRED 
    gstreamer-1.0 
    gstreamer-base-1.0 
    gstreamer-video-1.0
)

# Find OpenCV if enabled
if(WITH_OPENCV)
    pkg_check_modules(OPENCV REQUIRED opencv4)
    if(NOT OPENCV_FOUND)
        message(FATAL_ERROR "OpenCV4 not found via pkg-config")
    endif()
endif()

# CUDA paths
set(CUDA_INCLUDE_PATH "/usr/local/cuda-${CUDA_VER}/include")
set(CUDA_LIB_PATH "/usr/local/cuda-${CUDA_VER}/lib64")

# Validate CUDA installation
if(NOT EXISTS ${CUDA_INCLUDE_PATH})
    message(FATAL_ERROR "CUDA include directory not found: ${CUDA_INCLUDE_PATH}")
endif()

if(NOT EXISTS ${CUDA_LIB_PATH})
    message(FATAL_ERROR "CUDA library directory not found: ${CUDA_LIB_PATH}")
endif()

# Source files (conditional based on optimization flag)
if(USE_OPTIMIZED_DSEXAMPLE)
    set(SRCS gstdsexample_optimized.cpp)
    message(STATUS "Using optimized dsexample source")
else()
    set(SRCS gstdsexample.cpp)
    message(STATUS "Using standard dsexample source")
endif()

# Header files
file(GLOB INCS "*.h")

# Dependency library
set(DEP_LIB "dsexample_lib/libdsexample.a")

# Build dependency library first
add_custom_target(build_dsexample_lib
    COMMAND $(MAKE) -C dsexample_lib/
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building dsexample static library"
)

# Create the GStreamer plugin shared library
add_library(nvdsgst_dsexample SHARED ${SRCS})

# Add dependency on the static library build
add_dependencies(nvdsgst_dsexample build_dsexample_lib)

# Include directories
target_include_directories(nvdsgst_dsexample PRIVATE
    ${CUDA_INCLUDE_PATH}
    "/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/sources/includes"
    ${GSTREAMER_INCLUDE_DIRS}
)

# Add OpenCV includes if enabled (using system location like Makefile)
if(WITH_OPENCV)
    target_include_directories(nvdsgst_dsexample PRIVATE
        /usr/include/opencv4
        ${OPENCV_INCLUDE_DIRS}
    )
endif()

# Compiler definitions
target_compile_definitions(nvdsgst_dsexample PRIVATE
    -DDS_VERSION="8.0.0"
)

# Add OpenCV definition if enabled
if(WITH_OPENCV)
    target_compile_definitions(nvdsgst_dsexample PRIVATE -DWITH_OPENCV)
endif()

# Compiler options
target_compile_options(nvdsgst_dsexample PRIVATE
    -fPIC
    ${GSTREAMER_CFLAGS_OTHER}
)

# Add OpenCV compile flags if enabled
if(WITH_OPENCV)
    target_compile_options(nvdsgst_dsexample PRIVATE ${OPENCV_CFLAGS_OTHER})
endif()

# Link directories
target_link_directories(nvdsgst_dsexample PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/dsexample_lib
    ${CUDA_LIB_PATH}
    ${LIB_INSTALL_DIR}
)

# Link libraries
target_link_libraries(nvdsgst_dsexample PRIVATE
    # Dependency library (static)
    ${CMAKE_CURRENT_SOURCE_DIR}/${DEP_LIB}
    
    # CUDA runtime
    cudart
    dl
    
    # NPP libraries
    nppc
    nppig
    npps
    nppicc
    nppidei
    
    # DeepStream libraries
    nvdsgst_helper
    nvdsgst_meta
    nvds_meta
    nvbufsurface
    nvbufsurftransform

    hiredis
    
    # GStreamer libraries
    ${GSTREAMER_LIBRARIES}
)

# Add OpenCV libraries if enabled (via pkg-config like Makefile)
if(WITH_OPENCV)
    target_link_libraries(nvdsgst_dsexample PRIVATE ${OPENCV_LIBRARIES})
endif()

# Set shared library properties
set_target_properties(nvdsgst_dsexample PROPERTIES
    PREFIX "lib"
    SUFFIX ".so"
    POSITION_INDEPENDENT_CODE ON
    LINK_FLAGS "-shared -Wl,-no-undefined -Wl,-rpath,${LIB_INSTALL_DIR}"
)

# Install target
install(TARGETS nvdsgst_dsexample
    LIBRARY DESTINATION ${GST_INSTALL_DIR}
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
)

# Clean target for dependency library
add_custom_target(clean-dsexample-lib
    COMMAND $(MAKE) -C dsexample_lib/ clean
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Cleaning dsexample static library"
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== GStreamer Plugin Configuration ===")
message(STATUS "Plugin Name: nvdsgst_dsexample")
message(STATUS "CUDA Version: ${CUDA_VER}")
message(STATUS "DeepStream Version: ${NVDS_VERSION}")
message(STATUS "Target Device: ${TARGET_DEVICE}")
message(STATUS "Source File: ${SRCS}")
message(STATUS "Use Optimized: ${USE_OPTIMIZED_DSEXAMPLE}")
message(STATUS "OpenCV Support: ${WITH_OPENCV}")
message(STATUS "GST Install Dir: ${GST_INSTALL_DIR}")
message(STATUS "Lib Install Dir: ${LIB_INSTALL_DIR}")
message(STATUS "=====================================")
message(STATUS "")

# Usage instructions
message(STATUS "Build Instructions:")
message(STATUS "  export CUDA_VER=12.2  # Set your CUDA version")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  make")
message(STATUS "  make install")
message(STATUS "")
message(STATUS "Options:")
message(STATUS "  -DWITH_OPENCV=ON/OFF              # Enable/disable OpenCV (default: OFF)")
message(STATUS "  -DUSE_OPTIMIZED_DSEXAMPLE=ON/OFF  # Use optimized source (default: OFF)")
message(STATUS "  -DGST_INSTALL_DIR=path            # Override GST plugin install path")
message(STATUS "  -DLIB_INSTALL_DIR=path            # Override library install path")