cmake_minimum_required(VERSION 3.10)

# Project configuration
set(APP_NAME "deepstream-test-app")
project(${APP_NAME} VERSION 1.0)

# Set CUDA_VER with automatic detection and fallback
if(DEFINED ENV{CUDA_VER})
    set(CUDA_VER $ENV{CUDA_VER})
    message(STATUS "Using CUDA_VER from environment: ${CUDA_VER}")
else()
    # Try to detect CUDA version automatically
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1.\\2" CUDA_VER ${CUDA_VERSION_STRING})
        message(STATUS "Auto-detected CUDA version: ${CUDA_VER}")
    else()
        # Fallback to common default version
        set(CUDA_VER "12.2")
        message(STATUS "CUDA not found, using default version: ${CUDA_VER}")
        message(STATUS "You can override by setting CUDA_VER environment variable")
    endif()
endif()
set(NVDS_VERSION "8.0")

# Detect target device
execute_process(
    COMMAND gcc -dumpmachine
    COMMAND cut -f1 -d-
    OUTPUT_VARIABLE TARGET_DEVICE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set installation directories
set(LIB_INSTALL_DIR "/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib/" CACHE STRING "DeepStream library installation directory")
set(APP_INSTALL_DIR "/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/bin/" CACHE STRING "DeepStream application installation directory")

# Use C for compatibility with original Makefile (change to CXX if using C++)
set(CMAKE_C_STANDARD 99)
enable_language(C)

# Find required packages
find_package(PkgConfig REQUIRED)

# Find GStreamer using pkg-config (exactly like the original Makefile)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)

# Validate CUDA installation
set(CUDA_INCLUDE_PATH "/usr/local/cuda-${CUDA_VER}/include")
set(CUDA_LIB_PATH "/usr/local/cuda-${CUDA_VER}/lib64")

if(NOT EXISTS ${CUDA_INCLUDE_PATH})
    message(WARNING "CUDA include directory not found: ${CUDA_INCLUDE_PATH}")
    message(STATUS "Trying alternative CUDA paths...")
    
    # Try some common alternative paths
    set(ALT_CUDA_PATHS 
        "/usr/local/cuda/include"
        "/opt/cuda/include"  
        "/usr/include/cuda"
    )
    
    foreach(alt_path ${ALT_CUDA_PATHS})
        if(EXISTS ${alt_path})
            get_filename_component(CUDA_ROOT ${alt_path} DIRECTORY)
            set(CUDA_INCLUDE_PATH ${alt_path})
            set(CUDA_LIB_PATH "${CUDA_ROOT}/lib64")
            if(NOT EXISTS ${CUDA_LIB_PATH})
                set(CUDA_LIB_PATH "${CUDA_ROOT}/lib")
            endif()
            message(STATUS "Found CUDA at: ${CUDA_ROOT}")
            break()
        endif()
    endforeach()
    
    if(NOT EXISTS ${CUDA_INCLUDE_PATH})
        message(FATAL_ERROR "CUDA installation not found. Please install CUDA or set CUDA_VER correctly.")
    endif()
endif()

message(STATUS "Using CUDA include path: ${CUDA_INCLUDE_PATH}")
message(STATUS "Using CUDA library path: ${CUDA_LIB_PATH}")

# Include directories (matching original Makefile exactly)
include_directories(
    "/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/sources/includes"
    "/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/sources/apps/apps-common/includes"
    ${CUDA_INCLUDE_PATH}
    ${GSTREAMER_INCLUDE_DIRS}
)

# Compiler flags (matching original Makefile)
add_compile_options(${GSTREAMER_CFLAGS_OTHER})

# Source files (all .c files in current directory)
file(GLOB SRCS "*.c" "/opt/nvidia/deepstream/deepstream-8.0/sources/apps/apps-common/src/deepstream_common.c")

# Header files (all .h files in current directory)  
file(GLOB INCS "*.h")

# Create executable
add_executable(${APP_NAME} ${SRCS})

# Set target properties
set_target_properties(${APP_NAME} PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
)

# Link directories (matching original Makefile exactly)
target_link_directories(${APP_NAME} PRIVATE
    ${CUDA_LIB_PATH}
    ${LIB_INSTALL_DIR}
)

# Link libraries (exactly matching original Makefile order and options)
target_link_libraries(${APP_NAME}
    ${GSTREAMER_LIBRARIES}
    cudart
    nvdsgst_helper
    m
    nvdsgst_meta
    nvds_meta
    nvds_yml_parser
    cuda
)

# Set RPATH (equivalent to -Wl,-rpath in original Makefile)
set_target_properties(${APP_NAME} PROPERTIES
    INSTALL_RPATH "${LIB_INSTALL_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Install target (matching original Makefile install behavior)
install(TARGETS ${APP_NAME}
    DESTINATION ${APP_INSTALL_DIR}
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
)

# Print configuration information
message(STATUS "Building ${APP_NAME}")
message(STATUS "CUDA Version: ${CUDA_VER}")
message(STATUS "DeepStream Version: ${NVDS_VERSION}")
message(STATUS "Target Device: ${TARGET_DEVICE}")
message(STATUS "Library Install Dir: ${LIB_INSTALL_DIR}")
message(STATUS "App Install Dir: ${APP_INSTALL_DIR}")

# Add custom targets for compatibility with original Makefile
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
    COMMENT "Cleaning all build artifacts"
)

# Print source and header files being compiled (for verification)
message(STATUS "Source files: ${SRCS}")
message(STATUS "Header files: ${INCS}")