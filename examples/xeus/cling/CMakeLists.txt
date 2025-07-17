set(EXECUTABLE_NAME xcling)

# Configuration
include(GNUInstallDirs)

option(XEUS_STATIC_DEPENDENCIES "link statically with xeus dependencies" OFF)
if (XEUS_STATIC_DEPENDENCIES)
    set(xeus-zmq_target "xeus-zmq-static")
else ()
    set(xeus-zmq_target "xeus-zmq")
endif ()

# Dependencies
set(xeus-zmq_REQUIRED_VERSION 1.0.2)

find_package(xeus-zmq ${xeus-zmq_REQUIRED_VERSION} REQUIRED)
find_package(Threads)

# Flags
include(CheckCXXCompilerFlag)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    CHECK_CXX_COMPILER_FLAG("-std=c++14" HAS_CPP14_FLAG)

    if (HAS_CPP14_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    else ()
        message(FATAL_ERROR "Unsupported compiler -- xeus requires C++14 support!")
    endif()
endif()

# Target
set(XCLING_SRC
    src/custom_interpreter.cpp
    src/custom_interpreter.hpp
)

add_executable(${EXECUTABLE_NAME} src/main.cpp ${XCLING_SRC})
target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${xeus-zmq_target} Threads::Threads)
set_target_properties(${EXECUTABLE_NAME} PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Installation
install(TARGETS ${EXECUTABLE_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

set(JUPYTER_DATA_DIR "share/jupyter" CACHE STRING "jupyter data directory")
set(XCLING_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/share/jupyter/kernels)

install(DIRECTORY ${XCLING_DATA_DIR}
        DESTINATION ${JUPYTER_DATA_DIR}
        PATTERN "*.in" EXCLUDE)
