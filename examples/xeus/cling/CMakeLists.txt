cmake_minimum_required(VERSION 4.0)
project(xcling)
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

find_package(cppzmq REQUIRED)
find_package(xeus-zmq ${xeus-zmq_REQUIRED_VERSION} REQUIRED)
find_package(Boost REQUIRED COMPONENTS process)
find_package(Cling REQUIRED)
find_package(Threads)

message(STATUS "Looking for cling executable...")
find_program(CLING_EXECUTABLE cling REQUIRED)
if (CLING_EXECUTABLE)
    message(STATUS "Found cling: ${CLING_EXECUTABLE}")
else ()
    message(FATAL_ERROR "Cling was not found in PATH.")
endif ()

get_filename_component(CLING_BIN_DIR ${CLING_EXECUTABLE} DIRECTORY)
get_filename_component(CLING_ROOT_DIR ${CLING_BIN_DIR} DIRECTORY)
include_directories(${CLING_ROOT_DIR}/include)
link_directories(${CLING_ROOT_DIR}/lib)

# Flags
include(CheckCXXCompilerFlag)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    CHECK_CXX_COMPILER_FLAG("-std=c++17" HAS_CPP17_FLAG)

    if (HAS_CPP17_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    else ()
        message(FATAL_ERROR "Unsupported compiler -- xeus requires C++17 support!")
    endif()
else ()
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Target
set(XCLING_SRC
    src/xbuffer.hpp
    src/xinterpreter.cpp
    src/xinterpreter.hpp
)

add_executable(${EXECUTABLE_NAME} src/main.cpp ${XCLING_SRC})
target_link_libraries(${EXECUTABLE_NAME} PRIVATE
    clingInterpreter
    cppzmq
    ${xeus-zmq_target}
    Boost::process
    Threads::Threads
)
set_target_properties(${EXECUTABLE_NAME} PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

if (MSVC)
    set(CLING_SYMBOLS
        # use undname to get undecorated names
        __std_find_trivial_1 # std::cin
        __std_reverse_trivially_swappable_1 # std::cin
        __std_reverse_trivially_swappable_4 # std::reverse
        ?_Facet_Register@std@@YAXPEAV_Facet_base@1@@Z # std::cin
        ??_7type_info@@6B@ # std::cout
        ??2@YAPEAX_K@Z # std::ostringstream
        ??3@YAXPEAX_K@Z # std::cout
        ??3@YAXPEAX@Z # delete
    )

    foreach(SYMBOL ${CLING_SYMBOLS})
        set(CLING_EXPORTS "${CLING_EXPORTS} /EXPORT:${SYMBOL}")
    endforeach(SYMBOL ${CLING_SYMBOLS})

    set_property(TARGET ${EXECUTABLE_NAME} APPEND_STRING PROPERTY LINK_FLAGS ${CLING_EXPORTS})
endif ()

# Installation
install(TARGETS ${EXECUTABLE_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
# install(FILES $<TARGET_RUNTIME_DLLS:${EXECUTABLE_NAME}> TYPE BIN)

string(REPLACE "\\" "/" CONDA_PREFIX $ENV{CONDA_PREFIX})
install(DIRECTORY "${CONDA_PREFIX}/Library/bin/"
        TYPE BIN
        COMPONENT Runtime
        FILES_MATCHING
        PATTERN "*.dll"
        PATTERN "*.dylib"
        PATTERN "*.so"
        PATTERN "api-*" EXCLUDE
        PATTERN "msvc*" EXCLUDE
        PATTERN "vc*" EXCLUDE
)

set(JUPYTER_DATA_DIR "share/jupyter" CACHE STRING "jupyter data directory")
set(XCLING_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels)

install(DIRECTORY ${XCLING_DATA_DIR}
        DESTINATION ${JUPYTER_DATA_DIR}
        PATTERN "*.in" EXCLUDE
)
