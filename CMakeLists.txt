cmake_minimum_required(VERSION 3.1)

project(
        histogram-equalization
        VERSION 1.0
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "-Wall ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wall ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

add_executable(contrast contrast-enhancement.cpp histogram-equalization.cpp contrast.cpp)
target_link_libraries(contrast ${MPI_C_LIBRARIES})

add_executable(contrast-mpi contrast-mpi.cpp)
target_link_libraries(contrast-mpi PUBLIC MPI::MPI_CXX)

add_executable(contrast-mpi-openmp contrast-mpi-openmp.cpp)
target_link_libraries(contrast-mpi-openmp ${MPI_C_LIBRARIES})

add_executable(contrast-openmp contrast-openmp.cpp)
