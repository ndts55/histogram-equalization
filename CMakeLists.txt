cmake_minimum_required(VERSION 3.1)

project(
  histogram-equalization
  VERSION 1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

find_package(MPI)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(contrast contrast-enhancement.cpp histogram-equalization.cpp  contrast.cpp)
target_link_libraries(contrast ${MPI_C_LIBRARIES})

add_executable(contrast-mpi contrast-enhancement.cpp histogram-equalization.cpp  contrast-mpi.cpp omp-image-processing.h)
target_link_libraries(contrast ${MPI_C_LIBRARIES})

add_executable(contrast-mpi-openmp contrast-mpi-openmp.cpp)
target_link_libraries(contrast ${MPI_C_LIBRARIES})
