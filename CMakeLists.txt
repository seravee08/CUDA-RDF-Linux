cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(RDF CXX)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)


find_package(OpenCV REQUIRED COMPONENTS core imgproc video highgui features2d ml)
find_package(Boost REQUIRED COMPONENTS system 
                              filesystem 
                              log 
                              log_setup 
                              thread 
                              date_time 
                              regex 
                              unit_test_framework
                              program_options)
find_package(Protobuf REQUIRED)
find_package(CUDA REQUIRED)

                              
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${PROTOBUF_INCLUDE_DIRS})

set(rdf_LIBRARIES ${Boost_LIBRARIES}
                  ${OpenCV_LIBS}
		  ${PROTOBUF_LIBRARIES})

cuda_add_library(cu_kernel ${CMAKE_SOURCE_DIR}/src/rdf/rdf_cu.cu ${CMAKE_SOURCE_DIR}/include/rdf/rdf_cu.cuh OPTIONS -arch sm_35)


file(GLOB UTIL_SOURCES src/util/*.cpp src/util/*.c)

file(GLOB PROTO_SOURCES include/proto/*.cc include/proto/*.cpp include/proto/*.c)

add_subdirectory(src)
add_subdirectory(tools)
