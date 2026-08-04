cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(truck_detection_example)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(OpenCV_INCLUDE_DIRS /usr/local/include/opencv4)
#SET(OpenCV_LIBS /usr/local/lib64/)
#SET(OpenCV_LIBS /home/krtsmn/Develop/comp_vision/opencv/opencv/build/lib/)
#SET(OpenCV_LIBS /lib64)

# Save the command line compile commands in the build output
set(CMAKE_EXPORT_COMPILE_COMMANDS 14)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_C_COMPILER gcc)

find_package(OpenCV REQUIRED )

# ************* configurable ******************
set(HOMEDIR /home/krtsmn/Develop/comp_vision/dnn/devices_detection/truck_sample_cv/)
#set(ICONFIG config_kamatsu_20180831.h)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCONFIG_FILE=/var/local/config_pc220.conf")
set(CMAKE_VERBOSE_MAKEFILE ON)
# *********************************************

#set(EXCDIR ${HOMEDIR}/src/excavator_bucket/mycpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g -ggdb -O0")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include ${HOMEDIR}/include/configs/${ICONFIG}")

#include_directories(${EXCDIR} ${HOMEDIR}/include/configs ${HOMEDIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

#add_definitions(${PCL_DEFINITIONS})

#add_library (ms-exc-bucket-util.o STATIC ${HOMEDIR}/classes/ms-exc-bucket-util.cpp)
#add_library (ms-service.o STATIC ${HOMEDIR}/classes/ms-service.cpp)
add_library (truck_det.cpp.o STATIC ${HOMEDIR}/truck_det.cpp)
add_executable (truck_det truck_det.cpp)
#link_directories(${OpenCV_LIBS})
target_link_libraries (truck_det ${OpenCV_LIBS})
#target_link_libraries (truck_det})
