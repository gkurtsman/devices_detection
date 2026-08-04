cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(truck_detection_example)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
#set (CAFFE_DIR /home/krtsmn/Develop/comp_vision/dnn/caffe)

# Save the command line compile commands in the build output
find_package(OpenCV REQUIRED )
find_package(realsense2)

set(OpenCV_INCLUDE_DIRS /usr/include/opencv4)
SET(OpenCV_LIBS /home/krtsmn/Develop/comp_vision/opencv/opencv/build/lib/)
#SET(OpenCV_LIBS /lib64)

# ************* configurable ******************
set(HOMEDIR /home/krtsmn/Develop/comp_vision/dnn/devices_detection/truck_sample_caffe/)
#set (CAFFE_DIR /home/krtsmn/Develop/comp_vision/dnn/caffe_ssd)
set (CAFFE_DIR /home/krtsmn/Develop/comp_vision/dnn/caffe_VOC0712refinedet)
set(LIBREALSENSE_TREE /home/krtsmn/Develop/comp_vision/librealsense)
set(CMAKE_VERBOSE_MAKEFILE ON)
# *********************************************

SET(CAFFE_INCLUDE_DIRS ${CAFFE_DIR}/include/)
SET(CAFFE_LIBS ${CAFFE_DIR}/build/lib/)
SET(CAFFE_BUILD ${CAFFE_DIR}/build/)

set(CMAKE_EXPORT_COMPILE_COMMANDS 14)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_C_COMPILER gcc)

#set(EXCDIR ${HOMEDIR}/src/excavator_bucket/mycpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0 -DCPU_ONLY -DUSE_OPENCV -UUSE_MEAN")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14 -ggdb -O0")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include ${HOMEDIR}/include/configs/${ICONFIG}")

include_directories(${HOMEDIR} ${OpenCV_INCLUDE_DIRS} ${CAFFE_INCLUDE_DIRS} ${CAFFE_BUILD}/src)
include_directories(${realsense2_INCLUDE_DIRS})

#add_library (ms-exc-bucket-util.o STATIC ${HOMEDIR}/classes/ms-exc-bucket-util.cpp)
#add_library (ms-service.o STATIC ${HOMEDIR}/classes/ms-service.cpp)
#add_library (truck_det.cpp.o STATIC ${HOMEDIR}/truck_det.cpp)
add_library (cam_input.o STATIC ${HOMEDIR}/cam_input.cpp)
add_executable (truck_det truck_det.cpp)
#link_directories(${OpenCV_LIBS})

#target_link_libraries (truck_det ${OpenCV_LIBS} ${CAFFE_LIBS}/libcaffe.so)
#target_link_libraries (truck_det})
target_link_libraries (truck_det cam_input.o ${CAFFE_LIBS}/libcaffe.so libglog.so.0  libopencv_core.so libopencv_dnn.so libpng.so libopencv_imgcodecs.so realsense2 libopencv_imgproc.so libopencv_highgui.so libopencv_gapi.so libopencv_stitching.so libopencv_alphamat.so libopencv_aruco.so libopencv_bgsegm.so libopencv_bioinspired.so libopencv_ccalib.so libopencv_cvv.so libopencv_dnn_objdetect.so libopencv_dnn_superres.so libopencv_dpm.so libopencv_highgui.so libopencv_face.so libopencv_freetype.so libopencv_fuzzy.so libopencv_hdf.so libopencv_hfs.so libopencv_img_hash.so libopencv_intensity_transform.so libopencv_line_descriptor.so libopencv_quality.so libopencv_rapid.so libopencv_reg.so libopencv_rgbd.so libopencv_saliency.so libopencv_shape.so libopencv_stereo.so libopencv_structured_light.so libopencv_phase_unwrapping.so libopencv_superres.so libopencv_optflow.so libopencv_surface_matching.so libopencv_tracking.so libopencv_datasets.so libopencv_text.so libopencv_dnn.so libopencv_plot.so libopencv_ml.so libopencv_videostab.so libopencv_videoio.so libopencv_viz.so libopencv_ximgproc.so libopencv_video.so libopencv_xobjdetect.so libopencv_objdetect.so libopencv_calib3d.so libopencv_imgcodecs.so libopencv_features2d.so libopencv_flann.so libopencv_xphoto.so libopencv_photo.so libopencv_imgproc.so libopencv_core.so)
