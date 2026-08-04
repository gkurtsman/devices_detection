cmake_minimum_required(VERSION 3.0.0)
project(salient)

find_package(ngraph REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(OpenCV REQUIRED)
find_package(InferenceEngine REQUIRED)
#find_package(gflags REQUIRED)
#find_package(gflags COMPONENTS nothreads_static)

# set(InferenceEngine_DIR /opt/intel/openvino_2021/deployment_tools/inference_engine/)
include_directories(${InferenceEngine_DIR}/../samples/cpp/common/format_reader/)
include_directories(${InferenceEngine_DIR}/../samples/cpp/common/utils/include/)
#link_directories(/usr/lib/x86_64-linux-gnu/ /opt/intel/openvino_2021/deployment_tools/inference_engine/samples/cpp/object_detection_sample_ssd/build/intel64/Debug/lib/)
link_directories(/usr/lib/x86_64-linux-gnu/)

#set(CMAKE_LIBRARY_PATH /opt/intel/openvino_2021/deployment_tools/inference_engine/samples/cpp/object_detection_sample_ssd/build/intel64/Debug/lib/)

# set (gflags_DIR /usr/lib/x86_64-linux-gnu/) 

add_executable(${PROJECT_NAME} salient_detect.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE format_reader ie_samples_utils cnpy zlib ${InferenceEngine_LIBRARIES} ${OpenCV_LIBS} ${NGRAPH_LIBRARIES})
#target_link_libraries(${PROJECT_NAME} PRIVATE ${InferenceEngine_LIBRARIES} ${OpenCV_LIBS}    ${NGRAPH_LIBRARIES} gflags glog format_reader ie_samples_utils gflags_nothreads)
