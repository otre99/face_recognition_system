cmake_minimum_required(VERSION 3.5)

project(face_recognition_system LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV  4.5 REQUIRED)

set(VISION_SRC
    vision/predictor.h vision/predictor.cpp
    vision/opencv_predictor.h vision/opencv_predictor.cpp
    vision/common_utils.h     vision/common_utils.cpp
    vision/tracker.h     vision/tracker.cpp
    vision/kuhnmunkres.h     vision/kuhnmunkres.cpp
    vision/detection_decoders.h vision/detection_decoders.cpp
)

set(MAIN_SRC
face_detection.h face_detection.cpp
io_utils.h io_utils.cpp
)



add_executable(face_recognition_system main.cpp ${VISION_SRC} ${MAIN_SRC})
target_link_libraries(face_recognition_system ${OpenCV_LIBS})


add_executable(dbtool db_main.cpp dbmanager.cpp dbmanager.h ${MAIN_SRC} nlohmann/json.hpp)
target_link_libraries(dbtool ${OpenCV_LIBS})
