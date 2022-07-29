cmake_minimum_required(VERSION 3.5)

project(face_recognition_system LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV  4.5 REQUIRED)

set(COMMON_SRC
    predictor.h predictor.cpp
    opencv_predictor.h opencv_predictor.cpp
    common_utils.h     common_utils.cpp
    tracker.h     tracker.cpp
    kuhnmunkres.h     kuhnmunkres.cpp
    detection_decoders.h detection_decoders.cpp
    faces_manager.h faces_manager.cpp
    io_utils.h io_utils.cpp
    nlohmann/json.hpp
    face.cpp face.h
    draw_utils.cpp draw_utils.h
    dbmanager.cpp dbmanager.h
)


add_executable(face_recognition_system main.cpp recognition_status_tracker.cpp recognition_status_tracker.h ${COMMON_SRC})
target_link_libraries(face_recognition_system ${OpenCV_LIBS})

add_executable(dbcreator db_creator.cpp ${COMMON_SRC})
target_link_libraries(dbcreator ${OpenCV_LIBS})

#add_executable(embedding_matrix embedding_matrix.cpp dbmanager.cpp dbmanager.h)
