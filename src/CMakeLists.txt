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
    all_predictors.h    
)

set(OPENVINO_PREDICTOR_SRC
    openvino_predictor.cpp openvino_predictor.h
)

### face_recognition_system
#add_executable(face_recognition_system
#    main.cpp
#    recognition_status_tracker.cpp recognition_status_tracker.h
#    image_saver.cpp image_saver.h
#    ${COMMON_SRC})
#target_link_libraries(face_recognition_system ${OpenCV_LIBS} pthread)

### dbcreator
#add_executable(dbcreator db_creator.cpp ${COMMON_SRC})
#target_link_libraries(dbcreator ${OpenCV_LIBS})

### embedding_matrix
#add_executable(embedding_matrix embedding_matrix.cpp dbmanager.cpp dbmanager.h)


### TESTS
# test_openvino_predictor
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
add_executable(test_openvino_predictor
    test_openvino_predictor.cpp
    predictor.h
    ${OPENVINO_PREDICTOR_SRC}
)
target_link_libraries(test_openvino_predictor ${OpenCV_LIBS} openvino::runtime pthread)
