cmake_minimum_required(VERSION 3.5)

project(face_recognition_system LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV  4.5 REQUIRED)

add_executable(face_recognition_system main.cpp)
target_link_libraries(face_recognition_system ${OpenCV_LIBS})
