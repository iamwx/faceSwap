cmake_minimum_required(VERSION 2.8)

PROJECT(faceSwap)

include(./lib/dlib-18.18/dlib/cmake)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib/")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib/")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin/")
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
	include_directories(${OpenCV_INCLUDE_DIRS})
	ADD_EXECUTABLE(faceSwap faceSwap.cpp)
	TARGET_LINK_LIBRARIES(faceSwap dlib ${OpenCV_LIBS})
else()
	ADD_EXECUTABLE(faceSwap faceSwap.cpp)
	TARGET_LINK_LIBRARIES(faceSwap faceSwap.cpp)
endif()

if(DLIB_LINK_WITH_SQLITE3)
	add_example(sqlite_ex)
endif()

