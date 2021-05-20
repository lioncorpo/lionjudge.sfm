cmake_minimum_required(VERSION 3.0)

project(test_datastructures C CXX)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

set(CMAKE_MODULE_PATH ${test_datastructures_SOURCE_DIR}/cmake)

####### Compilation Options #######
# Visibility stuff
cmake_policy(SET CMP0063 NEW)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_INLINES ON)

# fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable all warnings
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
find_package(OpenCV 3 REQUIRED opencv_core opencv_imgproc opencv_highgui)# opencv_calib3d opencv_videoio opencv_imgcodecs )
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

if (OpenCV_FOUND)
        message("--- found OpenCV, compiling dso_opencv library.")
        include_directories( ${OpenCV_INCLUDE_DIRS} )
        set(HAS_OPENCV 1)
endif ()

list(APPEND LIBS ${OpenCV_LIBS})

find_package(Eigen3 REQUIRED)
# # add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../src/map)
set(MAP_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/map/shot.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/map/landmark.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/map/manager.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/map/defines.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/map/src/landmark.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/map/src/manager.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/map/src/shot.cc
)

include_directories(${EIGEN3_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/map 
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/map/src)

add_library(map STATIC ${MAP_FILES})
target_include_directories(map PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
add_executable(${PROJECT_NAME} test_datastructures.cc)
target_link_libraries(${PROJECT_NAME} PUBLIC map ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src)