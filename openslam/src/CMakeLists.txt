cmake_minimum_required(VERSION 3.0)

project(openslam C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_MODULE_PATH ${openslam_SOURCE_DIR}/cmake)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} \
    -fPIC \
    -fvisibility=hidden")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
#     -fPIC \
#     -fvisibility=hidden \
#     -fvisibility-inlines-hidden \
#     -std=c++14 \
#     -Wextra \
#     -fsanitize=address \
#     -fsanitize-address-use-after-scope \
#     -Wall")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
    -fPIC \
    -fvisibility=hidden \
    -fvisibility-inlines-hidden \
    -std=c++14 \
    -Wextra \
    -Wall -O3")
    
    # -fsanitize=address \
    # -fsanitize-address-use-after-scope \
    # -static-libasant")
    
    # \
    # -O3")

# Find dependencies.
# add_subdirectory(${opensfm_SOURCE_DIR}/third_party/pybind11)
add_subdirectory(${openslam_SOURCE_DIR}/third_party/pybind11)

MESSAGE("${openslam_SOURCE_DIR}/../../opensfm/src/third_party/pybind11 ${openslam_SOURCE_DIR}/third_party/pybind11")
include_directories(${PYTHON_INCLUDE_DIRS})

# find_package(OpenMP)
# set(OPENMP_FOUND 0)
# if (OPENMP_FOUND)
#   set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# endif()

# find_package(LAPACK)
# if (LAPACK_FOUND)
#   include_directories(${LAPACK_INCLUDE_DIRS})
# endif()

# find_package(SuiteSparse)
# if (SUITESPARSE_FOUND)
#   include_directories(${SUITESPARSE_INCLUDE_DIRS})
# endif()

# find_package(Eigen REQUIRED)
# include_directories(${EIGEN_INCLUDE_DIRS})

# find_package(Ceres REQUIRED)
# include_directories(${CERES_INCLUDE_DIR})

# find_package(Gflags REQUIRED)
# include_directories(${GFLAGS_INCLUDE_DIR})

# find_package(Glog REQUIRED)
# include_directories(${GLOG_INCLUDE_DIR})

find_package(OpenCV)
# OpenCV's OpenCVConfig will enforce imgcodecs for < 3.0 
# (even if OPTIONAL_COMPONENTS) so we remove it as we don't need it
# Cause is imread/imwrite moved to imgcodecs on > 3.0
if(${OpenCV_VERSION} LESS 3.0)
    find_package(OpenCV REQUIRED core imgproc calib3d highgui)
else()
    find_package(OpenCV REQUIRED core imgproc calib3d highgui OPTIONAL_COMPONENTS imgcodecs)
endif()
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${openslam_SOURCE_DIR})


# # Python wrapper
# pybind11_add_module(orb_extractor
#     third_party/orb_extractor_py.cc
#     third_party/openvslam/feature/orb_extractor.cc
#     third_party/openvslam/feature/orb_extractor.h
#     third_party/openvslam/feature/orb_params.h
#     third_party/openvslam/feature/orb_params.cc
#     third_party/openvslam/feature/orb_extractor_node.h
#     third_party/openvslam/feature/orb_extractor_node.cc
#     third_party/openvslam/feature/orb_point_pairs.h
#     ${openslam_SOURCE_DIR}/types.h
#     ${openslam_SOURCE_DIR}/types.cc
#     slam_datastructures/frame.h
#     slam_datastructures/frame.cc
#     # ${openslam_SOURCE_DIR}/../../opensfm/src/types.h
# )
# pybind11_add_module(guided_matching
#     third_party/openvslam/util/angle_checker.h
#     third_party/openvslam/util/guided_matching.h
#     third_party/openvslam/util/guided_matching.cc
#     ${openslam_SOURCE_DIR}/guided_matching_utils_py.cc
    
# )


# pybind11_add_module(cslam_types
#     ${openslam_SOURCE_DIR}/cslam_types.cc
#     slam_datastructures/frame.h
#     slam_datastructures/frame.cc
#     slam_datastructures/keyframe.h
#     slam_datastructures/keyframe.cc
#     slam_datastructures/landmark.h
#     slam_datastructures/landmark.cc
#     slam_datastructures/camera.cc
#     slam_datastructures/camera.h
#     # third_party/openvslam/feature/orb_extractor.cc
#     # third_party/openvslam/feature/orb_extractor.h
#     # ${openslam_SOURCE_DIR}/types.h
#     # ${openslam_SOURCE_DIR}/types.cc
# )
# # pybind11_add_libr

# find_package(Eigen REQUIRED)
# include_directories(${EIGEN_INCLUDE_DIRS})

# target_link_libraries(orb_extractor PRIVATE
#     ${OpenCV_LIBS}
# )
# target_link_libraries(guided_matching PRIVATE
#     ${OpenCV_LIBS}
# )
# target_link_libraries(cslam_types PRIVATE
#     ${OpenCV_LIBS}
#     ./orb_extractor.o
# )
# Python wrapper
pybind11_add_module(cslam
    ${openslam_SOURCE_DIR}/cslam.cc
    # third_party/orb_extractor_py.cc
    ${openslam_SOURCE_DIR}/third_party/openvslam/feature/orb_extractor.cc
    ${openslam_SOURCE_DIR}/third_party/openvslam/feature/orb_extractor.h
    ${openslam_SOURCE_DIR}/third_party/openvslam/feature/orb_params.h
    ${openslam_SOURCE_DIR}/third_party/openvslam/feature/orb_params.cc
    ${openslam_SOURCE_DIR}/third_party/openvslam/feature/orb_extractor_node.h
    ${openslam_SOURCE_DIR}/third_party/openvslam/feature/orb_extractor_node.cc
    ${openslam_SOURCE_DIR}/third_party/openvslam/feature/orb_point_pairs.h
    ${openslam_SOURCE_DIR}/third_party/openvslam/util/angle_checker.h
    ${openslam_SOURCE_DIR}/third_party/openvslam/util/guided_matching.h
    ${openslam_SOURCE_DIR}/third_party/openvslam/util/guided_matching.cc
    ${openslam_SOURCE_DIR}/third_party/openvslam/data/graph_node.cc
    ${openslam_SOURCE_DIR}/third_party/openvslam/data/graph_node.h
    ${openslam_SOURCE_DIR}/types.h
    ${openslam_SOURCE_DIR}/types.cc
    ${openslam_SOURCE_DIR}/types.h
    ${openslam_SOURCE_DIR}/slam_utilities.h
    ${openslam_SOURCE_DIR}/slam_utilities.cc
    ${openslam_SOURCE_DIR}/slam_datastructures/frame.h
    ${openslam_SOURCE_DIR}/slam_datastructures/frame.cc
    ${openslam_SOURCE_DIR}/slam_datastructures/keyframe.h
    ${openslam_SOURCE_DIR}/slam_datastructures/keyframe.cc
    ${openslam_SOURCE_DIR}/slam_datastructures/landmark.h
    ${openslam_SOURCE_DIR}/slam_datastructures/landmark.cc
    ${openslam_SOURCE_DIR}/slam_datastructures/camera.cc
    ${openslam_SOURCE_DIR}/slam_datastructures/camera.h
    ${openslam_SOURCE_DIR}/slam_debug.h
    ${openslam_SOURCE_DIR}/slam_debug.cc
    ${openslam_SOURCE_DIR}/slam_datastructures/local_map_cleaner.h
    ${openslam_SOURCE_DIR}/slam_datastructures/local_map_cleaner.cc
    ${openslam_SOURCE_DIR}/slam_datastructures/slam_reconstruction.h
    ${openslam_SOURCE_DIR}/slam_datastructures/slam_reconstruction.cc
)
# pybind11_add_libr

find_package(Eigen REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})

# target_link_libraries(orb_extractor PRIVATE
#     ${OpenCV_LIBS}
# )
# target_link_libraries(guided_matching PRIVATE
#     ${OpenCV_LIBS}
# )
target_link_libraries(cslam PRIVATE
    ${OpenCV_LIBS}
    # ./orb_extractor.o
)
# set_target_properties(orb_extractor PROPERTIES
#     LIBRARY_OUTPUT_DIRECTORY "${openslam_SOURCE_DIR}/.."
# )

# set_target_properties(guided_matching PROPERTIES
#     LIBRARY_OUTPUT_DIRECTORY "${openslam_SOURCE_DIR}/.."
# )

set_target_properties(cslam PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${openslam_SOURCE_DIR}/.."
)

# if (CMAKE_BUILD_TYPE MATCHES DEBUG)
#     add_executable(debug_c_extension
#         debug_c_extension.cc
#         third_party/PythonExtensionPatterns/py_import_call_execute.c
#     )
#     target_link_libraries(debug_c_extension ${PYTHON_LIBRARIES})
# endif()