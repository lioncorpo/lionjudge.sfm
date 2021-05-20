#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "types.h"
namespace cslam
{
class Frame;
// class BrownPerspectiveCamera;
class SlamDebug
{
public:
static void
print_matches_from_lms(const cslam::Frame& frame1, const cslam::Frame& frame2,
                       const csfm::pyarray_uint8 image1, const csfm::pyarray_uint8 image2);

// static void
// reproject_last_lms(const Frame& frame1, const Frame& frame2,
//                    const csfm::pyarray_uint8 image2, const BrownPerspectiveCamera& camera);
};
}