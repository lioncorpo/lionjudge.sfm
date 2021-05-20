#pragma once
#include <opencv2/core.hpp>
#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>

namespace cslam
{
    struct GridParameters;
}
namespace cslam
{
class Frame;
struct Camera
{
    Camera(const size_t width_, const size_t height_, const std::string& projection_type_):
        width(width_), height(height_), projectionType(projection_type_)
    {}

    const size_t width;
    const size_t height;
    const std::string projectionType;
    virtual void undistKeyptsFrame(Frame& frame) const = 0;
    virtual void convertKeyptsToBearingsFrame(Frame& frame) const = 0;
    virtual void convertKeyptsToBearings(const std::vector<cv::KeyPoint>& keypts, std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& bearings) const = 0;
    virtual void undistKeypts(const std::vector<cv::KeyPoint>& keypts, std::vector<cv::KeyPoint>& undist_keypts) const = 0;
    virtual bool reproject_to_image(const Eigen::Matrix3f& R_cw, const Eigen::Vector3f& t_cw, const Eigen::Vector3f& ptWorld,
                                    const cslam::GridParameters& gridParams, Eigen::Vector2f& pt2D) const = 0;
};

/**
 * Define a perspective camera.

    Attributes:
        width (int): image width.
        height (int): image height.
        focal_x (real): estimated focal length for the X axis.
        focal_y (real): estimated focal length for the Y axis.
        c_x (real): estimated principal point X.
        c_y (real): estimated principal point Y.
        k1 (real): estimated first radial distortion parameter.
        k2 (real): estimated second radial distortion parameter.
        p1 (real): estimated first tangential distortion parameter.
        p2 (real): estimated second tangential distortion parameter.
        k3 (real): estimated third radial distortion parameter.
**/
struct BrownPerspectiveCamera: Camera
{
    BrownPerspectiveCamera(const size_t width_, const size_t height_, const std::string& projection_type_,
                           const float fx_, const float fy_, const float cx_, const float cy_,
                           const float k1_, const float k2_, const float p1_, const float p2_, const float k3_): 
        Camera(width_, height_, projection_type_), fx(fx_), fy(fy_), cx(cx_), cy(cy_),
        k1(k1_), k2(k2_), p1(p1_), p2(p2_), k3(k3_)
    {
        K = (cv::Mat_<float>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        distCoeff = (cv::Mat_<float>(1,5) << k1, k2, p1, p2, k3);

        const auto s = std::max(width,height);
        cv::Mat norm_mat = (cv::Mat_<float>(3,3) << s, 0, 0.5*(width-1), 0, s, 0.5*(height-1), 0, 0, 1);
        
        K_pixel = norm_mat*K; //(cv::Mat_<float>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        cv::cv2eigen(K_pixel, K_pixel_eig);
        fx_p = K_pixel_eig(0,0);
        fy_p = K_pixel_eig(1,1);
        cx_p = K_pixel_eig(0,2);
        cy_p = K_pixel_eig(1,2);

    }
    float fx_p, fy_p; // focal lengths in pixels
    float cx_p, cy_p; // principal points in pixels
    
    const float fx, fy; // focal lengths
    const float cx, cy; // principal points
    const float k1, k2, p1, p2, k3; // distortion coefficients
    cv::Mat K, K_pixel; //intrinsic camera matrix
    cv::Mat distCoeff; //distortion coefficients
    Eigen::Matrix3f K_pixel_eig;
    virtual void undistKeypts(const std::vector<cv::KeyPoint>& keypts, std::vector<cv::KeyPoint>& undist_keypts) const;
    virtual void undistKeyptsFrame(Frame& frame) const;
    virtual void convertKeyptsToBearingsFrame(Frame& frame) const;
    virtual void convertKeyptsToBearings(const std::vector<cv::KeyPoint>& keypts, std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& bearings) const;
    virtual bool reproject_to_image(const Eigen::Matrix3f& R_cw, const Eigen::Vector3f& t_cw, const Eigen::Vector3f& ptWorld,
                                    const cslam::GridParameters& gridParams, Eigen::Vector2f& pt2D) const;
    virtual bool reproject_to_bearing(const Eigen::Matrix3f& R_cw, const Eigen::Vector3f& t_cw, const Eigen::Vector3f& ptWorld,
                                      const cslam::GridParameters& gridParams, Eigen::Vector3f& bearing) const;
    // virtual bool reproject_to_image_dist(const Eigen::Matrix3f& R_cw, const Eigen::Vector3f& t_cw, const Eigen::Vector3f& ptWorld,
                                        //  const cslam::GridParameters& gridParams, Eigen::Vector2f& pt2D) const;

};


}