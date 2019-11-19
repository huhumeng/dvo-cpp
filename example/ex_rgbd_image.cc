#include "core/interpolation.h"
#include "core/rgbd_image.h"
#include "core/dense_tracking.h"

#include "util/tic_toc.h"
#include "util/file_reader.h"

#include <iostream>
#include <iomanip>

#include <opencv2/highgui/highgui.hpp>

using namespace std;

using dvo::core::IntrinsicMatrix;
using dvo::core::RgbdCamera;
using dvo::core::RgbdCameraPtr;
using dvo::core::RgbdCameraPyramid;
using dvo::core::RgbdCameraPyramidPtr;
using dvo::core::RgbdImagePyramid;
using dvo::core::RgbdImagePyramidPtr;
using dvo::core::DenseTracker;
using dvo::core::Result;

using dvo::util::FileReader;
using dvo::util::Groundtruth;
using dvo::util::RGBDPair;

void convertRawDepthImageSse(const cv::Mat &input, cv::Mat &output, float scale)
{
    output.create(input.rows, input.cols, CV_32FC1);

    const unsigned short *input_ptr = input.ptr<unsigned short>();
    float *output_ptr = output.ptr<float>();

    __m128 _scale = _mm_set1_ps(scale);
    __m128 _zero = _mm_setzero_ps();
    __m128 _nan = _mm_set1_ps(std::numeric_limits<float>::quiet_NaN());

    for (int idx = 0; idx < input.size().area(); idx += 8, input_ptr += 8, output_ptr += 8)
    {
        __m128 _input, mask;
        __m128i _inputi = _mm_load_si128((__m128i *)input_ptr);

        // load low shorts and convert to float
        _input = _mm_cvtepi32_ps(_mm_unpacklo_epi16(_inputi, _mm_setzero_si128()));

        mask = _mm_cmpeq_ps(_input, _zero);

        // zero to nan
        _input = _mm_or_ps(_input, _mm_and_ps(mask, _nan));
        // scale
        _input = _mm_mul_ps(_input, _scale);
        // save
        _mm_store_ps(output_ptr + 0, _input);

        // load high shorts and convert to float
        _input = _mm_cvtepi32_ps(_mm_unpackhi_epi16(_inputi, _mm_setzero_si128()));

        mask = _mm_cmpeq_ps(_input, _zero);

        // zero to nan
        _input = _mm_or_ps(_input, _mm_and_ps(mask, _nan));
        // scale
        _input = _mm_mul_ps(_input, _scale);
        // save
        _mm_store_ps(output_ptr + 4, _input);
    }
}

RgbdCameraPyramidPtr camera_pyramid = nullptr;

RgbdImagePyramidPtr load(string rgb_file, string depth_file)
{
    cv::Mat rgb = cv::imread(rgb_file, 1);
    cv::Mat depth = cv::imread(depth_file, -1);

    cv::Mat grey, grey_s16, depth_mask, depth_mono, depth_float;

    bool rgb_available = false;
    if (rgb.type() != CV_32FC1)
    {
        if (rgb.type() == CV_8UC3)
        {
            cv::cvtColor(rgb, grey, CV_BGR2GRAY);
            rgb_available = true;
        }
        else
        {
            grey = rgb;
        }

        grey.convertTo(grey_s16, CV_32F);
    }
    else
    {
        grey_s16 = rgb;
    }

    if (depth.type() != CV_32FC1)
    {
        convertRawDepthImageSse(depth, depth_float, 1.035f / 5000.0f);
    }
    else
    {
        depth_float = depth;
    }

    RgbdImagePyramidPtr result(new RgbdImagePyramid(*camera_pyramid, grey_s16, depth_float));

    if (rgb_available)
        rgb.convertTo(result->level(0).rgb, CV_32FC3);

    return result;
}

string rgbd_pair_path = "/Volumes/Passport/Dataset/TUM_rgbd/rgbd_dataset_freiburg3_long_office_household/associate.txt";
string groundtruth_path = "/Volumes/Passport/Dataset/TUM_rgbd/rgbd_dataset_freiburg3_long_office_household/groundtruth.txt";
string folder = "/Volumes/Passport/Dataset/TUM_rgbd/rgbd_dataset_freiburg3_long_office_household/";

int main()
{
    FileReader<RGBDPair> rgbd_pair(rgbd_pair_path);
    FileReader<Groundtruth> groundtruth(groundtruth_path);

    std::vector<RGBDPair> rgbd_pairs;
    std::vector<Groundtruth> groundtruths;

    rgbd_pair.skipComments();
    rgbd_pair.readAllEntries(rgbd_pairs);

    groundtruth.skipComments();
    groundtruth.readAllEntries(groundtruths);

    IntrinsicMatrix matrix = IntrinsicMatrix::create(535.4, 539.2, 320.1, 247.6);

    RgbdCamera camera(640, 480, matrix);

    camera_pyramid.reset(new RgbdCameraPyramid(camera));


    DenseTracker tracker(DenseTracker::getDefaultConfig());

    
    RgbdImagePyramidPtr reference, current;
    Eigen::Affine3d trajectory;
    trajectory.setIdentity();

    ofstream fout("estimated.txt");
    fout << setprecision(16);

    for (const auto &pair : rgbd_pairs)
    {   
        reference = current;
        current = load(folder + pair.rgb_file_, folder + pair.depth_file_);

        if(reference == nullptr){
            continue;
        }

        Result result;
        tracker.match(*reference, *current, result);

        trajectory = trajectory * (result.Transformation);
        
        Groundtruth gt;
        gt.timestamp_ = pair.rgb_timestamp_;
        gt.postion_x_ = trajectory.translation()(0);
        gt.postion_y_ = trajectory.translation()(1);
        gt.postion_z_ = trajectory.translation()(2);

        Eigen::Quaterniond q(trajectory.rotation());
        gt.orientation_w_ = q.w();
        gt.orientation_x_ = q.x();
        gt.orientation_y_ = q.y();
        gt.orientation_z_ = q.z();

        fout << gt;
        // cout << result.Transformation.matrix() << endl;
        // cv::Mat gray;
        // // cv::cvtColor(image->level(0).intensity, gray, cv::COLOR_G)
        // image->level(0).intensity.convertTo(gray, CV_8UC1);

        // cv::imshow("gray", gray);

        // if ('q' == cv::waitKey(1))
        //     break;
    }

    fout.close();
    return 0;
}