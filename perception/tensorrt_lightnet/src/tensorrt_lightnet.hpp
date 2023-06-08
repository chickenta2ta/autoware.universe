#ifndef TENSORRT_LIGHTNET__TENSORRT_LIGHTNET_HPP_
#define TENSORRT_LIGHTNET__TENSORRT_LIGHTNET_HPP_

#include <opencv2/opencv.hpp>

#include <vector>

namespace tensorrt_lightnet
{
class TrtLightNet
{
public:
  TrtLightNet();

  bool doInference(const std::vector<cv::Mat> & images)
}
}

#endif