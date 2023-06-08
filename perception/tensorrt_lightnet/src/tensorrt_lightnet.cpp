#include "./modules/class_detector.h"

#include "./tensorrt_lightnet.hpp"

namespace tensorrt_lightnet
{
TrtLightNet::TrtLightNet(
)
{
  // initialize a detector
  ::Config config;
  config.file_model_cfg = "./configs/lightNet-BDD100K-det-semaseg-1280x960.cfg";
  config.file_model_weights = "./weights/lightNet-BDD100K-det-semaseg-1280x960.weights";
  config.batch = 1;
  config.width = 1280;
  config.height = 960;
  std::unique_ptr<::Detector> detector(new ::Detector());
  detector->init(config);
}

bool TrtLightNet::doInference(const std::vector<cv::Mat> & images)
{
  detector->segment(images);
  return true;
}
}