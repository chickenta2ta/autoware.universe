#ifndef TENSORRT_LIGHTNET__TENSORRT_LIGHTNET_NODE_HPP_
#define TENSORRT_LIGHTNET__TENSORRT_LIGHTNET_NODE_HPP_

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include "./tensorrt_lightnet.hpp"

#include <sensor_msgs/msg/image.hpp>

namespace tensorrt_lightnet
{
class TrtLightNetNode : public rclcpp::Node
{
public:
  explicit TrtLightNetNode(const rclcpp::NodeOptions & node_options);

private:
  void onConnect();
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  image_transport::Publisher image_pub_;

  image_transport::Subscriber image_sub_;

  rclcpp::TimerBase::SharedPtr timer_;
  std::unique_ptr<tensorrt_lightnet::TrtLightNet> trt_lightnet_;
};

}

#endif