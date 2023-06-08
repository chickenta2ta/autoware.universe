#include "./tensorrt_lightnet_node.hpp"

namespace tensorrt_lightnet
{
TrtLightNetNode::TrtLightNetNode(const rclcpp::NodeOptions & node_options)
: Node("tensorrt_lightnet", node_options)
{
  using std::placeholders::_1;
  using std::chrono_literals::operator""ms;

  trt_lightnet_ = std::make_unique<tensorrt_lightnet::TrtLightNet>();

  timer_ =
    rclcpp::create_timer(this, get_clock(), 100ms, std::bind(&TrtLightNetNode::onConnect, this));

  image_pub_ = image_transport::create_publisher(this, "~/out/image");
}

void TrtLightNetNode::onConnect()
{
  using std::placeholders::_1;
  if (
    image_pub_.getNumSubscribers() == 0) {
    image_sub_.shutdown();
  } else if (!image_sub_) {
    image_sub_ = image_transport::create_subscription(
      this, "~/in/image", std::bind(&TrtLightNetNode::onImage, this, _1), "raw",
      rmw_qos_profile_sensor_data);
  }
}

void TrtLightNetNode::onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  cv_bridge::CvImagePtr in_image_ptr;
  try {
    in_image_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  if (!trt_lightnet_->doInference({in_image_ptr->image})) {
    RCLCPP_WARN(this->get_logger(), "Fail to inference");
    return;
  }

  image_pub_.publish(in_image_ptr->toImageMsg());
}

}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(tensorrt_lightnet::TrtLightNetNode)