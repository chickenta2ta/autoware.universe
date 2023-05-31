cmake_minimum_required(VERSION 3.14)
project(autoware_auto_msgs_adapter)

find_package(autoware_cmake REQUIRED)
autoware_package()

set(NODE_NAME ${PROJECT_NAME}_node)
set(EXEC_NAME ${PROJECT_NAME}_exe)

ament_auto_add_library(${NODE_NAME} SHARED
  src/autoware_auto_msgs_adapter_core.cpp
  include/autoware_auto_msgs_adapter/autoware_auto_msgs_adapter_core.hpp)

rclcpp_components_register_node(${NODE_NAME}
  PLUGIN "autoware_auto_msgs_adapter::AutowareAutoMsgsAdapterNode"
  EXECUTABLE ${EXEC_NAME})

ament_auto_package(INSTALL_TO_SHARE
  launch
  config)
