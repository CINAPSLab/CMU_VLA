cmake_minimum_required(VERSION 3.0.2)
project(cinaps_vlm)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  visualization_msgs
  cv_bridge
  geometry_msgs
  nav_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/world_model_node
  scripts/qa_node
  scripts/exploration_manager
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
