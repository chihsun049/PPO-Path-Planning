cmake_minimum_required(VERSION 3.0.2)
project(my_robot_control)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  gazebo_ros
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/rl_agent.py
  scripts/custom_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
