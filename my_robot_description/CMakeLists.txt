cmake_minimum_required(VERSION 3.0.2)
project(my_robot_description)

find_package(catkin REQUIRED COMPONENTS
  rospy
  gazebo_ros
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY launch worlds models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
