cmake_minimum_required(VERSION 3.0.2)
project(hio_sdf)


find_package(catkin REQUIRED COMPONENTS
  rospy
  tf2
  std_msgs
  nav_msgs
  visualization_msgs
  sensor_msgs
  geometry_msgs
  message_generation
)

catkin_python_setup()

add_service_files(
  FILES
  ShareWeights.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package()


install(PROGRAMS
   nodes/*
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


