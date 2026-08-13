cmake_minimum_required(VERSION 3.8)
project(trajectory_avoidance)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/trajectory_avoidance_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
