cmake_minimum_required(VERSION 2.8.3)

project(0000014503_A3T2.5-URDF-std-pingpang-0409)

find_package(catkin REQUIRED)

catkin_package()

find_package(roslaunch)

foreach(dir config launch meshes urdf)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
