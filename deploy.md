ssh agi@192.168.1.100

cd /home/agi/a3_remote_bridge_probe

A3_ENABLE_COMMAND_PUBLISH=1 \
A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION \
A3_ROBOT_SAFETY_READY=1 \
A3_STATUS_PERIOD_S=1 \
./scripts/run_mdu_planner_receiver.sh


ssh agi@192.168.1.100

sudo systemctl stop agibot_pm
source /agibot/software/v0/entry/env/env.sh
cd /agibot/software/v0
bash scripts/hal_ethercat/start_hal_ethercat.sh

env   ROS_DOMAIN_ID=232   ROS_LOCALHOST_ONLY=1   ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST   RMW_IMPLEMENTATION=rmw_fastrtps_cpp   FASTRTPS_DEFAULT_PROFILES_FILE=/home/bth/workspace/a3_remote_bridge/config/fastrtps_pc_local.xml   ros2 launch ppmocap_driver ppmocap.launch.py     rviz:=true     camera:=true

A3_NEW_TASK_TTS_MIN_S=0.20 A3_NEW_TASK_TTS_MAX_S=1.00 A3_STRIKE_Y_MARGIN_M=0.08 A3_STRIKE_Z_MARGIN_M=0.08 A3_RACKET_VELOCITY_MARGIN_MPS=0.15 A3_PC_WIRED_ADDRESS=192.168.1.11 A3_MDU_ADDRESS=192.168.1.100 A3_PLANNER_UDP_PORT=15001 ./scripts/run_local_receive_planner.sh
