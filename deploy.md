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


cd /home/agi/a3_remote_bridge_probe

A3_ENABLE_COMMAND_PUBLISH=1 \
A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION \
A3_GRIPPER_ACTUATION_CONFIRM=ENABLE_A3_GRIPPER \
A3_ROBOT_SAFETY_READY=1 \
A3_STATUS_PERIOD_S=1 \
./scripts/run_mdu_planner_receiver.sh



  env \
    ROS_DOMAIN_ID=232 \
    ROS_LOCALHOST_ONLY=1 \
    ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    FASTRTPS_DEFAULT_PROFILES_FILE=/home/bth/workspace/a3_remote_bridge/config/fastrtps_pc_local.xml \
    ros2 launch ppmocap_driver ppmocap.launch.py \
    site_params_file:=/home/bth/workspace/Mocap/install/ppmocap_driver/share/ppmocap_driver/config/sites/competition.yaml \
    rviz:=true \
    camera:=true

A3_NEW_TASK_TTS_MIN_S=0.25 A3_NEW_TASK_TTS_MAX_S=1.00 A3_STRIKE_Y_MARGIN_M=0.0 A3_STRIKE_Z_MARGIN_M=0.0 A3_RACKET_VELOCITY_MARGIN_MPS=0.0 A3_PC_WIRED_ADDRESS=192.168.1.11 A3_MDU_ADDRESS=192.168.1.100 A3_PLANNER_UDP_PORT=15001 ./scripts/run_local_receive_planner.sh

#旧的启动脚本
/home/agi/a3_remote_bridge_probe/scripts/run_mdu_planner_receiver.sh.pre_auto_hal_20260821
/home/agi/a3_lower_body_serve_probe/run_hope_lower_body.sh.pre_auto_hal_20260821
#test 命令
env \
  ROS_DOMAIN_ID=232 \
  ROS_LOCALHOST_ONLY=1 \
  ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  FASTRTPS_DEFAULT_PROFILES_FILE=/home/bth/workspace/a3_remote_bridge/config/fastrtps_pc_local.xml \
  ros2 launch ppmocap_driver ppmocap.launch.py \
    rviz:=true \
    camera:=true
    
    
    cd /home/bth/workspace/Mocap

env \
  ROS_DOMAIN_ID=232 \
  ROS_LOCALHOST_ONLY=1 \
  ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  FASTRTPS_DEFAULT_PROFILES_FILE=/home/bth/workspace/a3_remote_bridge/config/fastrtps_pc_local.xml \
  ./scripts/start_ppmocap.sh test
  
  
  source /agibot/software/v0/entry/env/env.sh
cd /agibot/software/v0
bash scripts/hal_elink/start_hal_elink.sh

下肢发球策略
cd /home/agi/a3_lower_body_serve_probe

A3_TRANSPORT=iceoryx \
A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION \
A3_GRIPPER_ACTUATION_CONFIRM=ENABLE_A3_GRIPPER \
A3_ROBOT_SAFETY_READY=1 \
./run_hope_lower_body.sh


#hhh
P：立即切换到阻尼模式
S：进入 PD 初始
M：启动下肢策略
V：进入发球 Ready
C：关闭夹爪
F：播放发球动作
G：打开夹爪
H：打印帮助
Q：退出程序


#策略
sudo systemctl stop agibot_pm #重启后执行，停止服务

cd /home/agi/a3_lower_body_serve_probe
A3_TRANSPORT=iceoryx \
A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION \
A3_GRIPPER_ACTUATION_CONFIRM=ENABLE_A3_GRIPPER \
A3_ROBOT_SAFETY_READY=1 \
./run_hope_lower_body.sh #控制命令行

cd /home/bth/workspace/a3_remote_bridge
./scripts/run_lower_body_serve_mujoco.sh #先测试mujoco

cd /home/bth/workspace/a3_remote_bridge
./scripts/run_lower_body_serve_mujoco.sh

