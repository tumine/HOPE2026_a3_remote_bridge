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
S：进入 PD 站立位，同时将双臂/夹爪平滑略微抬起
M：启动下肢策略
R：进入发球 Home/Ready
C：关闭夹爪
F：播放旧版插值发球动作
V：播放离线 IK 发球动作
G：打开夹爪
Y：在 Idle 或 Home/Ready 状态动态重载当前发球 YAML；修改
   `config/a3_serve_ik.yaml` 后按 Y 即可，无需重启程序
H：打印帮助
Q：退出程序

数字轨迹选择：

1. 先按 `S` 进入 PD 站立模式。
2. 按 `1`～`6`，程序读取
   `/home/agi/a3_lower_body_serve_probe/tracks/<数字>.yaml`，并自动将手臂
   平滑移动到该轨迹的 Home 初始位。
3. 等待终端显示 `arm: ready/home`，然后继续按 `C` 关夹爪、按 `F`
   执行当前数字轨迹。数字 YAML 缺失或校验失败时保留当前轨迹且不动作。

编号与原始文件的对应关系见 `Serve_A3_leg_model/tracks/README.md`。


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



ssh agi@10.231.32.183

#部署指令

#启动动捕
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
  
  
#启动本地planner
cd workspace/a3_remote_bridge
A3_NEW_TASK_TTS_MIN_S=0.25 A3_NEW_TASK_TTS_MAX_S=1.00 A3_STRIKE_Y_MARGIN_M=0.0 A3_STRIKE_Z_MARGIN_M=0.0 A3_RACKET_VELOCITY_MARGIN_MPS=0.0 A3_PC_WIRED_ADDRESS=192.168.1.11 A3_MDU_ADDRESS=192.168.1.100 A3_PLANNER_UDP_PORT=15001 ./scripts/run_local_receive_planner.sh

#mdu命令
cd /home/agi/a3_remote_bridge_probe
sudo systemctl stop agibot_pm #停止服务

A3_ENABLE_COMMAND_PUBLISH=1 \
A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION \
A3_GRIPPER_ACTUATION_CONFIRM=ENABLE_A3_GRIPPER \
A3_ROBOT_SAFETY_READY=1 \
A3_STATUS_PERIOD_S=1 \
./scripts/run_mdu_planner_receiver.sh


#下肢发球策略
cd /home/agi/a3_lower_body_serve_probe

A3_TRANSPORT=iceoryx \
A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION \
A3_GRIPPER_ACTUATION_CONFIRM=ENABLE_A3_GRIPPER \
A3_ROBOT_SAFETY_READY=1 \
./run_hope_lower_body.sh

#球体物理
/home/bth/workspace/a3_remote_bridge/model_72500_deploy_bundle/config/ball_physics.yaml


#夹爪
vi /home/agi/a3_remote_bridge_probe/config/serve_tracks/1.yaml

 cd /home/agi/a3_remote_bridge_probe

A3_GRIPPER_OPEN_POSITION=4096 \
A3_GRIPPER_CLOSE_POSITION=350 \
A3_ENABLE_COMMAND_PUBLISH=1 \
A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION \
A3_GRIPPER_ACTUATION_CONFIRM=ENABLE_A3_GRIPPER \
A3_ROBOT_SAFETY_READY=1 \
./scripts/run_mdu_planner_receiver.sh  



