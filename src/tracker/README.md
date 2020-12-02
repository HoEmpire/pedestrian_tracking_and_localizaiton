# object_detector

<div align=center><img width="600" height="400" src="img/demo.gif"/></div>

## Content

- [Introduction](#Introduction)
- [Dependence](#Dependence)
- [Build](#Build)
- [Usage](#Usage)

## Introduction

点云聚类和图像识别。

## Prerequisite

- Ubuntu 18.04
- ROS Melodic
- PCL 1.7.0+
- Opencv 4.0.0+
- [usv-sensor-fusion-system](https://gitee.com/csc105/usv-sensor-fusion-system)

## Build

1. Build a workspace

   ```shell
   mkdir ws/src
   ```

2. Clone this repository and prerequisite package in `/ws/src`

   ```shell
   cd ws/src
   git clone https://gitee.com/csc105/usv-sensor-fusion-system.git
   git clone https://github.com/HoEmpire/pointcloud-fusion.git
   ```

3. Build the files

   ```shell
   catkin_make
   ```

## Usage

### 1. Usage on the stabilization platform

1.  本地监视

    用网线连接云台，设置 IP 为`222.31.31.X`，在本地 terminal 输入以下语句

    ```shell
    export ROS_MASTER_URI=http://222.31.31.50:11311
    ```

    设置本地 ros master 为云台，即可在本地检测数据。可视化可打开`rviz`，使用`object_detector/rviz/debug.rviz`的配置进行监视。

2.  NX 上手动启动代码

    先关闭开机自启动的 service, 后在`object_detecor`文件夹内重新启动所有代码

    ```shell
    systemctl stop auto-start.service
    ./shell/start_onboard.sh
    ```

3.  外参标定

    将外参标定得到的`.csv`文件改名为`data.csv`，将`data.csv`放入 matlab 工作空间内，在 matlab 内运行`object_detector/matlab/data_process.m`，即可得到 yaw 外参的标定结果（角度值）

4.  参数说明

    参数文件主要为`obejct_detector.yaml`
    说明如下

    - 数据的 topic

      ```yaml
      data_topic:
        lidar_topic: /livox/lidar #激光雷达topic
        camera_topic: /pointgrey/image_color # 相机topic
      ```

    - 聚类后检测相关参数

      $$距离系数=\frac{当前距离}{校准距离}$$

      检测最小阈值与距离系数成二次关系，检测最大阈值与距离系数成线性关系，如 points_number_distance_coeff =100，points_number_max = 5000，points_number_min = 80， 则在 50m 处的聚类后点云数量为 320-10000 点可视为目标，400m 处的聚类后点云数量为 20-2500 点可视为目标

      ```yaml
      detection:
        points_number_max: 5000 #在points_number_distance_coeff距离下最大的目标检测点数
        points_number_min: 9 #在points_number_distance_coeff距离下最小的目标检测点数
        points_number_distance_coeff: 300.0 # 目标检测校准距离，单位为m
        mode: "lidar_first" #检测方法，参数为“camera_first” or  "lidar_first"，前者暂时禁用
        boat_pixel_height: 80 #船检测的最小二维图像像素高度，用于滤除假目标和远处船
      ```

    - 点云滤波相关参数

      ```yaml
      filter:
        filter_pt_num: 1 #statisticalfilter的一个batch点的数量
        filter_std: 10.0 #statisticalfilter的方差阈值
        filter_distance_max: 600.0 #检测点云的最远距离（单位：m）
        filter_distance_min: 100.0 #检测点云的最近距离（单位：m）
      ```

    - 聚类参数

      ```yaml
      cluster:
        cluster_tolerance: 2.0 #聚类距离的最大阈值（单位m）
        cluster_size_min: 5 #聚类的最小点云数量
        cluster_size_max: 5000 ##聚类的最大点云数量
      ```

    - yolo 参数

      ```yaml
      cam:
        cam_net_type: "YOLOV4_TINY" #网络类型
        cam_file_model_cfg: "/asset/yolov4-tiny-usv.cfg" #config文件地址
        cam_file_model_weights: "/asset/yolov4-tiny-usv_best.weights" #权重文件地址
        cam_inference_precison: "FP32" #浮点数精度，NX 上设置 WieFP16
        cam_n_max_batch: 1 #batch数量
        cam_prob_threshold: 0.5 #图像检测的可信度阈值，设置为0~1
        cam_min_width: 0 #图像检测的最大最小宽度和高度
        cam_max_width: 1440
        cam_min_height: 0
        cam_max_height: 1080
      ```

    - 外参参数

      ```yaml
      extrinsic_parameter:
        translation: [0.0, 0.0, 0.0] #外参的[x.,y,z]位移，单位为m
        rotation: [0.0, 0.0, 0.0] #外参[roll, pitch, yaw]旋转，单位为degree
      ```

    - 云台指令参数

      ```yaml
      platform:
        scan_cycle_time: 20.0 #转动一个周期的时间
        scan_range: 80.0 #总零位转动的最大角度
        track_timeout: 10.0 #从目指切换到周扫的等待时间
      ```

5.  云台调试相关

    1. 打开 GUI 调试界面

       ```
       cd /home/nvidia/SimpleBGC_GUI_2_70b0
       sudo ./run.sh
       ```

    2. PID 调整

       1. 点击左上角的 connect 连接云台（需要先停止代码，详见 2. 手动调试中关闭开机自启动 service 的部分）
       <div align=center><img width="720" height="540" src="img/start.png"/></div>

       2. 在下面的窗口内调整各轴的 PID
       <div align=center><img width="720" height="540" src="img/PID.png"/></div>

       3. 调好后点击右下角的 write 写入

       4. 可进入 Monitoring 监视调试效果，主要监视 `ERROR_ROLL`, `ERROR_PITCH`, `ERROR_YAW`，调试方法为电机左下角`MOTOR ON/OFF`断电后上电，同时监视误差大小，观察是否震荡，超调大小以及响应速度
       <div align=center><img width="720" height="540" src="img/Monitor.png"/></div>

       **PID 调试常见注意事项**

       - 出现高频震荡，则减小该轴的 D
       - yaw 轴为了能顺滑运动，建议调整得到的曲线没有超调后者超调较小，通过减小 P 值和 I 值来保证
       - 如需完整调试 PID 参数参见[官方文档](https://www.basecamelectronics.com/files/v3/SimpleBGC_32bit_manual_2_6x_chn.pdf)10-11 页
