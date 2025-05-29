#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose
from pyquaternion import Quaternion as PyQuaternion
import json
import datetime
import os
import signal
import sys
import numpy as np
import time
from threading import Thread, Lock
from rclpy.callback_groups import ReentrantCallbackGroup
from pymoveit2 import MoveIt2, MoveIt2State
# 假设您有gripper控制模块
# from .gen3lite_pymoveit2 import Gen3LiteGripper


# 机械臂控制参数
DEFAULT_Z_HEIGHT = 0.3     # 默认高度 (安全高度)
PICK_Z_HEIGHT = 0.115      # 抓取高度
PUT_Z_X = 0.16             # 放置位置X
PUT_Z_Y = 0.3              # 放置位置Y
PUT_Z_HEIGHT = 0.214       # 放置高度
SAFE_Z_HEIGHT = 0.3        # 安全移动高度

# 机器人状态常量
class RobotStatus:
    WAITING_FOR_HELP = "waiting_for_help"
    MOVING_TO_TARGET = "moving_to_target"
    GRASPING = "grasping"
    MOVING_TO_RELEASE = "moving_to_release"
    RELEASING = "releasing"
    RETURNING_HOME = "returning_home"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"


class HelpTaskExecutor(Node):
    def __init__(self):
        super().__init__('help_task_executor')
        self.callback_group = ReentrantCallbackGroup()
        
        # 状态跟踪
        self.current_status = RobotStatus.WAITING_FOR_HELP
        self.status_lock = Lock()
        self.task_in_progress = False
        
        # 创建订阅器，监听 /require_help 话题
        self.help_subscription = self.create_subscription(
            String,
            '/require_help',
            self.help_callback,
            10
        )
        
        # 状态发布器
        self.status_publisher = self.create_publisher(String, '/helper_robot_status', 10)
        
        # 任务完成通知发布器 (10Hz)
        self.completion_publisher = self.create_publisher(Bool, '/task_completion_notification', 10)
        
        # 初始化MoveIt2
        print("\033[1;36mInitializing MoveIt2 for helper robot...\033[0m")
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=[
                'joint_1', 'joint_2', 'joint_3',
                'joint_4', 'joint_5', 'joint_6',
                'end_effector_link'
            ],
            base_link_name='base_link',
            end_effector_name='end_effector_link',
            group_name='arm',
            callback_group=self.callback_group,
        )
        
        # 设置运动规划参数
        self.moveit2.pipeline_id = 'pilz_industrial_motion_planner'
        self.moveit2.planner_id = 'LIN'
        self.moveit2.allowed_planning_time = 5.0
        self.moveit2.num_planning_attempts = 10
        self.moveit2.max_velocity = 0.1
        self.moveit2.max_acceleration = 0.1
        self.moveit2.cartesian_jump_threshold = 0.0
        
        # 执行器
        self.executor = rclpy.executors.MultiThreadedExecutor(2)
        self.executor.add_node(self)
        self.executor_thread = Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
        self.create_rate(1.0).sleep()
        
        # 初始化gripper
        # self.gripper = Gen3LiteGripper()
        
        # 默认位置 (初始位置/返回位置)
        self.default_position = {
            'x': 0.3,
            'y': 0.0,
            'z': DEFAULT_Z_HEIGHT
        }
        
        # 状态发布定时器
        self.status_timer = self.create_timer(0.1, self.publish_status)  # 10Hz
        
        # 任务完成通知定时器 (开始时不启动)
        self.completion_timer = None
        
        # 移动到默认位置
        self.move_to_default_position()
        
        # 创建日志文件
        self.log_file_path = os.path.join(os.getcwd(), 'help_executor_log.txt')
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(f"\n{'='*60}\n")
            log_file.write(f"Help Task Executor Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"{'='*60}\n")
        
        print(f"\033[1;32m{'='*60}\033[0m")
        print(f"\033[1;32m  Help Task Executor Ready\033[0m")
        print(f"\033[1;32m  Listening on: /require_help\033[0m")
        print(f"\033[1;32m  Status: {self.current_status}\033[0m")
        print(f"\033[1;32m  Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\033[0m")
        print(f"\033[1;32m{'='*60}\033[0m")
        
        # 帮助请求计数器
        self.help_request_count = 0

    def set_status(self, status):
        """线程安全的状态设置"""
        with self.status_lock:
            if self.current_status != status:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\033[1;35m[{timestamp}] Helper status: {self.current_status} -> {status}\033[0m")
                self.current_status = status

    def publish_status(self):
        """发布当前机器人状态 10Hz"""
        with self.status_lock:
            status_msg = String()
            status_msg.data = self.current_status
            self.status_publisher.publish(status_msg)

    def start_completion_notification(self):
        """开始10Hz的任务完成通知"""
        if self.completion_timer is not None:
            self.completion_timer.cancel()
        
        print(f"\033[1;32mStarting 10Hz task completion notification...\033[0m")
        self.completion_timer = self.create_timer(0.1, self.publish_completion_notification)  # 10Hz

    def stop_completion_notification(self):
        """停止任务完成通知"""
        if self.completion_timer is not None:
            self.completion_timer.cancel()
            self.completion_timer = None
            print(f"\033[1;32mStopped task completion notification\033[0m")

    def publish_completion_notification(self):
        """发布任务完成通知"""
        completion_msg = Bool()
        completion_msg.data = True
        self.completion_publisher.publish(completion_msg)

    def help_callback(self, msg):
        """处理接收到的 require_help 消息"""
        if self.task_in_progress:
            print(f"\033[1;33mTask already in progress, ignoring new help request\033[0m")
            return
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.help_request_count += 1
        
        try:
            # 解析JSON消息
            help_data = json.loads(msg.data)
            coordinates = help_data.get('coordinates', {})
            x_coord = coordinates.get('x', None)
            y_coord = coordinates.get('y', None)
            
            if x_coord is None or y_coord is None:
                print(f"\033[1;31mError: Invalid coordinates in help request\033[0m")
                return
            
            print(f"\n\033[1;31m{'*'*60}\033[0m")
            print(f"\033[1;31m  🚨 HELP REQUEST #{self.help_request_count} RECEIVED 🚨\033[0m")
            print(f"\033[1;31m{'*'*60}\033[0m")
            print(f"\033[1;36mTarget coordinates: X={x_coord}, Y={y_coord}\033[0m")
            print(f"\033[1;36mStarting pick and place task...\033[0m")
            
            # 记录到日志
            with open(self.log_file_path, 'a') as log_file:
                log_file.write(f"\n--- Help Request #{self.help_request_count} ---\n")
                log_file.write(f"Received at: {timestamp}\n")
                log_file.write(f"Target coordinates: X={x_coord}, Y={y_coord}\n")
                log_file.write(f"Starting task execution...\n")
            
            # 执行抓取任务
            self.execute_pick_and_place_task(x_coord, y_coord)
            
        except json.JSONDecodeError as e:
            print(f"\033[1;31mError: Failed to parse JSON: {e}\033[0m")
        except Exception as e:
            print(f"\033[1;31mError processing help request: {e}\033[0m")

    def move_to_default_position(self):
        """移动到默认位置"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\033[1;36m[{timestamp}] Moving to default position...\033[0m")
        
        # 创建默认位置的位姿
        default_pose = Pose()
        default_pose.position.x = float(self.default_position['x'])
        default_pose.position.y = float(self.default_position['y'])
        default_pose.position.z = float(self.default_position['z'])
        
        # 垂直向下的方向
        vertical_orientation = PyQuaternion(array=np.array([0.0, 1.0, 0.0, 0.0]))
        default_pose.orientation.x = float(vertical_orientation[0])
        default_pose.orientation.y = float(vertical_orientation[1])
        default_pose.orientation.z = float(vertical_orientation[2])
        default_pose.orientation.w = float(vertical_orientation[3])
        
        self.inverse_kinematic_movement(default_pose, cartesian=True)
        print(f"\033[1;32m[{timestamp}] Moved to default position\033[0m")

    def execute_pick_and_place_task(self, target_x, target_y):
        """执行抓取和放置任务"""
        self.task_in_progress = True
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            print(f"\n\033[1;36m[{timestamp}] ===== Starting Pick and Place Task =====\033[0m")
            
            # 停止任务完成通知 (如果正在发送)
            self.stop_completion_notification()
            
            # 1. 创建垂直向下的方向
            vertical_orientation = PyQuaternion(array=np.array([0.0, 1.0, 0.0, 0.0]))
            
            # 2. 移动到目标位置上方 (安全高度)
            print(f"\033[1;36m[{timestamp}] Moving above target position...\033[0m")
            approach_pose = Pose()
            approach_pose.position.x = float(target_x)
            approach_pose.position.y = float(target_y)
            approach_pose.position.z = float(SAFE_Z_HEIGHT)
            approach_pose.orientation.x = float(vertical_orientation[0])
            approach_pose.orientation.y = float(vertical_orientation[1])
            approach_pose.orientation.z = float(vertical_orientation[2])
            approach_pose.orientation.w = float(vertical_orientation[3])
            
            if not self.inverse_kinematic_movement(approach_pose, cartesian=True, status=RobotStatus.MOVING_TO_TARGET):
                raise Exception("Failed to move above target position")
            
            # 3. 下降到抓取高度
            print(f"\033[1;36m[{timestamp}] Descending to pick height...\033[0m")
            pick_pose = Pose()
            pick_pose.position.x = float(target_x)
            pick_pose.position.y = float(target_y)
            pick_pose.position.z = float(PICK_Z_HEIGHT)
            pick_pose.orientation.x = float(vertical_orientation[0])
            pick_pose.orientation.y = float(vertical_orientation[1])
            pick_pose.orientation.z = float(vertical_orientation[2])
            pick_pose.orientation.w = float(vertical_orientation[3])
            
            if not self.inverse_kinematic_movement(pick_pose, cartesian=True, status=RobotStatus.MOVING_TO_TARGET):
                raise Exception("Failed to move to pick position")
            
            # 4. 抓取物体
            print(f"\033[1;36m[{timestamp}] Grasping object...\033[0m")
            self.set_status(RobotStatus.GRASPING)
            if not self.grasp_object():
                raise Exception("Failed to grasp object")
            
            # 5. 上升到安全高度
            print(f"\033[1;36m[{timestamp}] Rising to safe height...\033[0m")
            if not self.inverse_kinematic_movement(approach_pose, cartesian=True):
                raise Exception("Failed to rise after grasping")
            
            # 6. 移动到放置位置
            print(f"\033[1;36m[{timestamp}] Moving to release position...\033[0m")
            put_pose = Pose()
            put_pose.position.x = float(PUT_Z_X)
            put_pose.position.y = float(PUT_Z_Y)
            put_pose.position.z = float(PUT_Z_HEIGHT)
            put_pose.orientation.x = float(vertical_orientation[0])
            put_pose.orientation.y = float(vertical_orientation[1])
            put_pose.orientation.z = float(vertical_orientation[2])
            put_pose.orientation.w = float(vertical_orientation[3])
            
            if not self.inverse_kinematic_movement(put_pose, cartesian=True, status=RobotStatus.MOVING_TO_RELEASE):
                raise Exception("Failed to move to release position")
            
            # 7. 释放物体
            print(f"\033[1;36m[{timestamp}] Releasing object...\033[0m")
            self.set_status(RobotStatus.RELEASING)
            if not self.release_object():
                raise Exception("Failed to release object")
            
            # 8. 返回默认位置
            print(f"\033[1;36m[{timestamp}] Returning to default position...\033[0m")
            self.set_status(RobotStatus.RETURNING_HOME)
            self.move_to_default_position()
            
            # 9. 任务完成，开始发送通知
            self.set_status(RobotStatus.TASK_COMPLETED)
            print(f"\033[1;32m[{timestamp}] Task completed successfully!\033[0m")
            print(f"\033[1;32m[{timestamp}] Starting 10Hz completion notification...\033[0m")
            
            # 开始10Hz的任务完成通知
            self.start_completion_notification()
            
            # 记录到日志
            with open(self.log_file_path, 'a') as log_file:
                log_file.write(f"Task completed successfully at {timestamp}\n")
                log_file.write(f"Starting 10Hz completion notification\n")
            
        except Exception as e:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\033[1;31m[{timestamp}] Task failed: {e}\033[0m")
            self.set_status(RobotStatus.TASK_FAILED)
            
            # 尝试返回默认位置
            try:
                print(f"\033[1;33m[{timestamp}] Attempting to return to default position...\033[0m")
                self.move_to_default_position()
            except:
                pass
            
            # 记录错误到日志
            with open(self.log_file_path, 'a') as log_file:
                log_file.write(f"Task failed at {timestamp}: {e}\n")
        
        finally:
            self.task_in_progress = False
            self.set_status(RobotStatus.WAITING_FOR_HELP)
            print(f"\033[1;36m[{timestamp}] Ready for next help request\033[0m")

    def inverse_kinematic_movement(self, target_pose, cartesian=False, status=None):
        """执行逆运动学移动"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if status:
            self.set_status(status)
        
        print(f"\033[1;36m[{timestamp}] Moving to: X={target_pose.position.x:.4f}, Y={target_pose.position.y:.4f}, Z={target_pose.position.z:.4f}\033[0m")
        
        try:
            self.moveit2.move_to_pose(
                pose=target_pose,
                cartesian=cartesian,
                cartesian_max_step=0.0025,
                cartesian_fraction_threshold=0.0,
            )
            
            rate = self.create_rate(100)
            while self.moveit2.query_state() != MoveIt2State.EXECUTING:
                rate.sleep()
                
            future = self.moveit2.get_execution_future()
            while not future.done():
                rate.sleep()
                
            print(f"\033[1;32m[{timestamp}] Movement completed\033[0m")
            return True
            
        except Exception as e:
            print(f"\033[1;31m[{timestamp}] Movement failed: {e}\033[0m")
            return False

    def grasp_object(self):
        """抓取物体"""
        try:
            # 这里应该控制gripper闭合
            # self.gripper.move_to_position(0.7)
            time.sleep(0.5)  # 模拟抓取时间
            print(f"\033[1;32mObject grasped successfully\033[0m")
            return True
        except Exception as e:
            print(f"\033[1;31mGrasping failed: {e}\033[0m")
            return False

    def release_object(self):
        """释放物体"""
        try:
            # 这里应该控制gripper打开
            # self.gripper.move_to_position(0.0)
            time.sleep(0.5)  # 模拟释放时间
            print(f"\033[1;32mObject released successfully\033[0m")
            return True
        except Exception as e:
            print(f"\033[1;31mReleasing failed: {e}\033[0m")
            return False

    def get_statistics(self):
        """获取统计信息"""
        return {
            "total_help_requests": self.help_request_count,
            "current_status": self.current_status,
            "log_file": self.log_file_path
        }


def main(args=None):
    # 设置信号处理器
    def signal_handler(sig, frame):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n\033[1;33m[{timestamp}] Shutting down help task executor...\033[0m")
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # 初始化ROS2
    rclpy.init(args=args)
    
    # 创建执行器节点
    executor = HelpTaskExecutor()
    
    try:
        print(f"\033[1;36mHelp Task Executor is running... Press Ctrl+C to stop\033[0m")
        
        # 运行节点
        rclpy.spin(executor)
        
    except KeyboardInterrupt:
        print(f"\n\033[1;33mReceived keyboard interrupt, shutting down...\033[0m")
    finally:
        # 清理资源
        stats = executor.get_statistics()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n\033[1;36m{'='*60}\033[0m")
        print(f"\033[1;36m  Session Summary\033[0m")
        print(f"\033[1;36m  Total Help Requests: {stats['total_help_requests']}\033[0m")
        print(f"\033[1;36m  Final Status: {stats['current_status']}\033[0m")
        print(f"\033[1;36m  Log File: {stats['log_file']}\033[0m")
        print(f"\033[1;36m  Session End: {timestamp}\033[0m")
        print(f"\033[1;36m{'='*60}\033[0m")
        
        # 停止任务完成通知
        executor.stop_completion_notification()
        
        # 记录会话结束到日志
        with open(executor.log_file_path, 'a') as log_file:
            log_file.write(f"\n{'='*60}\n")
            log_file.write(f"Session End: {timestamp}\n")
            log_file.write(f"Total Help Requests: {stats['total_help_requests']}\n")
            log_file.write(f"Final Status: {stats['current_status']}\n")
            log_file.write(f"{'='*60}\n")
        
        # 销毁节点
        executor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
