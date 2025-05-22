#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32
from collections import deque
import time
import subprocess
import threading


class F2GripperMonitor(Node):
    def __init__(self):
        super().__init__('f2_gripper_monitor')
        
        # Buffer to store last 30 measurements of F2 (joint_2 effort)
        self.f2_buffer = deque(maxlen=30)
        self.last_print_time = time.time()
        self.emergency_triggered = False
        
        # 添加夹爪位置监控变量
        self.gripper_position = None
        self.gripper_position_buffer = deque(maxlen=30)
        self.publish_gripper_position = True
        
        # 添加锁来防止并发触发紧急停止
        self.emergency_lock = threading.Lock()
        
        # 添加紧急恢复时间计时器
        self.emergency_reset_time = 60  # 60秒后重置紧急状态
        self.emergency_time = None
        
        # Create publisher for emergency stop
        self.emergency_publisher = self.create_publisher(
            Bool, 
            '/emergency_stop', 
            10
        )
        
        # Create publisher for gripper position (for debugging and Manipulation.py use)
        self.gripper_position_publisher = self.create_publisher(
            Float32,
            '/gripper_position',
            10
        )
        
        # Subscribe to joint states at 100Hz
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            100  # Higher QoS to ensure we get all messages
        )
        
        # 添加定时器定期检查紧急状态
        self.timer = self.create_timer(1.0, self.check_emergency_state)
        
        # 添加定时器定期发布夹爪位置 (10Hz)
        self.gripper_timer = self.create_timer(0.1, self.publish_gripper_data)
        
        # Print a clear message that we're starting
        print("\n==== F2 AND GRIPPER MONITOR STARTED ====")
        print("Collecting joint_2 effort values at 100Hz")
        print("Monitoring gripper position (right_finger_bottom_joint)")
        print("Publishing gripper position to /gripper_position topic at 10Hz")
        print("Will print last 30 values every second")
        print("Will trigger emergency stop if difference between current F2 and any value in window exceeds 10\n")
        print("Emergency state will auto-reset after 60 seconds")

    def joint_state_callback(self, msg):
        # Extract F2 (joint_2 effort) from the message
        if 'joint_2' in msg.name:
            idx = msg.name.index('joint_2')
            if idx < len(msg.effort):
                current_f2 = msg.effort[idx]
                
                # 只在未触发紧急状态时检查变化
                if not self.emergency_triggered and len(self.f2_buffer) > 0:
                    self.check_for_significant_changes(current_f2)
                
                # Add F2 value to buffer (即使在紧急状态下也继续收集数据)
                self.f2_buffer.append(current_f2)
                
                # Print every second
                current_time = time.time()
                if current_time - self.last_print_time >= 1.0:
                    self.last_print_time = current_time
                    self.print_values()
        
        # Extract gripper position (right_finger_bottom_joint)
        if 'right_finger_bottom_joint' in msg.name:
            idx = msg.name.index('right_finger_bottom_joint')
            if idx < len(msg.position):
                self.gripper_position = msg.position[idx]
                self.gripper_position_buffer.append(self.gripper_position)
                # 这里不需要打印，已经统一在print_values中打印

    def publish_gripper_data(self):
        """Publish current gripper position at 10Hz"""
        if self.gripper_position is not None and self.publish_gripper_position:
            msg = Float32()
            msg.data = float(self.gripper_position)
            self.gripper_position_publisher.publish(msg)

    def check_for_significant_changes(self, current_f2):
        # 使用锁防止多线程同时触发紧急情况
        with self.emergency_lock:
            # 再次检查紧急状态（双重检查模式）
            if self.emergency_triggered:
                return
                
            # Check if current value differs from any value in the window by more than 10
            for i, past_f2 in enumerate(self.f2_buffer):
                difference = abs(current_f2 - past_f2)
                if difference > 10:
                    self.trigger_emergency(current_f2, past_f2, difference)
                    return  # Only trigger once

    def trigger_emergency(self, current_f2, past_f2, difference):
        # 再次检查状态，预防竞态条件
        if self.emergency_triggered:
            return
            
        # 设置紧急状态
        self.emergency_triggered = True
        self.emergency_time = time.time()
        
        # Print alert in red
        print(f"\033[1;31m!!! EMERGENCY !!! Large F2 change detected: {past_f2:.3f} → {current_f2:.3f} (Diff: {difference:.3f})\033[0m")
        print("\033[1;31mTriggering emergency stop and sending command to raise z by 5cm\033[0m")
        
        # Publish emergency stop message ONLY ONCE
        msg = Bool()
        msg.data = True
        self.emergency_publisher.publish(msg)
        
        # 在发送紧急信号后等待一小段时间，确保消息被处理
        time.sleep(0.1)
        
        # Try to kill the Manipulation process
        try:
            subprocess.run(["pkill", "-f", "manipulation Manipulation"], 
                          stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            print("\033[1;31mSent termination signal to Manipulation program\033[0m")
        except Exception as e:
            print(f"\033[1;31mFailed to terminate Manipulation program: {e}\033[0m")
    
    def check_emergency_state(self):
        """定期检查紧急状态，在指定时间后自动重置"""
        if self.emergency_triggered and self.emergency_time is not None:
            elapsed = time.time() - self.emergency_time
            
            # 如果已经过了重置时间，重置紧急状态
            if elapsed >= self.emergency_reset_time:
                print(f"\033[1;32m紧急状态已自动重置（{self.emergency_reset_time}秒后）\033[0m")
                self.emergency_triggered = False
                self.emergency_time = None

    def print_values(self):
        # Clear formatting to make output stand out
        print("\033[1;32m")  # Bold green text
        
        print("\n====== JOINT MONITORING - LAST VALUES ======")
        print(f"Time: {time.strftime('%H:%M:%S')}")
        print(f"F2 Buffer size: {len(self.f2_buffer)}")
        print(f"Gripper Position: {self.gripper_position:.6f if self.gripper_position is not None else 'Not available'}")
        print(f"Emergency Status: {'ACTIVE' if self.emergency_triggered else 'INACTIVE'}")
        
        if self.emergency_triggered and self.emergency_time is not None:
            elapsed = time.time() - self.emergency_time
            print(f"Emergency active for: {elapsed:.1f}s (resets after {self.emergency_reset_time}s)")
        
        if len(self.f2_buffer) > 0:
            print("\nF2 Values:")
            values = list(self.f2_buffer)
            for i, val in enumerate(values[-5:]):  # Show only the last 5 values to save space
                print(f"[{i}]: {val:.6f}")
        else:
            print("No F2 measurements received yet")
            
        if len(self.gripper_position_buffer) > 0:
            print("\nGripper Position Values:")
            values = list(self.gripper_position_buffer)
            for i, val in enumerate(values[-5:]):  # Show only the last 5 values to save space
                print(f"[{i}]: {val:.6f}")
        else:
            print("No gripper position measurements received yet")
        
        print("==========================================\033[0m\n")  # Reset text formatting


def main(args=None):
    rclpy.init(args=args)
    monitor = F2GripperMonitor()
    
    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        print("Monitor stopped by user")
    finally:
        # Cleanup
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
