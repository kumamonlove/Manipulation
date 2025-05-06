#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from collections import deque
import time
import subprocess


class F2Monitor(Node):
    def __init__(self):
        super().__init__('f2_monitor')
        
        # Buffer to store last 30 measurements of F2 (joint_2 effort)
        self.f2_buffer = deque(maxlen=30)
        self.last_print_time = time.time()
        self.emergency_triggered = False
        
        # Create publisher for emergency stop
        self.emergency_publisher = self.create_publisher(
            Bool, 
            '/emergency_stop', 
            10
        )
        
        # Subscribe to joint states at 100Hz
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            100  # Higher QoS to ensure we get all messages
        )
        
        # Print a clear message that we're starting
        print("\n==== F2 MONITOR STARTED ====")
        print("Collecting joint_2 effort values at 100Hz")
        print("Will print last 30 values every second")
        print("Will trigger emergency stop if difference between current F2 and any value in window exceeds 10\n")

    def joint_state_callback(self, msg):
        # Extract F2 (joint_2 effort) from the message
        if 'joint_2' in msg.name:
            idx = msg.name.index('joint_2')
            if idx < len(msg.effort):
                current_f2 = msg.effort[idx]
                
                # Check for significant changes if we have previous values
                if len(self.f2_buffer) > 0 and not self.emergency_triggered:
                    self.check_for_significant_changes(current_f2)
                
                # Add F2 value to buffer
                self.f2_buffer.append(current_f2)
                
                # Print every second
                current_time = time.time()
                if current_time - self.last_print_time >= 1.0:
                    self.last_print_time = current_time
                    self.print_f2_values()

    def check_for_significant_changes(self, current_f2):
        # Check if current value differs from any value in the window by more than 10
        for i, past_f2 in enumerate(self.f2_buffer):
            difference = abs(current_f2 - past_f2)
            if difference > 10:
                self.trigger_emergency(current_f2, past_f2, difference)
                return  # Only trigger once

    def trigger_emergency(self, current_f2, past_f2, difference):
        # Print alert in red
        print(f"\033[1;31m!!! EMERGENCY !!! Large F2 change detected: {past_f2:.3f} → {current_f2:.3f} (Diff: {difference:.3f})\033[0m")
        print("\033[1;31mTriggering emergency stop and sending command to raise z by 5cm\033[0m")
        
        # Set flag to prevent multiple triggers
        self.emergency_triggered = True
        
        # Publish emergency stop message
        msg = Bool()
        msg.data = True
        self.emergency_publisher.publish(msg)
        
        # Try to kill the Manipulation process
        try:
            subprocess.run(["pkill", "-f", "manipulation Manipulation"], 
                          stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            print("\033[1;31mSent termination signal to Manipulation program\033[0m")
        except Exception as e:
            print(f"\033[1;31mFailed to terminate Manipulation program: {e}\033[0m")

    def print_f2_values(self):
        # Clear formatting to make output stand out
        print("\033[1;32m")  # Bold green text
        
        print("\n====== F2 MONITORING - LAST 30 VALUES ======")
        print(f"Time: {time.strftime('%H:%M:%S')}")
        print(f"Buffer size: {len(self.f2_buffer)}")
        
        if len(self.f2_buffer) > 0:
            print("\nValues:")
            values = list(self.f2_buffer)
            for i, val in enumerate(values):
                print(f"[{i}]: {val:.6f}")
        else:
            print("No F2 measurements received yet")
        
        print("==========================================\033[0m\n")  # Reset text formatting


def main(args=None):
    rclpy.init(args=args)
    f2_monitor = F2Monitor()
    
    try:
        rclpy.spin(f2_monitor)
    except KeyboardInterrupt:
        print("F2 Monitor stopped by user")
    finally:
        # Cleanup
        f2_monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
