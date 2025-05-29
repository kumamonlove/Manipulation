#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
import threading
import time
import random

class SimulatedGripperPublisher(Node):
    """
    Simulates a gripper position publisher for testing recovery behavior.
    
    This node:
    1. Publishes simulated gripper position data to /gripper_position
    2. Listens to /robot_status to detect when gripper operations are happening
    3. Simulates both successful and failed gripper operations
    4. Provides configurable failure rates for testing
    """
    
    def __init__(self):
        super().__init__('simulated_gripper_publisher')
        
        # Parameters
        self.declare_parameter('grip_failure_rate', 0.7)  # 70% failure rate for gripping
        self.declare_parameter('release_failure_rate', 0.3)  # 30% failure rate for releasing
        self.declare_parameter('publish_rate', 10.0)  # Hz
        self.declare_parameter('open_position', 0.0)  # Fully open position
        self.declare_parameter('closed_position', 0.7)  # Fully closed position
        
        # Get parameters
        self.grip_failure_rate = self.get_parameter('grip_failure_rate').value
        self.release_failure_rate = self.get_parameter('release_failure_rate').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.open_position = self.get_parameter('open_position').value
        self.closed_position = self.get_parameter('closed_position').value
        
        # Current gripper state
        self.current_position = self.open_position  # Start in open position
        self.target_position = self.open_position
        self.current_status = "waiting_for_task"
        self.previous_status = "waiting_for_task"
        self.movement_in_progress = False
        self.operation_success = True  # Default to success
        
        # Create publisher for gripper position
        self.position_publisher = self.create_publisher(
            Float32, 
            '/gripper_position', 
            10
        )
        
        # Create subscriber for robot status
        self.status_subscriber = self.create_subscription(
            String,
            '/robot_status',
            self.status_callback,
            10
        )
        
        # Timer for publishing gripper position
        self.timer = self.create_timer(1.0/self.publish_rate, self.publish_position)
        
        # Animation thread for smooth position changes
        self.animation_thread = threading.Thread(target=self.animate_position, daemon=True)
        self.animation_thread.start()
        
        self.get_logger().info('Simulated Gripper Publisher started')
        self.get_logger().info(f'Grip failure rate: {self.grip_failure_rate * 100:.0f}%')
        self.get_logger().info(f'Release failure rate: {self.release_failure_rate * 100:.0f}%')
        
        # Print colored terminal message
        print(f"\033[1;32m{'='*60}\033[0m")
        print(f"\033[1;32m  Simulated Gripper Position Publisher Started\033[0m")
        print(f"\033[1;32m  Publishing to: /gripper_position at {self.publish_rate}Hz\033[0m")
        print(f"\033[1;32m  Grip failure rate: {self.grip_failure_rate * 100:.0f}%\033[0m")
        print(f"\033[1;32m  Release failure rate: {self.release_failure_rate * 100:.0f}%\033[0m")
        print(f"\033[1;32m{'='*60}\033[0m")

    def status_callback(self, msg):
        """Handle robot status updates"""
        self.previous_status = self.current_status
        self.current_status = msg.data
        
        # Detect gripper operations based on status changes
        if self.previous_status != self.current_status:
            if self.current_status == "grasping":
                self.handle_grip_command()
            elif self.current_status == "releasing":
                self.handle_release_command()
    
    def handle_grip_command(self):
        """Simulate gripper closing with potential failures"""
        self.get_logger().info('Gripper close command detected')
        print(f"\033[1;36mSimulating gripper close operation\033[0m")
        
        # Determine if this grip operation will succeed or fail
        self.operation_success = random.random() > self.grip_failure_rate
        
        if self.operation_success:
            # Successful grip - move to closed position
            self.target_position = self.closed_position
            print(f"\033[1;32mSimulating successful grip\033[0m")
        else:
            # Failed grip - move slightly but stay closer to open
            self.target_position = self.open_position + 0.05
            print(f"\033[1;31mSimulating failed grip - gripper will remain nearly open\033[0m")
        
        self.movement_in_progress = True
    
    def handle_release_command(self):
        """Simulate gripper opening with potential failures"""
        self.get_logger().info('Gripper open command detected')
        print(f"\033[1;36mSimulating gripper open operation\033[0m")
        
        # Determine if this release operation will succeed or fail
        self.operation_success = random.random() > self.release_failure_rate
        
        if self.operation_success:
            # Successful release - move to open position
            self.target_position = self.open_position
            print(f"\033[1;32mSimulating successful release\033[0m")
        else:
            # Failed release - move slightly but stay closer to closed
            self.target_position = self.closed_position - 0.05
            print(f"\033[1;31mSimulating failed release - gripper will remain nearly closed\033[0m")
        
        self.movement_in_progress = True
    
    def animate_position(self):
        """Animate gripper position changes for smooth transitions"""
        rate = 50  # Hz for animation updates
        step_size = 0.02  # Position change per step
        
        while rclpy.ok():
            if self.movement_in_progress:
                # Calculate direction and step
                diff = self.target_position - self.current_position
                
                if abs(diff) < step_size:
                    # Close enough, snap to target
                    self.current_position = self.target_position
                    self.movement_in_progress = False
                    
                    if self.operation_success:
                        print(f"\033[1;32mGripper operation completed successfully: position={self.current_position:.2f}\033[0m")
                    else:
                        print(f"\033[1;31mGripper operation failed: position={self.current_position:.2f}\033[0m")
                else:
                    # Move one step toward target
                    step = step_size if diff > 0 else -step_size
                    self.current_position += step
            
            time.sleep(1.0/rate)
    
    def publish_position(self):
        """Publish the current gripper position"""
        msg = Float32()
        msg.data = float(self.current_position)
        self.position_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimulatedGripperPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\033[1;33mShutting down simulated gripper publisher\033[0m")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
