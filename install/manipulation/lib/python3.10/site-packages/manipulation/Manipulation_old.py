#!/usr/bin/env python3
import rclpy
import time
import os
import numpy as np
import signal
import sys
import datetime

from rclpy.node import Node
from geometry_msgs.msg import Pose
from pyquaternion import Quaternion as PyQuaternion
from threading import Thread
from rclpy.callback_groups import ReentrantCallbackGroup
from pymoveit2 import MoveIt2, MoveIt2State
from std_msgs.msg import Bool
from .gen3lite_pymoveit2 import Gen3LiteGripper


# Print script information and path
script_dir = os.path.dirname(os.path.realpath(__file__))
print(f"\033[1;36mScript execution path: {script_dir}\033[0m")

# Define Z position constants - updated based on log data
PICK_Z_HEIGHT = 0.115  # Z height for pick operations
PUT_Z_X = 0.16
PUT_Z_Y = 0.3
PUT_Z_HEIGHT = 0.214   # Z height for put operations
SAFE_Z_HEIGHT = 0.3    # Safe Z height for movement between operations

class Gen3LiteArm:
    def __init__(self):
        print("\033[1;36mInitializing Gen3LiteArm...\033[0m")
        self.node = Node("gen3_lite_arm")
        self.callback_group = ReentrantCallbackGroup()
        
        # Add emergency state tracking
        self.emergency_active = False
        self.current_pose = Pose()
        self.restart_ready = False  # Flag for restart readiness
        self.emergency_operation_in_progress = False  # Flag for emergency operation in progress
        
        # Subscribe to emergency stop topic
        print("\033[1;36mSubscribing to emergency stop topic: /emergency_stop\033[0m")
        self.emergency_subscription = self.node.create_subscription(
            Bool,
            '/emergency_stop',
            self.emergency_callback,
            10  # QoS profile
        )
        
        # Initialize MoveIt2
        print("\033[1;36mConfiguring MoveIt2...\033[0m")
        self.moveit2 = MoveIt2(
            node=self.node,
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
        self.executor = rclpy.executors.MultiThreadedExecutor(2)
        self.executor.add_node(self.node)
        self.executor_thread = Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
        self.node.create_rate(1.0).sleep()

        # Set motion planning parameters
        self.moveit2.pipeline_id = 'pilz_industrial_motion_planner'
        self.moveit2.planner_id = 'LIN'
        self.moveit2.allowed_planning_time = 5.0
        self.moveit2.num_planning_attempts = 10
        self.moveit2.max_velocity = 0.1
        self.moveit2.max_acceleration = 0.1
        self.moveit2.cartesian_jump_threshold = 0.0
        
        print("\033[1;32mGen3LiteArm initialization complete. Emergency stop handler activated.\033[0m")
        
        # Record start time
        self.start_time = datetime.datetime.now()
        print(f"\033[1;36mSystem start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\033[0m")

    def emergency_callback(self, msg):
        """Handle emergency stop messages"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prevent duplicate emergency handling
        if msg.data and not self.emergency_operation_in_progress:
            print(f"\033[1;31m[{timestamp}] !!! Emergency stop signal received !!!\033[0m")
            self.node.get_logger().error("Emergency stop signal received! Stopping current task and moving to Z position of 0.3")
            
            # Mark emergency operation in progress to prevent duplicate handling
            self.emergency_operation_in_progress = True
            
            # Set emergency flag to immediately stop current operation
            self.emergency_active = True
            
            # Stop current action by canceling execution
            try:
                future = self.moveit2.get_execution_future()
                if future and not future.done():
                    future.cancel()
                    print(f"\033[1;33m[{timestamp}] Successfully canceled current execution\033[0m")
                self.node.get_logger().info("Current execution stopped")
            except Exception as e:
                self.node.get_logger().error(f"Failed to stop execution: {e}")
                print(f"\033[1;31m[{timestamp}] Failed to stop execution: {e}\033[0m")
            
            # Brief wait to ensure action has stopped
            time.sleep(0.5)
            
            # Execute Z-axis movement
            print(f"\033[1;33m[{timestamp}] Executing emergency Z position movement to 0.3...\033[0m")
            self.emergency_raise_z()
            
            # Add 5-second delay - maintain emergency state for 5 seconds
            print(f"\033[1;33m[{timestamp}] Maintaining emergency state for 5 seconds...\033[0m")
            for i in range(5, 0, -1):
                print(f"\033[1;33m[{timestamp}] Emergency state remaining time: {i} seconds\033[0m")
                time.sleep(1)
            
            # Reset emergency state after 5 seconds
            self.emergency_active = False
            print(f"\033[1;32m[{timestamp}] Emergency state automatically reset after 5 seconds\033[0m")
            
            # Mark emergency operation as complete
            self.emergency_operation_in_progress = False
            print(f"\033[1;32m[{timestamp}] Emergency handling complete, system can continue operation\033[0m")
            
            # Mark system ready for restart
            self.restart_ready = True

    def emergency_raise_z(self):
        """In case of emergency, move the robot arm to Z position 0.3"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create emergency pose, Z position at 0.3
        emergency_pose = Pose()
        emergency_pose.position.x = float(self.current_pose.position.x)
        emergency_pose.position.y = float(self.current_pose.position.y)
        emergency_pose.position.z = float(0.3)
        
        # Keep the same orientation
        emergency_pose.orientation = self.current_pose.orientation
        
        print(f"\033[1;33m[{timestamp}] Emergency move to Z position 0.3: X={emergency_pose.position.x:.4f}, Y={emergency_pose.position.y:.4f}, Z={emergency_pose.position.z:.4f}\033[0m")
        
        # Temporarily increase speed for emergency move
        original_velocity = self.moveit2.max_velocity
        original_acceleration = self.moveit2.max_acceleration
        
        try:
            # Set faster motion parameters for emergency
            print(f"\033[1;36m[{timestamp}] Temporarily increasing speed parameters for emergency movement: velocity={0.3}, acceleration={0.3}\033[0m")
            self.moveit2.max_velocity = 0.3
            self.moveit2.max_acceleration = 0.3
            
            # Execute emergency movement
            print(f"\033[1;36m[{timestamp}] Starting emergency position movement...\033[0m")
            
            # Try multiple methods to execute emergency lift
            try:
                # Method 1: Direct position setting
                self.moveit2.move_to_pose(
                    pose=emergency_pose,
                    cartesian=True,
                    cartesian_max_step=0.01
                )
                
                print(f"\033[1;36m[{timestamp}] Emergency position command sent\033[0m")
                
                # Don't wait for execution to complete, allow system to continue running
                # No waiting or checking here to avoid blocking the program
                
                print(f"\033[1;32m[{timestamp}] Emergency position movement initiated\033[0m")
                
            except Exception as e:
                print(f"\033[1;31m[{timestamp}] Emergency position method 1 failed: {e}\033[0m")
                print(f"\033[1;36m[{timestamp}] Trying backup method...\033[0m")
                
                try:
                    # Method 2: Try simple joint space movement
                    current_joint_positions = self.moveit2.get_joint_positions()
                    if current_joint_positions:
                        print(f"\033[1;36m[{timestamp}] Attempting joint space movement...\033[0m")
                        # May need adjustment based on actual joint configuration
                        # Assuming joint 3 controls height
                        current_joint_positions[2] += 0.05  # Increase joint angle equivalent to 5cm
                        self.moveit2.move_to_joint_position(current_joint_positions)
                        print(f"\033[1;32m[{timestamp}] Joint space movement command sent\033[0m")
                except Exception as e:
                    print(f"\033[1;31m[{timestamp}] Emergency position method 2 also failed: {e}\033[0m")
                    print(f"\033[1;31m[{timestamp}] Unable to execute emergency position movement, please manually check arm status\033[0m")
                
        except Exception as e:
            print(f"\033[1;31m[{timestamp}] Emergency movement failed: {e}\033[0m")
        finally:
            # Restore original parameters
            print(f"\033[1;36m[{timestamp}] Restoring original speed parameters: velocity={original_velocity}, acceleration={original_acceleration}\033[0m")
            self.moveit2.max_velocity = original_velocity
            self.moveit2.max_acceleration = original_acceleration

    def inverse_kinematic_movement(self, target_pose, cartesian=False):
        """Execute inverse kinematic movement to target pose"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Store current pose for emergency handling
        self.current_pose = target_pose
        
        # Modified: No longer checking emergency state, as emergency state will be handled immediately in the emergency callback
        # and will automatically reset, allowing the system to continue running
            
        self.node.get_logger().info(
            f"Moving to position: {target_pose.position} {target_pose.orientation} with cartesian={cartesian}"
        )
        print(f"\033[1;36m[{timestamp}] Starting movement to: X={target_pose.position.x:.4f}, Y={target_pose.position.y:.4f}, Z={target_pose.position.z:.4f}, Cartesian mode={cartesian}\033[0m")
        
        try:
            move_start_time = datetime.datetime.now()
            self.moveit2.move_to_pose(
                pose=target_pose,
                cartesian=cartesian,
                cartesian_max_step=0.0025,
                cartesian_fraction_threshold=0.0,
            )
            
            rate = self.node.create_rate(100)
            print(f"\033[1;36m[{timestamp}] Waiting for planning to complete...\033[0m")
            planning_start = datetime.datetime.now()
            
            while self.moveit2.query_state() != MoveIt2State.EXECUTING:
                # Check if emergency operation is in progress during planning
                if self.emergency_operation_in_progress:
                    print(f"\033[1;31m[{timestamp}] Emergency operation in progress - waiting for completion\033[0m")
                    # Wait for emergency operation to complete
                    while self.emergency_operation_in_progress:
                        rate.sleep()
                rate.sleep()
                
            planning_time = (datetime.datetime.now() - planning_start).total_seconds()
            print(f"\033[1;36m[{timestamp}] Planning completed, duration: {planning_time:.2f} seconds\033[0m")
            print(f"\033[1;36m[{timestamp}] Starting execution of movement...\033[0m")
            
            execution_start = datetime.datetime.now()
            future = self.moveit2.get_execution_future()
            while not future.done():
                # Check if emergency operation is in progress during execution
                if self.emergency_operation_in_progress:
                    # Emergency will be handled in the callback, canceling current execution
                    # No additional action needed here, let the emergency callback handle it
                    print(f"\033[1;31m[{timestamp}] Emergency operation in progress - waiting for completion\033[0m")
                    # Wait for emergency operation to complete
                    while self.emergency_operation_in_progress:
                        rate.sleep()
                    # After emergency operation completes, this movement may have been canceled
                    if future.done():
                        break
                rate.sleep()
                
            execution_time = (datetime.datetime.now() - execution_start).total_seconds()
            total_time = (datetime.datetime.now() - move_start_time).total_seconds()
            print(f"\033[1;32m[{timestamp}] Movement complete! Execution time: {execution_time:.2f} seconds, Total time: {total_time:.2f} seconds\033[0m")
            
            return True
        except Exception as e:
            self.node.get_logger().error(f"Movement error: {e}")
            print(f"\033[1;31m[{timestamp}] Movement failed: {e}\033[0m")
            return False
    
    # Get emergency status
    def get_emergency_status(self):
        """Get detailed information about current emergency status"""
        status = {
            "emergency_active": self.emergency_active,
            "emergency_operation_in_progress": self.emergency_operation_in_progress,
            "restart_ready": self.restart_ready
        }
        print(f"\033[1;36mCurrent emergency status: Active={status['emergency_active']}, " +
              f"Operation in progress={status['emergency_operation_in_progress']}, " +
              f"Ready for restart={status['restart_ready']}\033[0m")
        return status
    
    # Check if restart is needed
    def is_restart_ready(self):
        if self.restart_ready:
            print(f"\033[1;36mSystem marked as ready for restart: {self.restart_ready}\033[0m")
        return self.restart_ready
            
    def shutdown(self):
        """Shutdown executor"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\033[1;36m[{timestamp}] Shutting down Gen3LiteArm executor...\033[0m")
        self.executor.shutdown()
        
        # Calculate runtime
        run_time = datetime.datetime.now() - self.start_time
        print(f"\033[1;36mSystem runtime: {run_time.total_seconds():.2f} seconds\033[0m")


def execute_pick_and_place(arm, gripper, task_coordinates):
    """Execute a sequence of pick and place tasks based on the given coordinates"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\033[1;36m[{timestamp}] Starting pick and place sequence with {len(task_coordinates)} tasks\033[0m")
    
    # Create a vertical orientation quaternion
    vertical_orientation = PyQuaternion(array=np.array([0.0, 1.0, 0.0, 0.0]))
    
    for i, (x, y) in enumerate(task_coordinates):
        task_num = i + 1
        print(f"\033[1;36m[{timestamp}] ===== Executing Task {task_num}/{len(task_coordinates)} =====\033[0m")
        
        # Convert to float
        x_float = float(x)
        y_float = float(y)
        
        # Create poses for the task
        approach_pose = Pose()
        approach_pose.position.x = x_float
        approach_pose.position.y = y_float
        approach_pose.position.z = float(SAFE_Z_HEIGHT)
        approach_pose.orientation.x = float(vertical_orientation[0])
        approach_pose.orientation.y = float(vertical_orientation[1])
        approach_pose.orientation.z = float(vertical_orientation[2])
        approach_pose.orientation.w = float(vertical_orientation[3])
        
        pick_pose = Pose()
        pick_pose.position.x = x_float
        pick_pose.position.y = y_float
        pick_pose.position.z = float(PICK_Z_HEIGHT)
        pick_pose.orientation.x = float(vertical_orientation[0])
        pick_pose.orientation.y = float(vertical_orientation[1])
        pick_pose.orientation.z = float(vertical_orientation[2])
        pick_pose.orientation.w = float(vertical_orientation[3])
        
        put_pose = Pose()
        put_pose.position.x = float(PUT_Z_X)
        put_pose.position.y = float(PUT_Z_Y)
        put_pose.position.z = float(PUT_Z_HEIGHT)
        put_pose.orientation.x = float(vertical_orientation[0])
        put_pose.orientation.y = float(vertical_orientation[1])
        put_pose.orientation.z = float(vertical_orientation[2])
        put_pose.orientation.w = float(vertical_orientation[3])
        
        # Print task information
        print(f"\033[1;36m[{timestamp}] Moving to pick position for Task {task_num}: X={x_float:.4f}, Y={y_float:.4f}, Z={PICK_Z_HEIGHT:.4f}\033[0m")
        
        # Move to position above pick position first
        if not arm.inverse_kinematic_movement(approach_pose, cartesian=True):
            print(f"\033[1;31m[{timestamp}] Failed to move to approach position for Task {task_num}\033[0m")
            return False
        
        # Move to pick position
        if not arm.inverse_kinematic_movement(pick_pose, cartesian=True):
            print(f"\033[1;31m[{timestamp}] Failed to move to pick position for Task {task_num}\033[0m")
            return False
        
        # Close gripper to grasp object
        print(f"\033[1;36m[{timestamp}] Closing gripper to grasp object for Task {task_num}\033[0m")
        gripper_start = datetime.datetime.now()
        gripper.move_to_position(0.7)
        print(f"\033[1;32m[{timestamp}] Gripper close command sent, duration: {(datetime.datetime.now() - gripper_start).total_seconds():.2f} seconds\033[0m")
        time.sleep(0.5)
        
        # Move back to safe height
        if not arm.inverse_kinematic_movement(approach_pose, cartesian=True):
            print(f"\033[1;31m[{timestamp}] Failed to move back to safe height after pick for Task {task_num}\033[0m")
            return False
        
        # Print put information
        print(f"\033[1;36m[{timestamp}] Moving to put position for Task {task_num}: X={x_float:.4f}, Y={y_float:.4f}, Z={PUT_Z_HEIGHT:.4f}\033[0m")
        
        # Move to put position
        if not arm.inverse_kinematic_movement(put_pose, cartesian=True):
            print(f"\033[1;31m[{timestamp}] Failed to move to put position for Task {task_num}\033[0m")
            return False
        
        # Open gripper to release object
        print(f"\033[1;36m[{timestamp}] Opening gripper to release object for Task {task_num}\033[0m")
        gripper_start = datetime.datetime.now()
        gripper.move_to_position(0.0)
        print(f"\033[1;32m[{timestamp}] Gripper open command sent, duration: {(datetime.datetime.now() - gripper_start).total_seconds():.2f} seconds\033[0m")
        time.sleep(0.5)
        
        # Move back to safe height
        if not arm.inverse_kinematic_movement(approach_pose, cartesian=True):
            print(f"\033[1;31m[{timestamp}] Failed to move back to safe height after put for Task {task_num}\033[0m")
            return False
        
        print(f"\033[1;32m[{timestamp}] Task {task_num}/{len(task_coordinates)} completed successfully\033[0m")
    
    print(f"\033[1;32m[{timestamp}] All pick and place tasks completed successfully\033[0m")
    return True


def get_task_input():
    """Get task input from user"""
    print("\033[1;36mEnter task information for pick and place operations\033[0m")
    
    # Get number of tasks
    while True:
        try:
            task_number = int(input("\033[1;36mEnter the number of pick tasks to execute: \033[0m"))
            if task_number <= 0:
                print("\033[1;31mNumber of tasks must be greater than 0. Please try again.\033[0m")
                continue
            break
        except ValueError:
            print("\033[1;31mInvalid input. Please enter a valid number.\033[0m")
    
    # Get coordinates for each task
    task_coordinates = []
    print(f"\033[1;36mEnter coordinates for {task_number} tasks:\033[0m")
    
    for i in range(task_number):
        while True:
            try:
                print(f"\033[1;36m--- Task {i+1} ---\033[0m")
                x = float(input(f"\033[1;36mEnter X coordinate for task {i+1}: \033[0m"))
                y = float(input(f"\033[1;36mEnter Y coordinate for task {i+1}: \033[0m"))
                task_coordinates.append((x, y))
                break
            except ValueError:
                print("\033[1;31mInvalid input. Please enter valid numbers for coordinates.\033[0m")
    
    # Display the entered tasks
    print("\033[1;36m\nTask Summary:\033[0m")
    print("\033[1;36m------------------------------------\033[0m")
    print("\033[1;36m| Task |   X    |   Y    |   Z1   |   Z2   |\033[0m")
    print("\033[1;36m------------------------------------\033[0m")
    
    for i, (x, y) in enumerate(task_coordinates):
        print(f"\033[1;36m|  {i+1:2d}  | {x:.4f} | {y:.4f} | {PICK_Z_HEIGHT:.4f} | {PUT_Z_HEIGHT:.4f} |\033[0m")
    
    print("\033[1;36m------------------------------------\033[0m")
    print(f"\033[1;36mZ1 = {PICK_Z_HEIGHT} (Pick height)\033[0m")
    print(f"\033[1;36mZ2 = {PUT_Z_HEIGHT} (Put height)\033[0m")
    
    return task_coordinates


def main(args=None):
    """Main function"""
    # Set up signal handler for Ctrl+C
    def signal_handler(sig, frame):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\033[1;33m[{timestamp}] Received shutdown signal, cleaning up...\033[0m")
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Log file for execution records
    log_file_path = os.path.join(script_dir, "robot_execution_log.txt")
    print(f"\033[1;36mProgram log will be saved to: {log_file_path}\033[0m")
    
    with open(log_file_path, "a") as log_file:
        log_file.write(f"\n\n{'='*50}\n")
        log_file.write(f"Execution start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"{'='*50}\n")
    
    # Get task input from user
    task_coordinates = get_task_input()
    
    # Main loop for re-execution
    retry_count = 0
    max_retries = 5
    
    while retry_count < max_retries:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\033[1;36m[{timestamp}] Starting task execution cycle {retry_count+1}/{max_retries}\033[0m")
        
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Task execution cycle {retry_count+1} started at {timestamp}\n")
        
        rclpy.init(args=args)
        print(f"\033[1;32m[{timestamp}] Task started\033[0m")
        
        try:
            print(f"\033[1;36m[{timestamp}] Initializing arm and gripper...\033[0m")
            arm = Gen3LiteArm()
            gripper = Gen3LiteGripper()
            
            # Execute pick and place sequence
            print(f"\033[1;36m[{timestamp}] Starting pick and place sequence...\033[0m")
            with open(log_file_path, "a") as log_file:
                log_file.write(f"Starting pick and place sequence with {len(task_coordinates)} tasks at {timestamp}\n")
                
            success = execute_pick_and_place(arm, gripper, task_coordinates)
                        
            with open(log_file_path, "a") as log_file:
                log_file.write(f"Pick and place sequence {'successful' if success else 'failed'} at {timestamp}\n")

            # Check if restart needed
            restart_needed = arm.is_restart_ready()
            emergency_status = arm.get_emergency_status()
            
            # Record status to log
            with open(log_file_path, "a") as log_file:
                log_file.write(f"Status check: Emergency active={emergency_status['emergency_active']}, " + 
                              f"Emergency operation in progress={emergency_status['emergency_operation_in_progress']}, " +
                              f"Ready for restart={restart_needed}\n")
            
            # Only consider normal completion if task succeeded and no restart needed
            if success and not restart_needed:
                print(f"\033[1;32m[{timestamp}] Task completed successfully\033[0m")
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"Task completed successfully {timestamp}\n")
                break  # Normal completion, exit loop
            elif restart_needed:
                print(f"\033[1;33m[{timestamp}] Preparing to restart program...\033[0m")
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"Preparing to restart program {timestamp}\n")
                # Continue loop, will re-execute program
            else:
                print(f"\033[1;33m[{timestamp}] Task failed but not in emergency state\033[0m")
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"Task failed but not in emergency state {timestamp}\n")
                
            retry_count += 1
                
        except Exception as e:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\033[1;31m[{timestamp}] Main function error: {e}\033[0m")
            with open(log_file_path, "a") as log_file:
                log_file.write(f"Main function error: {e} {timestamp}\n")
            retry_count += 1
        finally:
            # Always clean up properly
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\033[1;36m[{timestamp}] Cleaning up resources...\033[0m")
            
            try:
                gripper.shutdown()
                print(f"\033[1;36m[{timestamp}] Gripper shut down\033[0m")
            except Exception as e:
                print(f"\033[1;31m[{timestamp}] Gripper shutdown error: {e}\033[0m")
                
            try:
                arm.shutdown()
                print(f"\033[1;36m[{timestamp}] Arm shut down\033[0m")
            except Exception as e:
                print(f"\033[1;31m[{timestamp}] Arm shutdown error: {e}\033[0m")
                
            try:
                rclpy.shutdown()
                print(f"\033[1;36m[{timestamp}] ROS2 shut down\033[0m")
            except Exception as e:
                print(f"\033[1;31m[{timestamp}] ROS2 shutdown error: {e}\033[0m")
            
            with open(log_file_path, "a") as log_file:
                log_file.write(f"Resource cleanup complete {timestamp}\n")
            
            # If restart needed, wait a short time before continuing loop
            if restart_needed:
                wait_time = 2
                print(f"\033[1;33m[{timestamp}] Waiting {wait_time} seconds before restarting...\033[0m")
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"Waiting {wait_time} seconds before restarting {timestamp}\n")
                time.sleep(wait_time)  # Wait for cleanup to complete
            else:
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"Task cycle ended, no restart needed {timestamp}\n")
                break  # If no restart needed, exit loop
    
    if retry_count >= max_retries:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\033[1;31m[{timestamp}] Maximum retry count reached ({max_retries}), program exiting\033[0m")
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Maximum retry count reached ({max_retries}), program exiting {timestamp}\n")
    
    with open(log_file_path, "a") as log_file:
        log_file.write(f"\n{'='*50}\n")
        log_file.write(f"Execution end time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"{'='*50}\n")
    
    print(f"\033[1;36mExecution log saved to: {log_file_path}\033[0m")


if __name__ == '__main__':
    print(f"\033[1;32m{'='*50}\033[0m")
    print(f"\033[1;32m  Gen3Lite Robotic Arm Control (with Emergency Stop)\033[0m")
    print(f"\033[1;32m  Execution time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\033[0m")
    print(f"\033[1;32m{'='*50}\033[0m")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\033[1;33mProgram interrupted by user\033[0m")
    except Exception as e:
        print(f"\033[1;31mProgram exception: {e}\033[0m")
    finally:
        print("\033[1;36mProgram exited\033[0m")
