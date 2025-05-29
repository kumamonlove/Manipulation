#!/usr/bin/env python3
import rclpy
import time
import os
import numpy as np
import signal
import sys
import datetime
import json

from rclpy.node import Node
from geometry_msgs.msg import Pose
from pyquaternion import Quaternion as PyQuaternion
from threading import Thread, Lock, Event
from rclpy.callback_groups import ReentrantCallbackGroup
from pymoveit2 import MoveIt2, MoveIt2State
from std_msgs.msg import Bool, String, Float32
from .gen3lite_pymoveit2 import Gen3LiteGripper
from collections import deque

# Print script information and path
script_dir = os.path.dirname(os.path.realpath(__file__))
print(f"\033[1;36mScript execution path: {script_dir}\033[0m")

# =================== CONSTANTS ===================
# Z positions
PICK_Z_HEIGHT = 0.13  # Z height for pick operations
PUT_Z_X = 0.141
PUT_Z_Y = 0.45
PUT_Z_HEIGHT = 0.129   # Z height for put operations
SAFE_Z_HEIGHT = 0.3    # Safe Z height for movement between operations

# Initial/Home position
HOME_X = 0.1
HOME_Y = 0.1  
HOME_Z = 0.3

# Gripper parameters
GRIPPER_DETECTION_THRESHOLD = 0.1
GRIPPER_CLOSE_TARGET = 0.7
GRIPPER_OPEN_TARGET = 0.0
OBJECT_DROPPED_THRESHOLD = 0.69  # >0.69 means dropped

# Help‑line threshold
Y_THRESHOLD_FOR_HELP = 0.35
# =================================================

class RobotStatus:
    WAITING_FOR_TASK = "waiting_for_task"
    MOVING_TO_TARGET = "moving_to_target"
    GRASPING = "grasping"
    MOVING_TO_RELEASE = "moving_to_release"
    RELEASING = "releasing"
    MOVE_TO_TARGET_FAILED = "move_to_target_failed"
    MOVE_TO_RELEASE_FAILED = "move_to_release_failed"
    GRASPING_FAILED = "grasping_failed"
    RELEASING_FAILED = "releasing_failed"
    OBJECT_DROPPED = "object_dropped"
    EMERGENCY_STOP = "emergency_stop"
    TASK_COMPLETED = "task_completed"
    REQUIRE_HELP = "require_help"
    RETURNING_HOME = "returning_home"

class Gen3LiteArm:
    def __init__(self):
        print("\033[1;36mInitializing Gen3LiteArm...\033[0m")
        self.node = Node("gen3_lite_arm")
        self.callback_group = ReentrantCallbackGroup()

        # -------- Runtime flags --------
        self.emergency_active = False
        self.current_pose = Pose()
        self.restart_ready = False
        self.emergency_operation_in_progress = False
        
        # -------- Object drop monitoring --------
        self.object_drop_detected = Event()
        self.object_drop_monitoring_active = False
        self.gripper_closed = False  # Track if gripper should be holding an object

        # -------- Status pub --------
        self.current_status = RobotStatus.WAITING_FOR_TASK
        self.status_lock = Lock()
        self.status_publisher = self.node.create_publisher(String, "/robot_status", 10)

        # -------- Gripper position sub & stats --------
        self.gripper_position = None
        self.gripper_recent = deque(maxlen=5)       # store last 5 readings
        self.gripper_subscription = self.node.create_subscription(
            Float32,
            "/gripper_position",
            self.gripper_callback,
            10,
            callback_group=self.callback_group,
        )

        # stats printer – once per second
        self.gripper_stats_timer = self.node.create_timer(1.0, self.print_gripper_stats)

        # -------- Help publisher --------
        self.require_help_publisher = self.node.create_publisher(String, "/require_help", 10)

        # -------- Emergency sub --------
        self.emergency_subscription = self.node.create_subscription(
            Bool,
            "/emergency_stop",
            self.emergency_callback,
            10,
        )

        # -------- MoveIt2 --------
        self.moveit2 = MoveIt2(
            node=self.node,
            joint_names=[
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "end_effector_link",
            ],
            base_link_name="base_link",
            end_effector_name="end_effector_link",
            group_name="arm",
            callback_group=self.callback_group,
        )
        self.executor = rclpy.executors.MultiThreadedExecutor(2)
        self.executor.add_node(self.node)
        Thread(target=self.executor.spin, daemon=True).start()
        self.node.create_rate(1.0).sleep()

        # planner tuning
        self.moveit2.pipeline_id = "pilz_industrial_motion_planner"
        self.moveit2.planner_id = "LIN"
        self.moveit2.allowed_planning_time = 5.0
        self.moveit2.num_planning_attempts = 10
        self.moveit2.max_velocity = 0.1
        self.moveit2.max_acceleration = 0.1
        self.moveit2.cartesian_jump_threshold = 0.0

        # 10 Hz status pub
        self.status_timer = self.node.create_timer(0.1, self.publish_status)

        print("\033[1;32mGen3LiteArm ready. Emergency handler active.\033[0m")
        self.start_time = datetime.datetime.now()

    # ------------------- Gripper helpers -------------------
    def gripper_callback(self, msg: Float32):
        self.gripper_position = msg.data
        self.gripper_recent.append(msg.data)
        
        # Real-time object drop detection
        if self.object_drop_monitoring_active and self.gripper_closed:
            if self.gripper_position > OBJECT_DROPPED_THRESHOLD:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\033[1;31m[{timestamp}] !!! OBJECT DROPPED DETECTED !!! Gripper position: {self.gripper_position:.4f} > {OBJECT_DROPPED_THRESHOLD}\033[0m")
                self.node.get_logger().error(f"Object dropped! Gripper position: {self.gripper_position}")
                
                # Set the drop detected event
                self.object_drop_detected.set()
                
                # Change status to object dropped
                self.set_status(RobotStatus.OBJECT_DROPPED)
                
                # Stop current execution
                self.stop_current_execution()

    def start_drop_monitoring(self):
        """Start monitoring for object drops"""
        self.object_drop_monitoring_active = True
        self.object_drop_detected.clear()
        print("\033[1;36mStarted object drop monitoring\033[0m")

    def stop_drop_monitoring(self):
        """Stop monitoring for object drops"""
        self.object_drop_monitoring_active = False
        self.object_drop_detected.clear()
        print("\033[1;36mStopped object drop monitoring\033[0m")

    def stop_current_execution(self):
        """Stop current robot movement execution"""
        try:
            future = self.moveit2.get_execution_future()
            if future and not future.done():
                future.cancel()
                print("\033[1;33mCanceled current execution due to object drop\033[0m")
            self.node.get_logger().info("Current execution stopped due to object drop")
        except Exception as e:
            self.node.get_logger().error(f"Failed to stop execution: {e}")
            print(f"\033[1;31mFailed to stop execution: {e}\033[0m")

    def object_dropped(self) -> bool:
        return (
            self.gripper_position is not None
            and self.gripper_position > OBJECT_DROPPED_THRESHOLD
        )

    def print_gripper_stats(self):
        """Print last‑second gripper stats every 1 s."""
        if len(self.gripper_recent) == 0:
            return
        vals = list(self.gripper_recent)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Add warning if close to drop threshold
        warning = ""
        if self.gripper_closed and max(vals) > (OBJECT_DROPPED_THRESHOLD - 0.05):
            warning = " \033[1;33m⚠️  NEAR DROP THRESHOLD!\033[0m"
        
        print(
            f"\033[1;34m[{timestamp}] Gripper last 5 vals: "
            + ", ".join(f"{v:.4f}" for v in vals)
            + f" | min={min(vals):.4f} max={max(vals):.4f}{warning}\033[0m"
        )

    # ------------------- Status utils -------------------
    def set_status(self, status):
        with self.status_lock:
            if self.current_status != status:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\033[1;35m[{ts}] Status: {self.current_status} -> {status}\033[0m")
                self.current_status = status

    def publish_status(self):
        msg = String()
        msg.data = self.current_status
        self.status_publisher.publish(msg)

    def publish_require_help(self, x, y):
        """Publish require_help message with coordinates"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        help_data = {
            "timestamp": timestamp,
            "coordinates": {
                "x": float(x),
                "y": float(y)
            },
            "reason": f"Y coordinate {y:.4f} exceeds threshold {Y_THRESHOLD_FOR_HELP:.4f}"
        }
        
        help_msg = String()
        help_msg.data = json.dumps(help_data)
        
        self.require_help_publisher.publish(help_msg)
        
        print(f"\033[1;33m[{timestamp}] Published require_help message for coordinates X={x:.4f}, Y={y:.4f}\033[0m")
        print(f"\033[1;33m[{timestamp}] Help message content: {help_msg.data}\033[0m")

    def emergency_callback(self, msg):
        """Handle emergency stop messages"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prevent duplicate emergency handling
        if msg.data and not self.emergency_operation_in_progress:
            print(f"\033[1;31m[{timestamp}] !!! Emergency stop signal received !!!\033[0m")
            self.node.get_logger().error("Emergency stop signal received! Stopping current task and moving to Z position of 0.3")
            
            # Set emergency status
            self.set_status(RobotStatus.EMERGENCY_STOP)
            
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
            
            # Return to waiting status
            self.set_status(RobotStatus.WAITING_FOR_TASK)

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

    def move_to_home_position(self):
        """Move to home position"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\033[1;36m[{timestamp}] Moving to home position...\033[0m")
        
        self.set_status(RobotStatus.RETURNING_HOME)
        
        # Create a vertical orientation quaternion
        vertical_orientation = PyQuaternion(array=np.array([0.0, 1.0, 0.0, 0.0]))
        
        # Create home position pose
        home_pose = Pose()
        home_pose.position.x = HOME_X
        home_pose.position.y = HOME_Y
        home_pose.position.z = HOME_Z
        home_pose.orientation.x = float(vertical_orientation[0])
        home_pose.orientation.y = float(vertical_orientation[1])
        home_pose.orientation.z = float(vertical_orientation[2])
        home_pose.orientation.w = float(vertical_orientation[3])
        
        print(f"\033[1;36m[{timestamp}] Moving to home position: X={HOME_X}, Y={HOME_Y}, Z={HOME_Z}\033[0m")
        success = self.inverse_kinematic_movement(home_pose, cartesian=False)
        
        if success:
            print(f"\033[1;32m[{timestamp}] Successfully moved to home position\033[0m")
            return True
        else:
            print(f"\033[1;31m[{timestamp}] Failed to move to home position\033[0m")
            return False

    def check_gripper_success(self, is_closing=True):
        """
        Check if the gripper operation was successful
        is_closing: True if checking grip success, False if checking release success
        """
        # Wait for gripper to settle
        time.sleep(0.5)
        
        # Get current gripper position
        if self.gripper_position is None:
            print("\033[1;33mNo gripper position data available, assuming success\033[0m")
            return True
            
        if is_closing:
            # For closing, check if position is near the target close position
            # If position is too close to open, it means object wasn't gripped
            if abs(self.gripper_position - GRIPPER_OPEN_TARGET) < GRIPPER_DETECTION_THRESHOLD:
                print(f"\033[1;31mGripper failed to grasp object: position={self.gripper_position:.4f}\033[0m")
                return False
            else:
                print(f"\033[1;32mGripper successfully grasped object: position={self.gripper_position:.4f}\033[0m")
                return True
        else:
            # For opening, check if position is near the target open position
            if abs(self.gripper_position - GRIPPER_OPEN_TARGET) > GRIPPER_DETECTION_THRESHOLD:
                print(f"\033[1;31mGripper failed to release object: position={self.gripper_position:.4f}\033[0m")
                return False
            else:
                print(f"\033[1;32mGripper successfully released object: position={self.gripper_position:.4f}\033[0m")
                return True

    def inverse_kinematic_movement(self, target_pose, cartesian=False, status_during_move=None):
        """Execute inverse kinematic movement to target pose"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Store current pose for emergency handling
        self.current_pose = target_pose
        
        # Set status during movement if provided
        if status_during_move:
            self.set_status(status_during_move)
            
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
                        
                # Check if object dropped during planning
                if self.object_drop_detected.is_set():
                    print(f"\033[1;31m[{timestamp}] Object drop detected during planning - aborting movement\033[0m")
                    return False
                    
                rate.sleep()
                
            planning_time = (datetime.datetime.now() - planning_start).total_seconds()
            print(f"\033[1;36m[{timestamp}] Planning completed, duration: {planning_time:.2f} seconds\033[0m")
            print(f"\033[1;36m[{timestamp}] Starting execution of movement...\033[0m")
            
            execution_start = datetime.datetime.now()
            future = self.moveit2.get_execution_future()
            while not future.done():
                # Check if emergency operation is in progress during execution
                if self.emergency_operation_in_progress:
                    print(f"\033[1;31m[{timestamp}] Emergency operation in progress - waiting for completion\033[0m")
                    # Wait for emergency operation to complete
                    while self.emergency_operation_in_progress:
                        rate.sleep()
                    # After emergency operation completes, this movement may have been canceled
                    if future.done():
                        break
                        
                # Check if object dropped during execution
                if self.object_drop_detected.is_set():
                    print(f"\033[1;31m[{timestamp}] Object drop detected during execution - movement interrupted\033[0m")
                    return False
                    
                rate.sleep()
                
            execution_time = (datetime.datetime.now() - execution_start).total_seconds()
            total_time = (datetime.datetime.now() - move_start_time).total_seconds()
            print(f"\033[1;32m[{timestamp}] Movement complete! Execution time: {execution_time:.2f} seconds, Total time: {total_time:.2f} seconds\033[0m")
            
            return True
        except Exception as e:
            self.node.get_logger().error(f"Movement error: {e}")
            print(f"\033[1;31m[{timestamp}] Movement failed: {e}\033[0m")
            return False
            
    def shutdown(self):
        """Shutdown executor"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\033[1;36m[{timestamp}] Shutting down Gen3LiteArm executor...\033[0m")
        self.executor.shutdown()
        
        # Calculate runtime
        run_time = datetime.datetime.now() - self.start_time
        print(f"\033[1;36mSystem runtime: {run_time.total_seconds():.2f} seconds\033[0m")


def execute_single_pick_and_place_task(arm, gripper, x, y):
    """Execute a single pick and place task without retry"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\033[1;36m[{timestamp}] Executing pick and place task: X={x:.4f}, Y={y:.4f}\033[0m")
    
    # Create a vertical orientation quaternion
    vertical_orientation = PyQuaternion(array=np.array([0.0, 1.0, 0.0, 0.0]))
    
    # Create poses for the task
    approach_pose = Pose()
    approach_pose.position.x = float(x)
    approach_pose.position.y = float(y)
    approach_pose.position.z = float(SAFE_Z_HEIGHT)
    approach_pose.orientation.x = float(vertical_orientation[0])
    approach_pose.orientation.y = float(vertical_orientation[1])
    approach_pose.orientation.z = float(vertical_orientation[2])
    approach_pose.orientation.w = float(vertical_orientation[3])
    
    pick_pose = Pose()
    pick_pose.position.x = float(x)
    pick_pose.position.y = float(y)
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
    
    try:
        # Move to position above pick position first
        if not arm.inverse_kinematic_movement(approach_pose, cartesian=True, status_during_move=RobotStatus.MOVING_TO_TARGET):
            print(f"\033[1;31m[{timestamp}] Failed to move to approach position\033[0m")
            arm.set_status(RobotStatus.MOVE_TO_TARGET_FAILED)
            return False
        
        # Move to pick position
        if not arm.inverse_kinematic_movement(pick_pose, cartesian=True, status_during_move=RobotStatus.MOVING_TO_TARGET):
            print(f"\033[1;31m[{timestamp}] Failed to move to pick position\033[0m")
            arm.set_status(RobotStatus.MOVE_TO_TARGET_FAILED)
            return False
        
        # Close gripper to grasp object
        print(f"\033[1;36m[{timestamp}] Closing gripper to grasp object\033[0m")
        arm.set_status(RobotStatus.GRASPING)
        
        try:
            gripper.move_to_position(GRIPPER_CLOSE_TARGET)
            time.sleep(0.5)
            
            # Check if gripping was successful
            if not arm.check_gripper_success(is_closing=True):
                arm.set_status(RobotStatus.GRASPING_FAILED)
                print(f"\033[1;31m[{timestamp}] Grasping failed - will return to home position\033[0m")
                # Open gripper before returning home
                gripper.move_to_position(GRIPPER_OPEN_TARGET)
                return False
            
            # Mark gripper as closed and start monitoring for drops
            arm.gripper_closed = True
            arm.start_drop_monitoring()
                
        except Exception as e:
            print(f"\033[1;31m[{timestamp}] Grasping failed: {e}\033[0m")
            arm.set_status(RobotStatus.GRASPING_FAILED)
            # Open gripper before returning home
            gripper.move_to_position(GRIPPER_OPEN_TARGET)
            return False
        
        # Check if object was dropped before continuing
        if arm.object_drop_detected.is_set():
            print(f"\033[1;31m[{timestamp}] Object dropped after grasping\033[0m")
            arm.gripper_closed = False
            arm.stop_drop_monitoring()
            # Open gripper
            gripper.move_to_position(GRIPPER_OPEN_TARGET)
            return False
        
        # Move back to safe height
        if not arm.inverse_kinematic_movement(approach_pose, cartesian=True):
            print(f"\033[1;31m[{timestamp}] Failed to move back to safe height after pick\033[0m")
            
            # Check if object was dropped
            if arm.object_drop_detected.is_set():
                print(f"\033[1;31m[{timestamp}] Object dropped during movement\033[0m")
                arm.gripper_closed = False
                arm.stop_drop_monitoring()
                # Open gripper
                gripper.move_to_position(GRIPPER_OPEN_TARGET)
                return False
            
            arm.set_status(RobotStatus.OBJECT_DROPPED)
            return False
        
        # Move to put position
        if not arm.inverse_kinematic_movement(put_pose, cartesian=True, status_during_move=RobotStatus.MOVING_TO_RELEASE):
            print(f"\033[1;31m[{timestamp}] Failed to move to put position\033[0m")
            
            # Check if object was dropped
            if arm.object_drop_detected.is_set():
                print(f"\033[1;31m[{timestamp}] Object dropped during movement\033[0m")
                arm.gripper_closed = False
                arm.stop_drop_monitoring()
                # Open gripper
                gripper.move_to_position(GRIPPER_OPEN_TARGET)
                return False
            
            arm.set_status(RobotStatus.MOVE_TO_RELEASE_FAILED)
            return False
        
        # Stop monitoring drops as we're about to release
        arm.stop_drop_monitoring()
        arm.gripper_closed = False
        
        # Open gripper to release object
        print(f"\033[1;36m[{timestamp}] Opening gripper to release object\033[0m")
        arm.set_status(RobotStatus.RELEASING)
        
        try:
            gripper.move_to_position(GRIPPER_OPEN_TARGET)
            time.sleep(0.5)
            
            # Check if releasing was successful
            if not arm.check_gripper_success(is_closing=False):
                arm.set_status(RobotStatus.RELEASING_FAILED)
                print(f"\033[1;31m[{timestamp}] Release failed\033[0m")
                return False
                
        except Exception as e:
            print(f"\033[1;31m[{timestamp}] Releasing failed: {e}\033[0m")
            arm.set_status(RobotStatus.RELEASING_FAILED)
            return False
        
        # Move back to safe height
        if not arm.inverse_kinematic_movement(approach_pose, cartesian=True):
            print(f"\033[1;31m[{timestamp}] Failed to move back to safe height after put\033[0m")
            return False
        
        # Task completed successfully
        print(f"\033[1;32m[{timestamp}] Task completed successfully: X={x:.4f}, Y={y:.4f}\033[0m")
        return True
        
    except Exception as e:
        # Unexpected error
        print(f"\033[1;31m[{timestamp}] Unexpected error during task execution: {e}\033[0m")
        
        # Ensure monitoring is stopped and gripper state is reset
        arm.stop_drop_monitoring()
        arm.gripper_closed = False
        
        # Open gripper
        try:
            gripper.move_to_position(GRIPPER_OPEN_TARGET)
        except:
            pass
            
        return False


def get_single_task_input():
    """Get single task input from user"""
    print("\033[1;36m" + "="*50 + "\033[0m")
    print("\033[1;36mEnter coordinates for next pick and place task\033[0m")
    print("\033[1;36mEnter 'q' to quit the program\033[0m")
    print("\033[1;36m" + "="*50 + "\033[0m")
    
    while True:
        try:
            x_input = input("\033[1;36mEnter X coordinate (or 'q' to quit): \033[0m")
            if x_input.lower() == 'q':
                return None, None, True  # Signal to quit
            
            x = float(x_input)
            
            y_input = input("\033[1;36mEnter Y coordinate: \033[0m")
            y = float(y_input)
            
            # Check and warn if Y coordinate exceeds threshold
            if y > Y_THRESHOLD_FOR_HELP:
                print(f"\033[1;33mWarning: Y coordinate {y:.4f} exceeds threshold {Y_THRESHOLD_FOR_HELP:.4f}")
                print(f"This task will trigger a require_help message.\033[0m")
                confirm = input(f"\033[1;36mContinue with this coordinate? (y/n): \033[0m")
                if confirm.lower() != 'y':
                    continue
            
            # Display task summary
            print("\033[1;36m" + "-"*30 + "\033[0m")
            print(f"\033[1;36mTask Summary:\033[0m")
            print(f"\033[1;36mX: {x:.4f}, Y: {y:.4f}\033[0m")
            print(f"\033[1;36mPick Z: {PICK_Z_HEIGHT:.4f}, Put Z: {PUT_Z_HEIGHT:.4f}\033[0m")
            if y > Y_THRESHOLD_FOR_HELP:
                print(f"\033[1;33mStatus: ⚠️ Requires Help\033[0m")
            else:
                print(f"\033[1;32mStatus: ✓ Normal\033[0m")
            print("\033[1;36m" + "-"*30 + "\033[0m")
            
            return x, y, False
            
        except ValueError:
            print("\033[1;31mInvalid input. Please enter valid numbers for coordinates.\033[0m")


def main(args=None):
    """Main function with continuous loop execution"""
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
        log_file.write(f"Single Task Loop Execution start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Y threshold for help: {Y_THRESHOLD_FOR_HELP}\n")
        log_file.write(f"Object drop threshold: {OBJECT_DROPPED_THRESHOLD}\n")
        log_file.write(f"Home position: X={HOME_X}, Y={HOME_Y}, Z={HOME_Z}\n")
        log_file.write(f"{'='*50}\n")
    
    # Initialize ROS2 and robot components
    rclpy.init(args=args)
    
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\033[1;32m[{timestamp}] Initializing robotic arm system...\033[0m")
        
        arm = Gen3LiteArm()
        gripper = Gen3LiteGripper()
        
        # Move to home position initially
        print(f"\033[1;36m[{timestamp}] Moving to home position...\033[0m")
        if not arm.move_to_home_position():
            print(f"\033[1;31m[{timestamp}] Failed to move to home position\033[0m")
        
        # Set initial status
        arm.set_status(RobotStatus.WAITING_FOR_TASK)
        
        task_count = 0
        
        # Main continuous loop
        while True:
            try:
                # Get next task input
                x, y, should_quit = get_single_task_input()
                
                if should_quit:
                    print(f"\033[1;33m[{timestamp}] User requested quit, exiting...\033[0m")
                    break
                
                task_count += 1
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"Task {task_count} started at {timestamp}: X={x:.4f}, Y={y:.4f}\n")
                
                # Check if Y coordinate exceeds threshold
                if y > Y_THRESHOLD_FOR_HELP:
                    print(f"\033[1;33m[{timestamp}] Y coordinate {y:.4f} exceeds threshold {Y_THRESHOLD_FOR_HELP:.4f}\033[0m")
                    print(f"\033[1;33m[{timestamp}] Publishing require_help message...\033[0m")
                    
                    # Publish require_help message
                    arm.publish_require_help(x, y)
                    
                    # Set status to require_help
                    arm.set_status(RobotStatus.REQUIRE_HELP)
                    
                    # Skip this task
                    print(f"\033[1;33m[{timestamp}] Task requires help - skipping to next task\033[0m")
                    
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"Task {task_count} skipped (requires help) at {timestamp}\n")
                    
                    continue
                
                # Execute the task
                print(f"\033[1;36m[{timestamp}] Starting task {task_count}: X={x:.4f}, Y={y:.4f}\033[0m")
                
                success = execute_single_pick_and_place_task(arm, gripper, x, y)
                
                if success:
                    print(f"\033[1;32m[{timestamp}] Task {task_count} completed successfully!\033[0m")
                    arm.set_status(RobotStatus.TASK_COMPLETED)
                    
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"Task {task_count} completed successfully at {timestamp}\n")
                else:
                    print(f"\033[1;31m[{timestamp}] Task {task_count} failed\033[0m")
                    
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"Task {task_count} failed at {timestamp}\n")
                
                # Always return to home position after each task
                print(f"\033[1;36m[{timestamp}] Returning to home position...\033[0m")
                if not arm.move_to_home_position():
                    print(f"\033[1;31m[{timestamp}] Failed to return to home position\033[0m")
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"Failed to return to home position after task {task_count} at {timestamp}\n")
                else:
                    print(f"\033[1;32m[{timestamp}] Successfully returned to home position\033[0m")
                
                # Set status back to waiting for next task
                arm.set_status(RobotStatus.WAITING_FOR_TASK)
                
                print(f"\033[1;36m[{timestamp}] Task {task_count} cycle completed. Ready for next task.\033[0m")
                print("\033[1;36m" + "="*60 + "\033[0m")
                
            except KeyboardInterrupt:
                print(f"\033[1;33m[{timestamp}] User interrupted, exiting...\033[0m")
                break
            except Exception as e:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\033[1;31m[{timestamp}] Error during task execution: {e}\033[0m")
                
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"Error during task {task_count}: {e} at {timestamp}\n")
                
                # Try to return to home position after error
                try:
                    print(f"\033[1;36m[{timestamp}] Attempting to return to home position after error...\033[0m")
                    arm.move_to_home_position()
                    arm.set_status(RobotStatus.WAITING_FOR_TASK)
                except Exception as home_error:
                    print(f"\033[1;31m[{timestamp}] Failed to return to home position after error: {home_error}\033[0m")
                
                continue  # Continue to next task even after error

    except Exception as e:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\033[1;31m[{timestamp}] Main function error: {e}\033[0m")
        
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Main function error: {e} at {timestamp}\n")
    
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
            log_file.write(f"Program ended at {timestamp}\n")
            log_file.write(f"Total tasks attempted: {task_count}\n")
            log_file.write(f"{'='*50}\n")
        
        print(f"\033[1;36mExecution log saved to: {log_file_path}\033[0m")
        print(f"\033[1;36mTotal tasks attempted: {task_count}\033[0m")


if __name__ == '__main__':
    print(f"\033[1;32m{'='*60}\033[0m")
    print(f"\033[1;32m  Gen3Lite Robotic Arm - Single Task Loop Control\033[0m")
    print(f"\033[1;32m  Features:\033[0m")
    print(f"\033[1;32m  - Single task execution with continuous loop\033[0m")
    print(f"\033[1;32m  - Automatic return to home position after each task\033[0m")
    print(f"\033[1;32m  - No retry on grasping failure - direct home return\033[0m")
    print(f"\033[1;32m  - Emergency stop and object drop detection\033[0m")
    print(f"\033[1;32m  - Enter 'q' to quit program\033[0m")
    print(f"\033[1;32m  Execution time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\033[0m")
    print(f"\033[1;32m{'='*60}\033[0m")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\033[1;33mProgram interrupted by user\033[0m")
    except Exception as e:
        print(f"\033[1;31mProgram exception: {e}\033[0m")
    finally:
        print("\033[1;36mProgram exited\033[0m")
