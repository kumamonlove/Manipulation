#!/usr/bin/env python3
import rclpy
import time
import os
import numpy as np
import signal
import sys

from rclpy.node import Node
from geometry_msgs.msg import Pose
from pyquaternion import Quaternion as PyQuaternion
from threading import Thread
from rclpy.callback_groups import ReentrantCallbackGroup
from pymoveit2 import MoveIt2, MoveIt2State
from std_msgs.msg import Bool
from .gen3lite_pymoveit2 import Gen3LiteGripper


script_dir = os.path.dirname(os.path.realpath(__file__))
green_cube_pick_data = np.loadtxt('/home/andy/ros2_ws/src/manipulation/Pick.csv', delimiter=',')
green_cube_put_data = np.loadtxt('/home/andy/ros2_ws/src/manipulation/Put.csv', delimiter=',')

class Gen3LiteArm:
    def __init__(self):
        self.node = Node("gen3_lite_arm")
        self.callback_group = ReentrantCallbackGroup()
        
        # Add emergency state tracking
        self.emergency_active = False
        self.current_pose = Pose()
        
        # Subscribe to emergency stop topic
        self.emergency_subscription = self.node.create_subscription(
            Bool,
            '/emergency_stop',
            self.emergency_callback,
            10  # QoS profile
        )
        
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

        self.moveit2.pipeline_id = 'pilz_industrial_motion_planner'
        self.moveit2.planner_id = 'LIN'
        self.moveit2.allowed_planning_time = 5.0
        self.moveit2.num_planning_attempts = 10
        self.moveit2.max_velocity = 0.1
        self.moveit2.max_acceleration = 0.1
        self.moveit2.cartesian_jump_threshold = 0.0
        
        print("Gen3LiteArm initialized. Emergency stop handler active.")

    def emergency_callback(self, msg):
        """Handle emergency stop messages"""
        if msg.data and not self.emergency_active:
            self.emergency_active = True
            self.node.get_logger().error("EMERGENCY STOP RECEIVED! Stopping current task and raising Z by 5cm")
            print("\033[1;31m!!! EMERGENCY STOP RECEIVED !!!\033[0m")
            
            # Stop any current movement by canceling execution
            try:
                self.moveit2.stop_execution()
                self.node.get_logger().info("Stopped current execution")
            except Exception as e:
                self.node.get_logger().error(f"Failed to stop execution: {e}")
            
            # Raise Z by 5cm
            self.emergency_raise_z()

    def emergency_raise_z(self):
        """Raise the arm by 5cm in Z direction during emergency"""
        # Create emergency pose 5cm higher in Z
        emergency_pose = Pose()
        emergency_pose.position.x = self.current_pose.position.x
        emergency_pose.position.y = self.current_pose.position.y
        emergency_pose.position.z = self.current_pose.position.z + 0.05  # Add 5cm
        emergency_pose.orientation = self.current_pose.orientation
        
        print(f"\033[1;33mEMERGENCY: Moving to Z+5cm position: {emergency_pose.position}\033[0m")
        
        # Temporarily increase speed for emergency movement
        original_velocity = self.moveit2.max_velocity
        original_acceleration = self.moveit2.max_acceleration
        
        try:
            # Set faster motion parameters for emergency
            self.moveit2.max_velocity = 0.3
            self.moveit2.max_acceleration = 0.3
            
            # Execute emergency move
            self.moveit2.move_to_pose(
                pose=emergency_pose,
                cartesian=True,
                cartesian_max_step=0.01
            )
            
            # Wait for completion
            rate = self.node.create_rate(10)
            while self.moveit2.query_state() != MoveIt2State.EXECUTING:
                rate.sleep()
            future = self.moveit2.get_execution_future()
            while not future.done():
                rate.sleep()
                
            print("\033[1;32mEmergency raise completed\033[0m")
            
        except Exception as e:
            print(f"\033[1;31mEmergency movement failed: {e}\033[0m")
        finally:
            # Restore original parameters
            self.moveit2.max_velocity = original_velocity
            self.moveit2.max_acceleration = original_acceleration

    def inverse_kinematic_movement(self, target_pose, cartesian=False):
        # Store current pose for emergency handling
        self.current_pose = target_pose
        
        # Check if emergency is active before moving
        if self.emergency_active:
            self.node.get_logger().error("Emergency active - movement canceled")
            print("\033[1;31mEmergency active - movement canceled\033[0m")
            return False
            
        self.node.get_logger().info(
            f"Moving to pose: {target_pose.position} {target_pose.orientation} with cartesian={cartesian}"
        )
        
        try:
            self.moveit2.move_to_pose(
                pose=target_pose,
                cartesian=cartesian,
                cartesian_max_step=0.0025,
                cartesian_fraction_threshold=0.0,
            )
            rate = self.node.create_rate(10)
            while self.moveit2.query_state() != MoveIt2State.EXECUTING:
                # Check for emergency during planning
                if self.emergency_active:
                    self.moveit2.stop_execution()
                    return False
                rate.sleep()
                
            future = self.moveit2.get_execution_future()
            while not future.done():
                # Check for emergency during execution
                if self.emergency_active:
                    self.moveit2.stop_execution()
                    return False
                rate.sleep()
            return True
        except Exception as e:
            self.node.get_logger().error(f"Movement error: {e}")
            return False
            
    def shutdown(self):
        self.executor.shutdown()


def slerp(data, qStart, qEnd, arm):
    qList = []
    for q in PyQuaternion.intermediates(qStart, qEnd, len(data) - 2, include_endpoints=True):
        qList.append(q.elements)
    for i in range(len(data)):
        pose = Pose()
        pose.position.x = data[i][0]
        pose.position.y = data[i][1]
        pose.position.z = data[i][2]
        pose.orientation.x = qList[i][0]
        pose.orientation.y = qList[i][1]
        pose.orientation.z = qList[i][2]
        pose.orientation.w = qList[i][3]
        
        # If movement fails (due to emergency), break out of the loop
        if not arm.inverse_kinematic_movement(pose):
            print(f"\033[1;31mSLERP interrupted at control point {i}\033[0m")
            break
            
        print(f"Reached control point {i}")


def pick_and_place(pick_data, put_data, arm, gripper, qStart, qPick, qPut):
    print("Moving to pick position (via SLERP)")
    slerp(pick_data, qStart, qPick, arm)
    
    # Check if emergency is active before continuing
    if arm.emergency_active:
        print("\033[1;31mEmergency active - skipping gripper close\033[0m")
        return

    print("Closing gripper to grasp")
    gripper.move_to_position(0.7)
    time.sleep(0.5)
    
    # Check again after gripper operation
    if arm.emergency_active:
        print("\033[1;31mEmergency active - skipping put movement\033[0m")
        return

    print("Moving to put position (via SLERP)")
    slerp(put_data, qPick, qPut, arm)
    
    # Final check before releasing
    if arm.emergency_active:
        print("\033[1;31mEmergency active - skipping gripper release\033[0m")
        return

    print("Opening gripper to release")
    gripper.move_to_position(0.0)


def main(args=None):
    # Set up signal handler for Ctrl+C
    def signal_handler(sig, frame):
        print("Received shutdown signal, cleaning up...")
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    rclpy.init(args=args)
    print("Task Start")
    
    try:
        arm = Gen3LiteArm()
        gripper = Gen3LiteGripper()
        qStartGreenCube = PyQuaternion(array=np.array([1, 0, 0, 0]))
        qPickGreenCube = PyQuaternion(array=np.array([1, 0, 0, 0]))
        qPutGreenCube = PyQuaternion(array=np.array([1, 0, 0, 0]))

        pick_and_place(green_cube_pick_data, green_cube_put_data, arm, gripper,
                      qStartGreenCube, qPickGreenCube, qPutGreenCube)

        # Only show Task Finished if no emergency occurred
        if not arm.emergency_active:
            print("Task Finished")
        else:
            print("\033[1;33mTask terminated due to emergency\033[0m")
            
    except Exception as e:
        print(f"Error in main function: {e}")
    finally:
        # Always clean up properly
        gripper.shutdown()
        arm.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()