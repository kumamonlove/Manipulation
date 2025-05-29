import rclpy
from rclpy.node import Node
import csv
import os
import threading
import sys
import termios
import tty
from sensor_msgs.msg import JointState
from datetime import datetime

class KinovaDataLogger(Node):
    def __init__(self):
        super().__init__('kinova_data_logger')
        self.joint_names_of_interest = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self.current_efforts = None
        self.data = []
        self.recording = False

        self.subscription = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.timer = self.create_timer(0.01, self.timer_callback)  # 100Hz
        self.get_logger().info("Kinova Data Logger started at 100Hz.")

        self.listener_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.listener_thread.start()

    def joint_state_callback(self, msg: JointState):
        efforts = []
        for j_name in self.joint_names_of_interest:
            if j_name in msg.name:
                idx = msg.name.index(j_name)
                if idx < len(msg.effort):
                    efforts.append(msg.effort[idx])
                else:
                    self.get_logger().warn(f"Index {idx} out of range for joint {j_name}")
                    return
            else:
                self.get_logger().warn(f"Joint {j_name} not found in /joint_states!")
                return

        if len(efforts) == 6:
            self.current_efforts = efforts
        else:
            self.get_logger().warn("Incomplete effort state received.")

    def timer_callback(self):
        if self.recording and self.current_efforts is not None:
            row = list(self.current_efforts)
            self.data.append(row)

    def save_data(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.expanduser("~/data")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"kinova_effort_data_{timestamp}.csv")
        header = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']
        try:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                writer.writerows(self.data)
            self.get_logger().info(f"Data saved to: {file_path}")
        except Exception as e:
            self.get_logger().error(f"Error saving data: {e}")

    def keyboard_listener(self):
        def get_key():
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                return sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        while rclpy.ok():
            key = get_key()
            if key == 's':
                if not self.recording:
                    self.recording = True
                    self.get_logger().info("Recording started.")
            elif key == 'q':
                if self.recording:
                    self.get_logger().info("Stopping recording and saving data...")
                    self.recording = False
                    self.save_data()
                rclpy.shutdown()
                break

def main(args=None):
    rclpy.init(args=args)
    node = KinovaDataLogger()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()

