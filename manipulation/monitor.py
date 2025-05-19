#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import datetime
import os
import sys
from collections import deque
import time

class RobotStatusMonitor(Node):
    def __init__(self):
        super().__init__('robot_status_monitor')
        
        # 状态历史记录（保留最近100条记录）
        self.status_history = deque(maxlen=100)
        self.last_status = None
        self.start_time = datetime.datetime.now()
        self.status_count = {}
        
        # 创建状态订阅者
        self.status_subscription = self.create_subscription(
            String,
            '/robot_status',
            self.status_callback,
            10
        )
        
        # 创建定时器，每秒显示一次统计信息
        self.stats_timer = self.create_timer(1.0, self.display_statistics)
        
        # 日志文件
        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.log_file_path = os.path.join(script_dir, "robot_status_log.txt")
        
        # 初始化日志文件
        with open(self.log_file_path, "w") as log_file:
            log_file.write(f"Robot Status Monitor Log\n")
            log_file.write(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"{'='*60}\n\n")
        
        print("\033[1;32m" + "="*60 + "\033[0m")
        print("\033[1;32mRobot Status Monitor Started\033[0m")
        print("\033[1;32m" + "="*60 + "\033[0m")
        print(f"\033[1;36mMonitoring robot status on topic: /robot_status\033[0m")
        print(f"\033[1;36mLog file: {self.log_file_path}\033[0m")
        print(f"\033[1;36mPress Ctrl+C to stop monitoring\033[0m")
        print("\033[1;32m" + "="*60 + "\033[0m\n")

    def status_callback(self, msg):
        """处理机器人状态消息"""
        timestamp = datetime.datetime.now()
        status = msg.data
        
        # 记录状态变化
        if status != self.last_status:
            status_change = {
                'timestamp': timestamp,
                'status': status,
                'previous_status': self.last_status
            }
            
            self.status_history.append(status_change)
            
            # 更新统计
            self.status_count[status] = self.status_count.get(status, 0) + 1
            
            # 控制台输出状态变化
            time_str = timestamp.strftime('%H:%M:%S.%f')[:-3]  # 包含毫秒
            if self.last_status:
                print(f"\033[1;33m[{time_str}] Status Change: \033[1;31m{self.last_status}\033[0m -> \033[1;32m{status}\033[0m")
            else:
                print(f"\033[1;33m[{time_str}] Initial Status: \033[1;32m{status}\033[0m")
            
            # 根据状态类型使用不同颜色显示
            color_code = self.get_status_color(status)
            print(f"{color_code}Current Status: {status}\033[0m")
            
            # 写入日志文件
            with open(self.log_file_path, "a") as log_file:
                log_file.write(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] ")
                if self.last_status:
                    log_file.write(f"{self.last_status} -> {status}\n")
                else:
                    log_file.write(f"Initial: {status}\n")
            
            self.last_status = status

    def get_status_color(self, status):
        """根据状态返回对应的颜色代码"""
        status_colors = {
            'waiting_for_task': '\033[1;36m',      # 青色
            'moving_to_target': '\033[1;34m',      # 蓝色
            'grasping': '\033[1;33m',              # 黄色
            'moving_to_release': '\033[1;35m',     # 紫色
            'releasing': '\033[1;33m',             # 黄色
            'task_completed': '\033[1;32m',        # 绿色
            'emergency_stop': '\033[1;31m',        # 红色
            'move_to_target_failed': '\033[1;31m', # 红色
            'move_to_release_failed': '\033[1;31m',# 红色
            'grasping_failed': '\033[1;31m',       # 红色
            'releasing_failed': '\033[1;31m',      # 红色
            'object_dropped': '\033[1;31m',        # 红色
        }
        return status_colors.get(status, '\033[0m')  # 默认白色

    def display_statistics(self):
        """每秒显示统计信息"""
        if not self.status_history:
            return
        
        # 清屏并显示统计信息
        # print("\033[2J\033[H", end="")  # 清屏（可选）
        
        runtime = datetime.datetime.now() - self.start_time
        current_status = self.last_status if self.last_status else "Unknown"
        
        print(f"\n\033[1;36m{'='*60}\033[0m")
        print(f"\033[1;36mRobot Status Monitor - Runtime: {str(runtime).split('.')[0]}\033[0m")
        print(f"\033[1;36m{'='*60}\033[0m")
        print(f"\033[1;37mCurrent Status: {self.get_status_color(current_status)}{current_status}\033[0m")
        print(f"\033[1;37mTotal Status Changes: {len(self.status_history)}\033[0m")
        
        # 显示状态统计
        print(f"\n\033[1;36mStatus Statistics:\033[0m")
        print(f"\033[1;36m{'-'*40}\033[0m")
        for status, count in sorted(self.status_count.items()):
            color = self.get_status_color(status)
            print(f"{color}{status:<25} {count:>3} times\033[0m")
        
        # 显示最近的状态变化
        print(f"\n\033[1;36mRecent Status Changes (Last 5):\033[0m")
        print(f"\033[1;36m{'-'*50}\033[0m")
        recent_changes = list(self.status_history)[-5:]
        for change in recent_changes:
            time_str = change['timestamp'].strftime('%H:%M:%S.%f')[:-3]
            prev_status = change['previous_status'] or "None"
            curr_status = change['status']
            print(f"\033[1;33m[{time_str}]\033[0m {prev_status} -> \033[1;32m{curr_status}\033[0m")
        
        print(f"\033[1;36m{'='*60}\033[0m\n")

    def save_detailed_log(self):
        """保存详细日志"""
        timestamp = datetime.datetime.now()
        detailed_log_path = self.log_file_path.replace('.txt', '_detailed.txt')
        
        with open(detailed_log_path, 'w') as log_file:
            log_file.write(f"Detailed Robot Status Monitor Log\n")
            log_file.write(f"Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"{'='*80}\n\n")
            
            # 写入统计信息
            log_file.write("STATUS STATISTICS:\n")
            log_file.write(f"{'-'*40}\n")
            for status, count in sorted(self.status_count.items()):
                log_file.write(f"{status:<30} {count:>5} times\n")
            
            log_file.write(f"\n\nSTATUS CHANGE HISTORY:\n")
            log_file.write(f"{'-'*80}\n")
            log_file.write(f"{'Timestamp':<25} {'Previous Status':<25} {'Current Status':<25}\n")
            log_file.write(f"{'-'*80}\n")
            
            for change in self.status_history:
                time_str = change['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                prev_status = change['previous_status'] or "None"
                curr_status = change['status']
                log_file.write(f"{time_str:<25} {prev_status:<25} {curr_status:<25}\n")
        
        print(f"\033[1;32mDetailed log saved to: {detailed_log_path}\033[0m")

    def shutdown(self):
        """清理资源并保存最终日志"""
        print(f"\n\033[1;33mShutting down Robot Status Monitor...\033[0m")
        self.save_detailed_log()
        
        # 保存最终统计到主日志文件
        with open(self.log_file_path, "a") as log_file:
            runtime = datetime.datetime.now() - self.start_time
            log_file.write(f"\n{'='*60}\n")
            log_file.write(f"Monitor stopped: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Total runtime: {str(runtime).split('.')[0]}\n")
            log_file.write(f"Total status changes: {len(self.status_history)}\n")
            log_file.write(f"{'='*60}\n")
        
        print(f"\033[1;32mLog files saved successfully\033[0m")


def main():
    """主函数"""
    
    # 初始化ROS2
    rclpy.init()
    
    try:
        # 创建状态监控节点
        monitor = RobotStatusMonitor()
        
        # 设置信号处理器
        def signal_handler(sig, frame):
            print(f"\n\033[1;33mReceived interrupt signal, shutting down...\033[0m")
            monitor.shutdown()
            rclpy.shutdown()
            sys.exit(0)
        
        import signal
        signal.signal(signal.SIGINT, signal_handler)
        
        # 运行节点
        rclpy.spin(monitor)
        
    except Exception as e:
        print(f"\033[1;31mError: {e}\033[0m")
    finally:
        # 清理资源
        if 'monitor' in locals():
            monitor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
