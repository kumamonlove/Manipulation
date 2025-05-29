#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import datetime
import os
import sys
from collections import deque

class RobotStatusMonitor(Node):
    def __init__(self):
        super().__init__('robot_status_monitor')

        # === 监控用变量 ===
        self.status_history = deque(maxlen=100)   # 最近 100 条状态变化
        self.last_status = None                   # 上一次状态
        self.start_time = datetime.datetime.now() # 启动时间
        self.status_count = {}                    # 各状态出现次数
        self.last_displayed_idx = 0               # ★ 上次已显示的变化数量

        # === 订阅 / 定时器 ===
        self.create_subscription(String,
                                 '/robot_status',
                                 self.status_callback,
                                 10)
        self.create_timer(1.0, self.display_statistics)  # 1 Hz 定时打印

        # === 日志文件 ===
        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.log_file_path = os.path.join(script_dir, 'robot_status_log.txt')
        with open(self.log_file_path, 'w') as f:
            f.write('Robot Status Monitor Log\n')
            f.write(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write('=' * 60 + '\n\n')

        # === 启动提示 ===
        print('\033[1;32m' + '=' * 60 + '\033[0m')
        print('\033[1;32mRobot Status Monitor Started\033[0m')
        print('\033[1;32m' + '=' * 60 + '\033[0m')
        print('\033[1;36mMonitoring topic: /robot_status\033[0m')
        print(f'\033[1;36mLog file: {self.log_file_path}\033[0m')
        print('\033[1;36mPress Ctrl+C to stop monitoring\033[0m\n')

    # ---------------------------------------------------------------------
    # 回调：收到状态消息
    # ---------------------------------------------------------------------
    def status_callback(self, msg: String):
        timestamp = datetime.datetime.now()
        status = msg.data

        # 若状态相同→直接返回
        if status == self.last_status:
            return

        # 记录状态变化
        change = {
            'timestamp': timestamp,
            'status': status,
            'previous_status': self.last_status
        }
        self.status_history.append(change)

        # 更新统计计数
        self.status_count[status] = self.status_count.get(status, 0) + 1

        # 控制台即时打印一次变化
        ts = timestamp.strftime('%H:%M:%S.%f')[:-3]
        if self.last_status:
            print(f"\033[1;33m[{ts}] Status Change: "
                  f"\033[1;31m{self.last_status}\033[0m -> "
                  f"\033[1;32m{status}\033[0m")
        else:
            print(f"\033[1;33m[{ts}] Initial Status: "
                  f"\033[1;32m{status}\033[0m")

        # 彩色显示当前状态
        print(f"{self.get_status_color(status)}Current Status: {status}\033[0m")

        # 写入日志
        with open(self.log_file_path, 'a') as f:
            f.write(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] ")
            if self.last_status:
                f.write(f"{self.last_status} -> {status}\n")
            else:
                f.write(f"Initial: {status}\n")

        self.last_status = status

    # ---------------------------------------------------------------------
    # 每秒统计打印——仅当有新变化时输出
    # ---------------------------------------------------------------------
    def display_statistics(self):
        # 没收过任何状态 → 不打印
        if not self.status_history:
            return

        # 若状态变化数量未增加 → 无新消息 → 不打印
        if len(self.status_history) == self.last_displayed_idx:
            return

        # 有新增变化 → 更新计数并打印
        self.last_displayed_idx = len(self.status_history)

        runtime = datetime.datetime.now() - self.start_time
        current_status = self.last_status or 'Unknown'

        print(f"\n\033[1;36m{'='*60}\033[0m")
        print(f"\033[1;36mRobot Status Monitor - Runtime: "
              f"{str(runtime).split('.')[0]}\033[0m")
        print(f"\033[1;36m{'='*60}\033[0m")
        print(f"\033[1;37mCurrent Status: "
              f"{self.get_status_color(current_status)}{current_status}\033[0m")
        print(f"\033[1;37mTotal Status Changes: "
              f"{len(self.status_history)}\033[0m\n")

        # 状态统计
        print(f"\033[1;36mStatus Statistics:\033[0m")
        print(f"\033[1;36m{'-'*40}\033[0m")
        for st, cnt in sorted(self.status_count.items()):
            print(f"{self.get_status_color(st)}{st:<25} {cnt:>3} times\033[0m")

        # 最近 5 次变化
        print(f"\n\033[1;36mRecent Status Changes (Last 5):\033[0m")
        print(f"\033[1;36m{'-'*50}\033[0m")
        for c in list(self.status_history)[-5:]:
            ts = c['timestamp'].strftime('%H:%M:%S.%f')[:-3]
            prev = c['previous_status'] or 'None'
            curr = c['status']
            print(f"\033[1;33m[{ts}]\033[0m "
                  f"{prev} -> \033[1;32m{curr}\033[0m")

        print(f"\033[1;36m{'='*60}\033[0m\n")

    # ---------------------------------------------------------------------
    # 工具函数：按状态给颜色
    # ---------------------------------------------------------------------
    def get_status_color(self, status):
        return {
            'waiting_for_task': '\033[1;36m',
            'moving_to_target': '\033[1;34m',
            'grasping':         '\033[1;33m',
            'moving_to_release':'\033[1;35m',
            'releasing':        '\033[1;33m',
            'task_completed':   '\033[1;32m',
            'emergency_stop':   '\033[1;31m',
            'move_to_target_failed':'\033[1;31m',
            'move_to_release_failed':'\033[1;31m',
            'grasping_failed':  '\033[1;31m',
            'releasing_failed': '\033[1;31m',
            'object_dropped':   '\033[1;31m',
        }.get(status, '\033[0m')

    # ---------------------------------------------------------------------
    # 保存详细日志 & 关机清理
    # ---------------------------------------------------------------------
    def save_detailed_log(self):
        ts = datetime.datetime.now()
        detail_path = self.log_file_path.replace('.txt', '_detailed.txt')
        with open(detail_path, 'w') as f:
            f.write('Detailed Robot Status Monitor Log\n')
            f.write(f"Generated: {ts.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write('='*80 + '\n\n')

            f.write('STATUS STATISTICS:\n' + '-'*40 + '\n')
            for st, cnt in sorted(self.status_count.items()):
                f.write(f"{st:<30} {cnt:>5} times\n")

            f.write('\nSTATUS CHANGE HISTORY:\n' + '-'*80 + '\n')
            f.write(f"{'Timestamp':<25} {'Previous':<25} {'Current':<25}\n")
            f.write('-'*80 + '\n')
            for c in self.status_history:
                ts = c['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                prev = c['previous_status'] or 'None'
                curr = c['status']
                f.write(f"{ts:<25} {prev:<25} {curr:<25}\n")
        print(f"\033[1;32mDetailed log saved to: {detail_path}\033[0m")

    def shutdown(self):
        print('\n\033[1;33mShutting down Robot Status Monitor...\033[0m')
        self.save_detailed_log()

        with open(self.log_file_path, 'a') as f:
            runtime = datetime.datetime.now() - self.start_time
            f.write('\n' + '='*60 + '\n')
            f.write(f"Monitor stopped: "
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total runtime: {str(runtime).split('.')[0]}\n")
            f.write(f"Total status changes: {len(self.status_history)}\n")
            f.write('='*60 + '\n')
        print('\033[1;32mLog files saved successfully\033[0m')

# -------------------------------------------------------------------------
# 主函数
# -------------------------------------------------------------------------
def main():
    rclpy.init()
    monitor = RobotStatusMonitor()

    # 捕获 Ctrl-C
    import signal
    def handler(sig, frame):
        monitor.shutdown()
        rclpy.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)

    try:
        rclpy.spin(monitor)
    finally:
        if rclpy.ok():
            monitor.shutdown()
            rclpy.shutdown()

if __name__ == '__main__':
    main()

