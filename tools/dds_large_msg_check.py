#!/usr/bin/env python3
"""Measure how many large best-effort messages survive ROS 2 delivery on this machine.

Publishes PointCloud2 messages the size of a Hesai ATX cloud (4.5 MB) at 10 Hz from one
process and counts what a second process receives, with the same QoS the car uses
(best effort, depth 1). Run it before and after the Cyclone DDS tuning
(docs/DDS_TUNING.md) to see the difference without driving.

    python3 tools/dds_large_msg_check.py                 # 4.5 MB, 10 Hz, 20 s
    python3 tools/dds_large_msg_check.py --size-mb 1 --seconds 10

Uses whatever RMW_IMPLEMENTATION / CYCLONEDDS_URI the shell has, so source
/etc/dv/cyclonedds.env first to test the tuned setup. Exit code 0 always: it reports,
it does not judge. Do not run it while the pipeline is running; it adds 45 MB/s of traffic.
"""
import argparse
import subprocess
import sys
import time

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2

TOPIC = '/dds_large_msg_check'
QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)


def subscribe(seconds):
    rclpy.init()
    node = rclpy.create_node('dds_large_msg_check_sub')
    seen = set()
    node.create_subscription(PointCloud2, TOPIC, lambda m: seen.add(m.header.stamp.sec), QOS)
    print('READY', flush=True)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    print('SEEN ' + ' '.join(map(str, sorted(seen))), flush=True)


def publish(size_mb, rate, seconds):
    sub = subprocess.Popen([sys.executable, __file__, '--subscriber', str(seconds + 6)],
                           stdout=subprocess.PIPE, text=True)
    assert sub.stdout.readline().strip() == 'READY'
    rclpy.init()
    node = rclpy.create_node('dds_large_msg_check_pub')
    pub = node.create_publisher(PointCloud2, TOPIC, QOS)
    time.sleep(2.0)  # discovery

    msg = PointCloud2()
    msg.height, msg.point_step = 1, 26
    msg.width = int(size_mb * 1e6) // msg.point_step
    msg.row_step = msg.width * msg.point_step
    msg.data = bytes(msg.row_step)
    n = int(rate * seconds)
    t0 = time.monotonic()
    for i in range(1, n + 1):
        msg.header.stamp.sec = i
        pub.publish(msg)
        time.sleep(max(0.0, t0 + i / rate - time.monotonic()))
    out = sub.communicate()[0].split()
    seen = {int(x) for x in out[out.index('SEEN') + 1:]}
    got = len(seen & set(range(1, n + 1)))
    print(f'size {msg.row_step / 1e6:.2f} MB  rate {rate:g} Hz  sent {n}  received {got}  '
          f'lost {n - got} ({100.0 * (n - got) / n:.1f} %)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--size-mb', type=float, default=4.524)
    ap.add_argument('--rate', type=float, default=10.0)
    ap.add_argument('--seconds', type=float, default=20.0)
    ap.add_argument('--subscriber', type=float, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.subscriber:
        subscribe(a.subscriber)
    else:
        publish(a.size_mb, a.rate, a.seconds)


if __name__ == '__main__':
    main()
