"""Turns the files in results/ into results/summary.txt lines.

Arguments: the step count to extrapolate to, and the step count the run trained.
"""
import csv
import re
import sys
from datetime import datetime


def parse_time(stamp):
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


target_steps, trained_steps = int(sys.argv[1]), int(sys.argv[2])
marks = [line.split() for line in open("results/timeline.log")]
times = lambda name: [parse_time(stamp) for stamp, mark in marks if mark == name]

# Setup from the first attempt, since a retry skips installs; training from the last attempt.
print(f"setup_minutes {(times('setup_end')[0] - times('setup_start')[0]).total_seconds() / 60:.1f}")
train_start = times("train_start")[-1]
print(f"train_wall_hours {(times('train_end')[-1] - train_start).total_seconds() / 3600:.3f}")


def checkpoint_step(name):
    # Intermediate saves carry the step (bench_000000250.safetensors); the final save doesn't (bench.safetensors).
    match = re.search(r"_(\d+)\.safetensors$", name)
    return int(match.group(1)) if match else trained_steps


points = sorted((checkpoint_step(name), int(timestamp)) for name, timestamp in csv.reader(open("results/checkpoints.csv")))
if len(points) > 1:
    (first_step, first_time), (last_step, last_time) = points[0], points[-1]
    rate = (last_step - first_step) / ((last_time - first_time) / 3600)
    print(f"steady_steps_per_hour {rate:.1f}  # checkpoint {first_step} to {last_step}, includes sampling every 250 steps")
    blocks = [(later[1] - earlier[1]) / 60 for earlier, later in zip(points, points[1:])]
    print(f"block_minutes_min_max {min(blocks):.1f} {max(blocks):.1f}  # per 250 steps; a narrow spread is what makes extrapolation safe")
    startup_hours = (first_time - train_start.timestamp()) / 3600 - first_step / rate
    print(f"startup_hours {startup_hours:.3f}  # model load and first sample, paid once")
    print(f"extrapolated_{target_steps}_step_hours {startup_hours + target_steps / rate:.3f}")

rows = [row for row in csv.reader(open("results/gpu.csv")) if len(row) == 5]
power = [float(row[1]) for row in rows]
print(f"gpu_power_avg_w {sum(power) / len(power):.0f}")
print(f"gpu_memory_peak_mib {max(float(row[2]) for row in rows):.0f}")
print(f"gpu_temp_max_c {max(float(row[4]) for row in rows):.0f}")
