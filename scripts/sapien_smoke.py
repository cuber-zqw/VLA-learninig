"""SAPIEN 最小验证：自由落体、接触；可选 NVIDIA 相机渲染。"""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true')
args = parser.parse_args()
if args.render:
    probe = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if probe.returncode:
        raise SystemExit('NVIDIA 驱动尚未可用，请先完成 MOK 登记并重启。\n' + probe.stderr)

import numpy as np
import sapien
import imageio.v2 as imageio

out = ROOT / '2026-10-19' / 'sapien_smoke'
out.mkdir(parents=True, exist_ok=True)
systems = [sapien.physx.PhysxCpuSystem()]
if args.render:
    systems.append(sapien.render.RenderSystem())
scene = sapien.Scene(systems)
scene.set_timestep(1 / 240)
scene.add_ground(0, render=args.render)
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[0.05, 0.05, 0.05])
if args.render:
    builder.add_box_visual(half_size=[0.05, 0.05, 0.05], material=[0.8, 0.15, 0.05, 1])
box = builder.build(name='falling_box')
box.set_pose(sapien.Pose([0, 0, 0.6]))
video = None
if args.render:
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [2, 2, 2], shadow=True)
    camera = scene.add_camera('camera', 640, 480, np.deg2rad(60), 0.01, 10)
    position = np.array([1.0, -1.0, 0.8])
    forward = np.array([0, 0, 0.15]) - position
    forward /= np.linalg.norm(forward)
    left = np.cross([0, 0, 1], forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([forward, left, up])
    matrix[:3, 3] = position
    camera.set_pose(sapien.Pose(matrix))
    video = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '640x480', '-r', '30', '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out / 'falling_box.mp4')], stdin=subprocess.PIPE)

positions = []
try:
    for step in range(480):
        positions.append(box.get_pose().p.copy())
        scene.step()
        if args.render and step % 8 == 0:
            scene.update_render()
            camera.take_picture()
            rgb = (np.clip(camera.get_picture('Color')[..., :3], 0, 1) * 255).astype(np.uint8)
            if step == 240:
                if float(rgb.std()) < 1:
                    raise RuntimeError('相机画面接近纯色，请检查渲染。')
                imageio.imwrite(out / 'camera.png', rgb)
            video.stdin.write(rgb.tobytes())
finally:
    if video is not None:
        video.stdin.close()
        if video.wait() != 0:
            raise RuntimeError('ffmpeg 编码失败')
positions = np.asarray(positions)
final_z = float(box.get_pose().p[2])
assert abs(final_z - 0.05) < 0.015, f'自由落体接触检查失败：z={final_z}'
np.save(out / 'positions.npy', positions)
result = {'physics_pass': True, 'render_pass': args.render, 'steps': 480, 'timestep': 1/240, 'start_z': float(positions[0, 2]), 'final_z': final_z, 'note': 'SAPIEN 基础验证，尚非 RoboTwin 专家任务；positions 是物体位置，不是机械臂动作。'}
(out / ('render_result.json' if args.render else 'physics_result.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
print('输出目录:', out)
