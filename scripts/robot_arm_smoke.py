"""Load the pinned RoboTwin Aloha asset and verify joint drive tracking.
This checks robot loading/control/rendering, not expert task success.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=ROOT / '2026-10-20/robot_arm_smoke')
args = parser.parse_args()
out = args.output_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
import numpy as np
import sapien
import imageio.v2 as imageio

scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem()])
scene.set_timestep(1 / 250)
scene.add_ground(0)
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [2, 2, 2], shadow=True)
loader = scene.create_urdf_loader()
loader.fix_root_link = True
urdf = ROOT / 'RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf'
robot = loader.load(str(urdf))
if robot is None:
    raise RuntimeError('URDF loading failed')
robot.set_pose(sapien.Pose([0, -0.65, 0], [0.70710678, 0, 0, 0.70710678]))
joints = robot.get_active_joints()
names = [j.name for j in joints]
for j in joints:
    j.set_drive_properties(stiffness=1000, damping=200)
    j.set_drive_target(0)
robot.set_qpos(np.zeros(robot.dof))
index = names.index('fl_joint1')
limits = robot.get_qlimits()[index]
assert limits[0] <= 0.2 <= limits[1]
camera = scene.add_camera('overview', 640, 480, np.deg2rad(60), 0.01, 10)
pos = np.array([1.5, 0.8, 1.5])
forward = np.array([0, -0.25, 0.8]) - pos
forward /= np.linalg.norm(forward)
left = np.cross([0, 0, 1], forward)
left /= np.linalg.norm(left)
mat = np.eye(4)
mat[:3, :3] = np.column_stack([forward, left, np.cross(forward, left)])
mat[:3, 3] = pos
camera.set_pose(sapien.Pose(mat))
video = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '640x480', '-r', '25', '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out / 'joint_drive.mp4')], stdin=subprocess.PIPE)
qpos, targets, frames = [], [], 0
try:
    for step in range(1000):
        # Smoothly command one joint from 0 to 0.2 radians, then hold.
        target = float(0.2 * min(max((step - 250) / 500, 0), 1))
        joints[index].set_drive_target(target)
        robot.set_qf(robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
        scene.step()
        qpos.append(robot.get_qpos().copy())
        targets.append(target)
        if step % 10 == 0:
            scene.update_render()
            camera.take_picture()
            rgb = (np.clip(camera.get_picture('Color')[..., :3], 0, 1) * 255).astype(np.uint8)
            video.stdin.write(rgb.tobytes())
            frames += 1
            if step == 900:
                imageio.imwrite(out / 'camera.png', rgb)
finally:
    video.stdin.close()
    if video.wait() != 0:
        raise RuntimeError('ffmpeg failed')
qpos = np.asarray(qpos)
targets = np.asarray(targets)
np.savez_compressed(out / 'joint_drive.npz', qpos=qpos, target=targets, time=np.arange(1, 1001) / 250)
error = abs(float(qpos[-1, index]) - targets[-1])
result = {'robot_load_pass': True, 'joint_tracking_pass': bool(error < 0.03), 'render_pass': True, 'expert_task_pass': False, 'joint_names': names, 'driven_joint': names[index], 'target_rad': targets[-1], 'actual_rad': float(qpos[-1, index]), 'absolute_error_rad': error, 'steps': 1000, 'timestep': 1 / 250, 'frames': frames, 'note': 'Direct joint drive test; not a RoboTwin expert demonstration.'}
(out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
assert result['joint_tracking_pass'], result
