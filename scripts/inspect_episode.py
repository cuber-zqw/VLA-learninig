"""Inspect a generated RoboTwin HDF5 and decode the first RGB record."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('episode', type=Path)
parser.add_argument('--output-dir', type=Path)
parser.add_argument('--grid-video', action='store_true', help='Export a labeled four-camera video at the upstream 30 fps.')
args = parser.parse_args()
out = (args.output_dir or args.episode.parent.parent / 'inspection').resolve()
out.mkdir(parents=True, exist_ok=True)
report = {'file': str(args.episode.resolve()), 'datasets': {}, 'images': {}}
with h5py.File(args.episode, 'r') as f:
    def inspect(name, item):
        if not isinstance(item, h5py.Dataset):
            return
        entry = {'shape': list(item.shape), 'dtype': str(item.dtype)}
        if item.size and item.dtype.kind in 'fiu':
            data = item[...]
            entry.update(finite=bool(np.isfinite(data).all()), min=float(data.min()), max=float(data.max()))
            assert entry['finite'], name
        report['datasets'][name] = entry
        if name.endswith('/rgb'):
            assert len(item) > 1, name
            for idx, tag in [(0, 'first'), (len(item) - 1, 'last')]:
                decoded = cv2.imdecode(np.frombuffer(item[idx], dtype=np.uint8), cv2.IMREAD_COLOR)
                assert decoded is not None, name
                target = out / (name.replace('/', '_') + '_' + tag + '.png')
                # Upstream feeds RGB directly to cv2.imencode; keep the decoded array
                # unchanged to recover the original channel ordering for PIL/imageio.
                imageio.imwrite(target, decoded)
            report['images'][name] = {'records': len(item), 'decoded_shape': list(decoded.shape)}
    f.visititems(inspect)
    targets = f['joint_action/vector'][...]
    assert targets.ndim == 2 and targets.shape[1] == 14, targets.shape
    assert len(targets) > 1
    assert max(np.ptp(targets, axis=0)) > 0.01
    for name, value in report['images'].items():
        assert value['records'] == len(targets), name
    if args.grid_video:
        cameras = ['head_camera', 'front_camera', 'left_camera', 'right_camera']
        datasets = [f['observation/' + name + '/rgb'] for name in cameras]
        shape = report['images']['observation/head_camera/rgb']['decoded_shape']
        height, width = shape[:2]
        video = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{2 * width}x{2 * height}', '-r', '30', '-i', '-',
            '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out / 'four_cameras.mp4')], stdin=subprocess.PIPE)
        try:
            for idx in range(len(targets)):
                panels = []
                for name, dataset in zip(cameras, datasets):
                    panel = cv2.imdecode(np.frombuffer(dataset[idx], dtype=np.uint8), cv2.IMREAD_COLOR)
                    assert panel.shape == (height, width, 3), panel.shape
                    cv2.putText(panel, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(panel, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    panels.append(panel)
                grid = np.concatenate([np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)], axis=0)
                video.stdin.write(grid.tobytes())
        finally:
            video.stdin.close()
            if video.wait() != 0:
                raise RuntimeError('Grid video encoding failed')
        report['grid_video'] = 'four_cameras.mp4'
    report['frames'] = len(targets)
    report['joint_action_semantics'] = ('Left 6 joint drive targets [rad], commanded left gripper [0,1], '
        'right 6 joint drive targets [rad], commanded right gripper [0,1]. '
        'These arm values come from get_drive_target(), not measured articulation qpos.')
    report['timing_note'] = ('Simulation timestep is 1/250 s. save_freq=15 within each movement; '
        'movement boundaries add extra samples, so do not infer uniform timestamps from row indices. '
        'The upstream mp4 is encoded at 30 fps and is not a physical-time clock.')
(out / 'schema.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
