"""Bounded RoboTwin expert planning + trajectory replay and RGB/state collection.
Run with .venv-robotwin after sourcing scripts/robotwin_env.sh.
Uses the upstream task, planner, controller, success check and data writer.
"""
import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=ROOT / '2026-10-20/expert')
parser.add_argument('--start-seed', type=int, default=0)
parser.add_argument('--max-attempts', type=int, default=5)
args = parser.parse_args()
out = args.output_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
os.chdir(ROOT / 'RoboTwin')
sys.path.insert(0, str(ROOT / 'RoboTwin'))
import yaml
import torch
import torch.multiprocessing as mp


def main():
    mp.set_start_method('spawn', force=True)
    task_name = 'stack_blocks_two'
    cfg = yaml.safe_load(Path('task_config/demo_clean.yml').read_text())
    robot_dir = './assets/embodiments/aloha-agilex/'
    embodiment = yaml.safe_load(Path(robot_dir, 'config.yml').read_text())
    for template in Path(robot_dir).glob('*_tmp.yml'):
        template.with_name(template.name.replace('_tmp.yml', '.yml')).write_text(
            template.read_text().replace('${ASSETS_PATH}', str(ROOT / 'RoboTwin'))
        )
    cfg.update(task_name=task_name, task_config='single_clean', episode_num=1,
               save_path=str(out), render_freq=0, left_robot_file=robot_dir,
               right_robot_file=robot_dir, dual_arm_embodied=True,
               left_embodiment_config=deepcopy(embodiment),
               right_embodiment_config=deepcopy(embodiment),
               embodiment_name='aloha-agilex', save_data=False, need_plan=True)
    (out / 'config.yml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    result = {'task': task_name, 'expert_task_pass': False, 'attempts': []}
    result_path = out / 'result.json'
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    cls = getattr(importlib.import_module('envs.' + task_name), task_name)
    class TaskWithCompatibleRenderer(cls):
        def setup_scene(self, **kwargs):
            super().setup_scene(**kwargs)
            # The bundled OIDN reports unsupported CUDA device on this setup.
            # Preserve ray tracing and sample count; disable only denoising.
            import sapien
            sapien.render.set_ray_tracing_denoiser('none')

    cfg['local_render_denoiser'] = 'none'
    (out / 'config.yml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    task = TaskWithCompatibleRenderer()
    try:
        successful_seed = None
        for seed in range(args.start_seed, args.start_seed + args.max_attempts):
            print(f'=== Planning seed {seed} ===', flush=True)
            task.setup_demo(now_ep_num=0, seed=seed, **cfg)
            task.play_once()
            success = bool(task.plan_success and task.check_success())
            record = {'seed': seed, 'plan_success': bool(task.plan_success), 'success': success}
            result['attempts'].append(record)
            result_path.write_text(json.dumps(result, indent=2) + '\n')
            print(record, flush=True)
            if success:
                task.save_traj_data(0)
                successful_seed = seed
                task.close_env()
                break
            task.close_env()
        if successful_seed is None:
            raise RuntimeError(f'No successful plan in {args.max_attempts} seeds')
        (out / 'seed.txt').write_text(str(successful_seed) + '\n')
        print(f'=== Collect replay seed {successful_seed} ===', flush=True)
        cfg.update(need_plan=False, save_data=True)
        task.setup_demo(now_ep_num=0, seed=successful_seed, **cfg)
        traj = task.load_tran_data(0)
        cfg.update(traj)
        task.set_path_lst(cfg)
        info = task.play_once()
        success = bool(task.plan_success and task.check_success())
        result.update(seed=successful_seed, replay_success=success,
                      frames=task.FRAME_IDX,
                      block1_position=task.block1.get_pose().p.tolist(),
                      block2_position=task.block2.get_pose().p.tolist())
        (out / 'scene_info.json').write_text(json.dumps(info, indent=2) + '\n')
        if not success:
            raise RuntimeError('Replay failed task success check')
        task.save_camera_rgb(str(out / 'final.png'))
        task.merge_pkl_to_hdf5_video()
        result.update(expert_task_pass=True, hdf5='data/episode0.hdf5', video='video/episode0.mp4')
        result_path.write_text(json.dumps(result, indent=2) + '\n')
        task.remove_data_cache()
        print(json.dumps(result, indent=2), flush=True)
    except Exception:
        result['error'] = traceback.format_exc()
        result_path.write_text(json.dumps(result, indent=2) + '\n')
        raise
    finally:
        if hasattr(task, 'scene'):
            task.close_env()
        # Separate embodiment planners use child processes; stop on completion/errors.
        robot = getattr(task, 'robot', None)
        if robot is not None:
            for name in ['left_proc', 'right_proc']:
                proc = getattr(robot, name, None)
                if proc is not None and proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)


if __name__ == '__main__':
    main()
