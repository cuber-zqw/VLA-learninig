"""Verify real CUDA kernels and a backward pass on the current GPU."""
import argparse
import json
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('2026-10-20/torch-gpu-result.json'))
args = parser.parse_args()
assert torch.cuda.is_available()
torch.manual_seed(0)
a_cpu = torch.randn(256, 256)
b_cpu = torch.randn(256, 256)
a = a_cpu.cuda().requires_grad_()
b = b_cpu.cuda()
y = a @ b
reference = a_cpu @ b_cpu
assert torch.allclose(y.cpu(), reference, atol=1e-3, rtol=1e-3)
y.square().mean().backward()
assert torch.isfinite(a.grad).all()
torch.cuda.synchronize()
result = {'cuda_compute_pass': True, 'backward_pass': True,
          'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
          'device': torch.cuda.get_device_name(0),
          'compute_capability': list(torch.cuda.get_device_capability(0)),
          'compiled_arches': torch.cuda.get_arch_list(),
          'max_abs_matmul_error': float((y.cpu() - reference).abs().max())}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
