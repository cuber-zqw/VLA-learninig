"""Download verified, pinned Aloha assets and local CUDA 12.8 compiler components.
Only Aloha-AgileX is extracted; no task object dataset or model weights.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/downloads'
CACHE.mkdir(parents=True, exist_ok=True)
CUDA = ROOT / 'tools/cuda-12.8'
CUDA.mkdir(parents=True, exist_ok=True)
REDIST = 'https://developer.download.nvidia.com/compute/cuda/redist/'


def download_verified(url, destination, checksum):
    if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest() != checksum:
        temporary = destination.with_suffix(destination.suffix + '.partial')
        print('Downloading', url, flush=True)
        urllib.request.urlretrieve(url, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f'SHA256 mismatch: {url}')
        temporary.replace(destination)
    print('Verified', destination.name, flush=True)
    return destination


def main():
    asset = json.loads((ROOT / '2026-10-20/assets.json').read_text())
    archive = download_verified(asset['url'], CACHE / 'embodiments.zip', asset['sha256'])
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.startswith(asset['selected']):
                if '..' in Path(name).parts or Path(name).is_absolute():
                    raise ValueError(name)
                z.extract(name, ROOT / 'RoboTwin/assets')
    manifest = json.loads((ROOT / '2026-10-20/cuda-redist.json').read_text())
    def fetch(item):
        return download_verified(REDIST + item['relative_path'], CACHE / Path(item['relative_path']).name, item['sha256'])
    with ThreadPoolExecutor(max_workers=3) as ex:
        archives = list(ex.map(fetch, manifest.values()))
    for archive in archives:
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                name = Path(*Path(member.name).parts[1:])
                if str(name) == '.':
                    continue
                if name.is_absolute() or '..' in name.parts:
                    raise ValueError(name)
                member.name = str(name)
                tf.extract(member, CUDA)
    print('Prepared Aloha assets and local CUDA compiler.')


if __name__ == '__main__':
    main()
