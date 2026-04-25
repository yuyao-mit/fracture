import time, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
print('[t+0.0] import utils', flush=True); t0 = time.time()
from utils import load_config, load_paths, run_identity_from_cfg
print(f'[t+{time.time()-t0:.1f}] import data', flush=True)
from data import resolve_split, build_datasets_for_split
print(f'[t+{time.time()-t0:.1f}] load cfg', flush=True)
cfg = load_config('configs/experiments/id/fno.yaml', overrides=['wandb.mode=offline'])
paths = load_paths(cfg)
ident = run_identity_from_cfg(cfg)
print(f'[t+{time.time()-t0:.1f}] resolve split', flush=True)
split = resolve_split(ident.split, paths.data_root, seed=0)
print(f'[t+{time.time()-t0:.1f}] build train', flush=True)
ds = build_datasets_for_split(split, 4, 1, True, roles=('train',))
n = len(ds.get('train', []))
print(f'[t+{time.time()-t0:.1f}] train built: {n} samples', flush=True)
if n > 0:
    s = ds['train'][0]
    print(f'[t+{time.time()-t0:.1f}] first sample: '
          f'input_states={tuple(s["input_states"].shape)} '
          f'target={tuple(s["target"].shape)}', flush=True)
