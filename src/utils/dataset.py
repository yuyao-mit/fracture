import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle


class ChunkedScalarDatasetEfficient(Dataset):
    """
    Dataset for Inference with Latent Physics framework.

    Each sample is a sliding window over a simulation trajectory:
      - input_states:   input_steps consecutive states  [input_steps,  C, W, H]
      - latent_vars:    latent physics variable at the last input step
      - rollout_states: rollout_steps future states starting right after the input window

    Args:
        folder (str): Path to folder containing .npz case files.
            Each file must contain:
                'inputs':           [T, C, W, H]  - state fields
                'targets':          [T, 1, W, H]  - next phase-field states
                'latent_variables': [T, ...]       - latent physics variables
                'lc':               [T]            - scalar length-scale per step
        input_steps (int): Number of history steps fed to the network.
        rollout_steps (int): Number of future steps used as rollout targets.
        infer_latent_physics (bool):
            - True:  __getitem__ returns latent_vars as the supervision target.
            - False: __getitem__ returns rollout_states as the supervision target.
        mode (str): Loading strategy ('mmap' recommended).
        samples_cache_file (str | None): Optional path to pickle cache for sample index.
    """

    def __init__(self, folder, input_steps=4, rollout_steps=1,
                 infer_latent_physics=True, mode='mmap',
                 cache_size=50, samples_cache_file=None):
        self.mode = mode
        self.cache_size = cache_size
        self.input_steps = input_steps
        self.rollout_steps = rollout_steps
        self.infer_latent_physics = infer_latent_physics
        self.samples = []   # list of (path, start_idx)
        self.file_map = {}  # path -> file index

        # Minimum trajectory length needed per sample
        self.min_len = input_steps + rollout_steps

        if samples_cache_file and os.path.exists(samples_cache_file):
            print(f"Loading samples from cache file: {samples_cache_file}")
            with open(samples_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.samples = cache_data['samples']
                self.file_map = cache_data['file_map']
            print(f"Loaded {len(self.samples)} samples from cache")
        else:
            print("Building sample list by scanning files...")
            file_list = sorted([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.endswith('.npz') and f.startswith('case')
            ])

            for file_idx, path in enumerate(file_list):
                data = np.load(path, mmap_mode='r')
                T = data['inputs'].shape[0]
                if T < self.min_len:
                    print(f"Warning: {path} has only {T} steps "
                          f"(need {self.min_len}), skipping.")
                    continue
                self.file_map[path] = file_idx
                # start_idx can range from 0 to T - min_len (inclusive)
                for start in range(T - self.min_len + 1):
                    self.samples.append((path, start))

            print(f"Found {len(self.samples)} samples from {len(self.file_map)} files")

            if samples_cache_file:
                print(f"Saving samples to cache file: {samples_cache_file}")
                os.makedirs(os.path.dirname(samples_cache_file), exist_ok=True)
                with open(samples_cache_file, 'wb') as f:
                    pickle.dump({'samples': self.samples, 'file_map': self.file_map}, f)
                print("Samples cache saved successfully")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            path, start = self.samples[idx]
            data = np.load(path, mmap_mode='r')

            end_input = start + self.input_steps          # exclusive
            end_rollout = end_input + self.rollout_steps  # exclusive

            # input_states: [input_steps, C, W, H]
            input_states = torch.from_numpy(
                data['inputs'][start:end_input].copy()
            ).float()

            # latent_vars at the last input step: [..., W, H] or [latent_dim]
            latent_vars = torch.from_numpy(
                data['latent_variables'][end_input - 1].copy()
            ).float()

            # rollout_states: [rollout_steps, 1, W, H]
            rollout_states = torch.from_numpy(
                data['targets'][end_input - 1:end_rollout - 1].copy()
            ).float()

            # lc at the last input step
            lc = torch.tensor(float(data['lc'][end_input - 1])).float()

            if self.infer_latent_physics:
                target = latent_vars
            else:
                target = rollout_states

            return {
                'input_states': input_states,   # [input_steps, C, W, H]
                'target': target,               # latent_vars or rollout_states
                'latent_vars': latent_vars,     # always provided for reference
                'rollout_states': rollout_states,
                'lc': lc,
            }

        except Exception as e:
            print(f"[Dataset Error] File: {self.samples[idx][0]} "
                  f"at start_idx {self.samples[idx][1]}, Error: {e}")
            raise


if __name__ == "__main__":
    dataset = ChunkedScalarDatasetEfficient(
        folder='/jet/home/ysunb/project/crack_ML/data/',
        input_steps=4,
        rollout_steps=2,
        infer_latent_physics=True,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    for batch in loader:
        print("input_states:", batch['input_states'].shape)
        print("target:      ", batch['target'].shape)
        break
