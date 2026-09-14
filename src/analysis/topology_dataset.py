"""Prepared relaxed-target inputs for the standard checkpoint analysis pipeline."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data_utils.relaxed_histories import RelaxedHistoryDataset
from src.data.trajectories.shooting import ShootingBinaryTrajectory


class RelaxedTopologyAnalysisDataset(Dataset):
    """Real model histories, anchor structures for plots, and physical MD centers."""

    def __init__(self, cfg, split, intervention='real'):
        self.base = RelaxedHistoryDataset(cfg, split)
        self.intervention = intervention
        self.indices = self.base.indices
        self.contexts = self.base.contexts[self.indices]
        self.sample_source_names = [f"source_{self.base.records[c]['source']:03d}_frame_{self.base.records[c]['frame']}"
                                    for c in self.contexts]
        original = json.loads(Path(cfg.data.relaxed_manifest).read_text())['shards']
        coordinates = []
        self.atom_ids = []
        trajectory = None
        for context in np.unique(self.contexts):
            record = original[int(context)]
            source = record['provenance']['source']
            if trajectory is None or trajectory.root != Path(source['path']):
                trajectory = ShootingBinaryTrajectory.load(source['path'])
            centers = np.load(Path(record['directory'])/'centers.npy')
            coordinates.append(trajectory.positions[record['frame'], centers].astype(np.float32))
            self.atom_ids.append(trajectory.atom_ids[centers])
        self.coordinates = np.concatenate(coordinates)
        self.atom_ids = np.concatenate(self.atom_ids)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        model_input = item['points']
        anchor = model_input[-1] if self.base.input_mode == 'history' else model_input
        if self.intervention == 'repeat_anchor':
            model_input = model_input[-1:].expand_as(model_input)
        elif self.intervention == 'reverse_past':
            model_input = torch.cat((model_input[:-1].flip(0), model_input[-1:]), dim=0)
        elif self.intervention != 'real':
            raise ValueError(f'Unknown history intervention: {self.intervention}')
        return dict(points=anchor, model_input=model_input,
                    coords=torch.from_numpy(self.coordinates[index]),
                    instance_id=int(self.atom_ids[index]), row=int(item['row']),
                    anchor_frame_index=int(item['frame']))


def topology_dataloader(dataset, batch_size, num_workers):
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      shuffle=False, drop_last=False, pin_memory=True)


class RelaxedTopologyAnalysisDataModule:
    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self, stage=None):
        # The main clustering/structure analysis uses the held-out cohort.
        # Topology probes separately load their training and validation splits.
        self.test_dataset = RelaxedTopologyAnalysisDataset(self.cfg, 'test')
        self.train_dataset = self.test_dataset
