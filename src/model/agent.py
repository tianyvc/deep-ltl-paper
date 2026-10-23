import numpy as np
import torch

import preprocessing
from torch.nn import Module
from sequence.search import SequenceSearch


class Agent:
    def __init__(self, model: Module, search: SequenceSearch, propositions: set[str], verbose=False):
        self.model = model
        self.search = search
        self.propositions = propositions
        self.verbose = verbose
        self.sequence = None

    def reset(self):
        self.sequence = None

    def get_action(self, obs, info, deterministic=False) -> np.ndarray:
        if 'ldba_state_changed' in info or self.sequence is None:
            self.sequence = self.search(obs['ldba'], obs['ldba_state'], obs)
            if self.verbose:
                print(f'Selected sequence: {self.sequence}')
        assert self.sequence is not None
        obs['goal'] = self.sequence
        return self.forward(obs, deterministic)

    def forward(self, obs, deterministic=False) -> np.ndarray:
        if not (isinstance(obs, list) or isinstance(obs, tuple)):
            obs = [obs]
        # Ensure tensors are created on the same device as the model
        device = next(self.model.parameters()).device
        preprocessed = preprocessing.preprocess_obss(obs, self.propositions, device=device)
        with torch.no_grad():
            dist, value = self.model(preprocessed)
            action = dist.mode if deterministic else dist.sample()
        # Ensure action moved to CPU before converting to numpy
        return action.detach().cpu().numpy()
