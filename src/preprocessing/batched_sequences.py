from dataclasses import dataclass

import torch
import numpy as np

ReachAvoidSet = tuple[list[int], list[int]]


@dataclass
class BatchedSequences:
    lens: torch.tensor  # (batch_size,) -> contains the length of each sequence
    data: torch.tensor  # (batch_size, max_len, max_set_size) -> contains the assignment ids
    device: str

    @classmethod
    def from_seqs(cls, seqs: list[list[list[int]]], device=None):
        lens = torch.tensor([len(seq) for seq in seqs], dtype=torch.long)  # needs to be on CPU
        batch_size = len(seqs)
        max_len = lens.max().item() if lens.numel() > 0 and lens.max().item() > 0 else 1
        max_set_size = max((len(a) for seq in seqs for a in seq), default=1)
        data = torch.zeros((batch_size, max_len, max_set_size), dtype=torch.long).to(device)
        for i, seq in enumerate(seqs):
            for j, a in enumerate(seq):
                data[i, j, :len(a)] = torch.tensor(a, dtype=torch.long)
        return cls(
            lens,
            data.to(device),
            device
        )

    def __getitem__(self, index) -> tuple[torch.tensor, torch.tensor]:
        return self.lens[index], self.data[index]

    def all(self):
        return self.lens, self.data


class BatchedReachAvoidSequences:
    def __init__(self, seqs: list[list[ReachAvoidSet]], device=None, ldba_sequences=None):
        """
        Args:
            seqs: list of reach-avoid sequences
            device: torch device
            ldba_sequences: optional list of原始LDBASequence对象，用于提取状态和转移信息
        """
        self.device = device
        self.reach_seqs, self.avoid_seqs = self.batch(seqs, device)
        self.sequences = ldba_sequences  # 保留原始序列以提取自动机信息

    @staticmethod
    def batch(seqs: list[list[ReachAvoidSet]], device=None) -> tuple[BatchedSequences, BatchedSequences]:
        reach_seqs = []
        avoid_seqs = []
        for seq in seqs:
            reach_seq, avoid_seq = BatchedReachAvoidSequences.split_seq(seq)
            reach_seqs.append(reach_seq)
            avoid_seqs.append(avoid_seq)
        reach_seqs = BatchedSequences.from_seqs(reach_seqs, device)
        avoid_seqs = BatchedSequences.from_seqs(avoid_seqs, device)
        return reach_seqs, avoid_seqs

    @staticmethod
    def split_seq(seq: list[ReachAvoidSet]) -> tuple[list[list[int]], list[list[int]]]:
        reach_seq = []
        avoid_seq = []
        for reach, avoid in seq:
            reach_seq.append(reach if isinstance(reach, list) else list(reach))
            avoid_seq.append(avoid if isinstance(avoid, list) else list(avoid))
        # 如果序列为空，返回包含一个空列表的列表（保持维度一致）
        if not reach_seq:
            reach_seq = [[]]
            avoid_seq = [[]]
        return reach_seq, avoid_seq

    def __getitem__(self, index: np.ndarray):
        """
        Returns a sub-batch of the given sequences.
        """
        return self.reach_seqs[index], self.avoid_seqs[index]

    def all(self):
        return self.reach_seqs.all(), self.avoid_seqs.all()
