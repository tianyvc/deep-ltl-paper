from typing import Any

import torch

import torch_ac
import numpy as np

from ltl.automata import LDBASequence
from ltl.logic import FrozenAssignment, Assignment
from preprocessing.vocab import VOCAB
from preprocessing.batched_sequences import BatchedReachAvoidSequences, ReachAvoidSet


def preprocess_obss(obss: list[dict[str, Any]], propositions: set[str], device=None) -> torch_ac.DictList:
    features = []
    seqs = []
    original_seqs = []  # 保留原始 LDBASequence 对象
    epsilon_mask = []
    for obs in obss:
        features.append(obs["features"])
        original_seqs.append(obs["goal"])  # 保留原始 LDBASequence
        seqs.append(list(reversed(obs["goal"])))  # 用于预处理的反转列表
    for seq, obs in zip(seqs, obss):
        # 处理空序列的情况（无限时域搜索可能失败）
        if not seq:
            epsilon_mask.append(False)
            continue
        epsilon_enabled = seq[-1][0] == LDBASequence.EPSILON
        if epsilon_enabled and len(seq) > 1:
            next_avoid = seq[-2][1]
            assignment = Assignment({p: (p in obs['propositions']) for p in propositions}).to_frozen()
            epsilon_enabled &= assignment not in next_avoid
        epsilon_mask.append(epsilon_enabled)
    
    # 保留原始 LDBASequence 对象以供 GNN 增强功能使用
    return torch_ac.DictList({
        "features": preprocess_features(features, device=device),
        "seq": BatchedReachAvoidSequences(
            [preprocess_sequence(seq) for seq in seqs], 
            device=device,
            ldba_sequences=original_seqs  # 传递原始 LDBASequence 对象
        ),
        "epsilon_mask": torch.tensor(epsilon_mask, dtype=torch.bool).to(device),
    })


def preprocess_features(features, device=None) -> torch.tensor:
    return torch.tensor(np.array(features), dtype=torch.float).to(device)


def preprocess_sequence(seq: LDBASequence) -> list[ReachAvoidSet]:
    if not seq:  # 处理空序列
        return []
    return [(preprocess_assignments(a), preprocess_assignments(b)) for a, b in seq]


def preprocess_assignments(assignments: frozenset[FrozenAssignment] | type(LDBASequence.EPSILON)) -> list[int]:
    if assignments == LDBASequence.EPSILON:
        return [VOCAB['EPSILON']]
    if len(assignments) == 0:
        return [VOCAB['NULL']]
    return [VOCAB[a] for a in assignments]
