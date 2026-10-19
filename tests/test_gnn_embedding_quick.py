import torch
import torch.nn as nn
import random

from preprocessing.batched_sequences import BatchedReachAvoidSequences
from model.gnn_net import GNNNet
from model.model import Model

# 轻量的 FakeLTLNet，用于测试 compute_embedding
class FakeLTLNet(nn.Module):
    def __init__(self, vocab_size=50, token_emb_dim=16, set_emb_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, token_emb_dim, padding_idx=0)
        # set_network: 将 (batch, max_len, max_set_size, token_emb_dim) -> (batch, max_len, set_emb_dim)
        # 我们先对 max_set_size 维度做 mean，再用线性层
        self.set_network = nn.Sequential(
            nn.Linear(token_emb_dim, set_emb_dim),
            nn.ReLU()
        )
        # LTLNet 的 embedding_dim（最终序列级别表示维度）
        self.embedding_dim = set_emb_dim * 2

    def forward(self, batched_seqs):
        # 接受 BatchedReachAvoidSequences 或 (reach, avoid)
        if hasattr(batched_seqs, 'all'):
            (reach_lens, reach_data), (avoid_lens, avoid_data) = batched_seqs.all()
        else:
            (reach_lens, reach_data), (avoid_lens, avoid_data) = batched_seqs
        # reach_data: (batch, max_len, max_set_size)
        # 先对 max_set_size 做 embedding 并取 mean
        # 转换为 long
        reach_data = reach_data.long()
        avoid_data = avoid_data.long()
        # embed -> (batch, max_len, max_set_size, token_emb_dim)
        reach_emb = self.embedding(reach_data)
        avoid_emb = self.embedding(avoid_data)
        # 对集合维度求平均 (注意可能有 PAD=0 的情况，但这里简单均值)
        reach_mean = reach_emb.mean(dim=2)
        avoid_mean = avoid_emb.mean(dim=2)
        # 通过 set_network
        reach_pos = self.set_network(reach_mean)
        avoid_pos = self.set_network(avoid_mean)
        # 拼接
        pos = torch.cat([reach_pos, avoid_pos], dim=-1)  # (batch, max_len, set_emb_dim*2)
        batch_size = pos.size(0)
        max_len = pos.size(1)
        # 根据 lens 做 mask 平均得到序列级表示
        device = pos.device
        outputs = []
        for i in range(batch_size):
            L = reach_lens[i].item()
            if L == 0:
                outputs.append(torch.zeros(self.embedding_dim, device=device))
            else:
                seq_i = pos[i, :L, :]
                outputs.append(seq_i.mean(dim=0))
        return torch.stack(outputs, dim=0)  # (batch, embedding_dim)


def build_minimal_model():
    # 构建 fake ltl_net 和 gnn_net
    fake_ltl = FakeLTLNet(vocab_size=100, token_emb_dim=16, set_emb_dim=16)
    gnn = GNNNet(input_dim=32, hidden_dim=32, num_layers=1)
    # actor/critic 占位
    actor = nn.Identity()
    critic = nn.Identity()
    model = Model(actor=actor, critic=critic, ltl_net=fake_ltl, gnn_net=gnn, env_net=None)
    return model


def make_dummy_batched_seq(batch_size=3, max_len=4, vocab_size=20):
    seqs = []
    for _ in range(batch_size):
        length = random.randint(1, max_len)
        seq = []
        for _ in range(length):
            # 每个集合包含1~2个 token id
            reach = [random.randint(1, vocab_size - 1) for _ in range(random.randint(1, 2))]
            avoid = [random.randint(1, vocab_size - 1) for _ in range(random.randint(1, 2))]
            seq.append((reach, avoid))
        seqs.append(seq)
    return BatchedReachAvoidSequences(seqs, device=None)


if __name__ == '__main__':
    torch.manual_seed(1)
    model = build_minimal_model()
    batched_seq = make_dummy_batched_seq(batch_size=4, max_len=6)

    class ObsBatch:
        pass

    obs = ObsBatch()
    obs.features = torch.randn(4, 8)
    obs.seq = batched_seq
    obs.epsilon_mask = torch.zeros(4, dtype=torch.bool)

    emb = model.compute_embedding(obs)
    print('embedding shape:', emb.shape)
