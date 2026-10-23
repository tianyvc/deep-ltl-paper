"""TLT long-manuscript modules attached to the original sequence policy.

The base actor/DeepSets/GRU/environment encoder retain their original shapes.
Graph conditioning enters through a zero-initialized residual, so transferring
an original checkpoint does not replace its trained policy with random actions.
Long head follows the manuscript's q/history-only MC-return surrogate; it is
not a calibrated satisfaction probability. No artificial skip edges are added.
"""
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_sequence
from torch_geometric.nn import GATv2Conv
from model.deepltl_reference import DeepLTLSequenceModel
from model.automaton_graph import AutomatonGraph

CONFIGURATIONS={'baseline':(False,False,False),'hv_only':(True,False,False),
                'msgnn_only':(False,True,False),'hv_track':(True,False,True),
                'msgnn_track':(False,True,True),'full_no_options':(True,True,True)}
PROTOCOL='tlt-upstream-curriculum-v1'


class TLTSequenceModel(nn.Module):
    recurrent=False
    def __init__(self,env,model_config,configuration='hv_track'):
        super().__init__()
        self.configuration=configuration
        self.hv,self.multiscale,self.tracking=CONFIGURATIONS[configuration]
        self.base=DeepLTLSequenceModel(env,model_config)
        self.propositions=tuple(sorted(env.get_propositions()))
        self._graph_cache=OrderedDict();self._encoded_cache={};self._history_cache=OrderedDict()
        if configuration=='baseline': return
        hidden=64
        self.token=nn.Embedding(8+len(self.propositions),hidden)
        self.guard=nn.GRU(hidden,hidden,batch_first=True)
        self.node=nn.Linear(4,hidden)
        self.edge=nn.Linear(6+hidden,hidden)
        self.convs=nn.ModuleList([GATv2Conv(hidden,hidden,heads=4,concat=False,edge_dim=hidden,add_self_loops=False) for _ in range(3)])
        self.transforms=nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden),nn.LeakyReLU(.2)) for _ in range(3)])
        self.norms=nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])
        self.pools=nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1)) for _ in range(3)])
        self.history=nn.LSTM(hidden,hidden,batch_first=True) if self.tracking else None
        graph_size=hidden*(3 if self.multiscale else 1)+hidden
        env_size=self.base.env_net.embedding_size if self.base.env_net is not None else env.observation_space['features'].shape[0]
        base_size=env_size+self.base.ltl_net.embedding_dim
        self.actor_residual=nn.Linear(graph_size,base_size)
        self.short=nn.Sequential(nn.Linear(env_size+graph_size,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1))
        self.long=nn.Sequential(nn.Linear(1+hidden+(hidden if self.tracking else 0),128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1)) if self.hv else None
        # Explicit compatibility initialization, not an additional reward/algorithm.
        nn.init.zeros_(self.actor_residual.weight);nn.init.zeros_(self.actor_residual.bias)
        nn.init.zeros_(self.short[-1].weight);nn.init.zeros_(self.short[-1].bias)

    @property
    def ltl_net(self): return self.base.ltl_net

    def invalidate_cache(self):
        self._encoded_cache.clear();self._history_cache.clear()

    def graph(self,ldba):
        if tuple(ldba.propositions)!=self.propositions:raise ValueError('Proposition vocabulary changed')
        if ldba not in self._graph_cache:
            self._graph_cache[ldba]=AutomatonGraph.from_ldba(ldba)
            if len(self._graph_cache)>256:self._graph_cache.popitem(last=False)
        return self._graph_cache[ldba]

    def encode_graph(self,ldba):
        if not torch.is_grad_enabled() and ldba in self._encoded_cache:return self._encoded_cache[ldba]
        graph=self.graph(ldba);device=next(self.parameters()).device
        tokens=[torch.tensor(t,device=device,dtype=torch.long) for t in graph.guard_tokens]
        packed=pack_padded_sequence(pad_sequence([self.token(t) for t in tokens],batch_first=True),[len(t) for t in tokens],batch_first=True,enforce_sorted=False)
        _,guard=self.guard(packed)
        edges=self.edge(torch.cat([graph.edge_features.to(device),guard[-1]],dim=-1))
        nodes=self.node(graph.nodes.to(device));layers=[]
        for conv,mlp,norm,pool in zip(self.convs,self.transforms,self.norms,self.pools):
            nodes=norm(nodes+mlp(conv(nodes,graph.edges.to(device),edges)))
            layers.append((torch.softmax(pool(nodes),dim=0)*nodes).sum(0))
        encoded=(nodes,torch.cat(layers) if self.multiscale else layers[-1],graph)
        if not torch.is_grad_enabled():
            if len(self._encoded_cache)>=256:self._encoded_cache.clear()
            self._encoded_cache[ldba]=encoded
        return encoded

    def representations(self,contexts):
        encoded={graph:self.encode_graph(graph) for graph in dict.fromkeys(c.ldba for c in contexts)}
        current=[];pooled=[];proximities=[];histories=[]
        for c in contexts:
            nodes,graph_pool,graph=encoded[c.ldba];mapping={s:i for i,s in enumerate(graph.state_ids)}
            current.append(nodes[mapping[c.state]]);pooled.append(graph_pool)
            proximities.append(1-graph.nodes[mapping[c.state],3].item())
            if self.tracking:histories.append(nodes[[mapping[q] for q in c.history]])
        history=None
        if self.tracking:
            if torch.is_grad_enabled():
                packed=pack_padded_sequence(pad_sequence(histories,batch_first=True),[len(h) for h in histories],batch_first=True,enforce_sorted=False)
                _,(last,_)=self.history(packed);history=last[-1]
            else:
                # Cached recurrent states are inference-only and invalidated on every update.
                outputs=[]
                for context,sequence in zip(contexts,histories):
                    key=(context.ldba,context.history)
                    if key not in self._history_cache:
                        prefix=(context.ldba,context.history[:-1])
                        previous=self._history_cache.get(prefix)
                        inputs=sequence[-1:].unsqueeze(0) if previous is not None else sequence.unsqueeze(0)
                        _,state=self.history(inputs,previous)
                        self._history_cache[key]=state
                        if len(self._history_cache)>512:self._history_cache.popitem(last=False)
                    outputs.append(self._history_cache[key][0][-1,0])
                history=torch.stack(outputs)
        return torch.stack(current),torch.stack(pooled),history,proximities

    def components(self,obs):
        environment=self.base.env_net(obs.features) if self.base.env_net is not None else obs.features
        embedding=torch.cat([environment,self.base.ltl_net(obs.seq)],dim=-1)
        base_value=self.base.critic(embedding).squeeze(-1)
        if self.configuration=='baseline':
            dist=self.base.actor(embedding);dist.set_epsilon_mask(obs.epsilon_mask)
            return dist,dict(total=base_value,short=base_value,long=torch.zeros_like(base_value),beta=torch.zeros_like(base_value))
        current,pooled,history,proximities=self.representations(obs.context.contexts)
        graph=torch.cat([pooled,current],dim=-1)
        dist=self.base.actor(embedding+self.actor_residual(graph));dist.set_epsilon_mask(obs.epsilon_mask)
        short=base_value+self.short(torch.cat([environment,graph],dim=-1)).squeeze(-1)
        proximity=torch.tensor(proximities,device=embedding.device,dtype=embedding.dtype).unsqueeze(-1)
        beta=torch.sigmoid(3*(proximity[:,0]-.5))
        if self.hv:
            data=[proximity,current]+([history] if self.tracking else [])
            long=self.long(torch.cat(data,dim=-1)).squeeze(-1)
            total=(1-beta)*short+beta*long
        else:
            long=torch.zeros_like(short);total=short
            if self.tracking:
                # Tracking-only variants must actually condition a trainable output.
                # History affects graph conditioning via a residual of fixed dimension.
                dist=self.base.actor(embedding+self.actor_residual(graph+torch.cat([torch.zeros_like(pooled),history],dim=-1)))
                dist.set_epsilon_mask(obs.epsilon_mask)
        return dist,dict(total=total,short=short,long=long,beta=beta)

    def forward(self,obs):
        dist,parts=self.components(obs)
        return dist,parts['total']
