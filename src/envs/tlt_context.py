"""Observation-only context for the original DeepLTL sequence curriculum.

No reward, termination, sampling, action or TimeLimit changes are made here.
Curriculum graphs are exact monitors of the sampled reach/avoid sequence, not
claimed to be the full automaton of an unrelated evaluation formula.
"""
from dataclasses import dataclass
from functools import lru_cache
import gymnasium as gym
import numpy as np
from ltl.automata import LDBA, LDBASequence


@dataclass(frozen=True)
class TLTContext:
    ldba: object
    state: int
    history: tuple


class ContextBatch:
    def __init__(self, contexts): self.contexts = tuple(contexts)
    def __len__(self): return len(self.contexts)
    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)): return self.contexts[int(index)]
        if isinstance(index, slice): return ContextBatch(self.contexts[index])
        return ContextBatch([self.contexts[int(i)] for i in index])


def assignment_label(assignments):
    return ' | '.join('('+a.to_label()+')' for a in sorted(assignments, key=lambda a:a.to_label()))


@lru_cache(maxsize=256)
def sequence_monitor(sequence, propositions, assignments):
    n = len(sequence)
    if not n: raise ValueError('Cannot monitor an empty curriculum sequence')
    graph=LDBA(set(propositions), simplify_labels=False)
    for state in range(n+2): graph.add_state(state, initial=state==0)
    sink=n+1
    universe=set(assignments)
    for i,(reach,avoid) in enumerate(sequence):
        avoid=set(avoid)&universe
        effective_reach=set() if reach==LDBASequence.EPSILON else (set(reach)&universe)-avoid
        stays=universe-avoid-effective_reach
        if reach==LDBASequence.EPSILON: graph.add_transition(i,i+1,None,i+1==n)
        elif effective_reach: graph.add_transition(i,i+1,assignment_label(effective_reach),i+1==n)
        if avoid: graph.add_transition(i,sink,assignment_label(avoid),False)
        if stays: graph.add_transition(i,i,assignment_label(stays),False)
    graph.add_transition(n,n,'t',True)
    graph.add_transition(sink,sink,'t',False)
    graph.sink_state=sink
    graph.compute_sccs()
    return graph


class CurriculumContext(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.props=tuple(sorted(env.get_propositions()))
        self.assignments=tuple(sorted((a.to_frozen() for a in env.get_possible_assignments()),key=lambda a:a.to_label()))
        self.completed=[]
        self.episode_results=[]
        self.trace=[]
        self.history=()
        self.last=None

    def snapshot(self, observation):
        obs=dict(observation,features=np.array(observation['features'],copy=True))
        initial=obs['initial_goal']
        current=len(initial)-len(obs['goal'])
        self.history=self.history+(current,)
        graph=sequence_monitor(initial,self.props,self.assignments)
        obs['tlt_context']=TLTContext(graph,current,self.history)
        return obs

    def reset(self, **kwargs):
        self.history=()
        self.trace=[]
        self.last=self.snapshot(self.env.reset(**kwargs))
        return self.last

    def step(self, action):
        obs,reward,done,info=self.env.step(np.array(action,copy=True))
        self.trace.append((self.last,float(reward)))
        if done:
            self.completed.append(tuple(self.trace))
            self.episode_results.append(dict(success=bool(info.get('success')),violation=bool(info.get('violation')),steps=len(self.trace),reward=sum(r for _,r in self.trace)))
        self.last=self.snapshot(obs)
        return self.last,reward,done,info

    def drain_completed(self):
        episodes=self.completed
        self.completed=[]
        self.episode_results=[]
        return episodes


def preprocess_tlt(observations, propositions, device=None):
    from preprocessing import preprocess_obss
    result=preprocess_obss(observations,propositions,device=device)
    result.context=ContextBatch([obs['tlt_context'] for obs in observations])
    return result
