"""Diagnostic: keep a frozen TLT actor, use only its inherited critic for search.

This isolates evaluation-time sequence ranking. It is not the primary HiLTL
protocol and must never silently replace it in a performance comparison.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
import evaluate_tlt
from envs.tlt_context import preprocess_tlt

class BaseCriticSearch(evaluate_tlt.StrictSearch):
    def get_value(self,seq,obs):
        if seq is None or not len(seq):return float('-inf')
        batch=preprocess_tlt([dict(obs,goal=seq)],self.propositions,
            device=next(self.model.parameters()).device)
        with torch.no_grad():_,value=self.model.base(batch)
        if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite base search value')
        return value.item()

def main():
    # Delegate the exact episode loop, seeds, action mode and outputs.
    args=list(sys.argv[1:])
    if '--output' not in args:raise ValueError('--output required')
    output=Path(args[args.index('--output')+1])
    evaluate_tlt.StrictSearch=BaseCriticSearch
    evaluate_tlt.main()
    path=output/'summary.json';summary=json.loads(path.read_text())
    summary.update(search_value_override='inherited_base_critic_only',
        diagnostic_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Diagnostic only: unchanged frozen TLT actor; inherited base critic ranks reach-avoid sequences. Not the primary protocol. No training or checkpoint modification.')
    path.write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
