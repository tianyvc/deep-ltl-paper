"""Audit completed minimal-study artifacts and optionally export lightweight evidence.

Run on the original server in wty_LTL. Exported checkpoints are indexed, not copied.
"""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

IMPLEMENTATION = ['src/model/paper_hiltl.py', 'src/model/paper_options.py',
                  'src/model/deepltl_reference.py', 'src/train/train_paper.py',
                  'src/train/paper_tasks.py']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root, export=None):
    import torch
    from paper_campaign import validate
    root = root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    state = json.loads((root / 'state.json').read_text())
    assert state['state'] == 'completed'
    assert state['manifest_sha256'] == digest(root / 'manifest.json')
    assert len(state['tasks']) == len(manifest['tasks']) == 51
    for task in manifest['tasks']:
        assert state['tasks'][task['id']]['state'] == 'completed', task['id']
        validate(task)
    frozen = manifest['frozen_hashes']
    for path, expected in frozen.items():
        assert digest(Path(manifest['source_root']) / path) == expected, path
    implementation = hashlib.sha256(b''.join(Path(p).read_bytes() for p in IMPLEMENTATION)).hexdigest()
    records = []
    episodes = 0
    for task in manifest['tasks']:
        if task['kind'] != 'minimal_evaluation':
            continue
        run = Path(task['output'])
        data = json.loads((run / 'evaluation-minimal.json').read_text())
        config = json.loads((run / 'config.json').read_text())
        assert data['implementation_sha256'] == config['implementation_sha256'] == implementation
        assert Path(config['task_set']).is_file()
        checkpoint = torch.load(task['checkpoint'], map_location='cpu', weights_only=False)
        original_config = checkpoint['config']
        config_differences = {k: [original_config.get(k), config.get(k)] for k in set(original_config)|set(config) if original_config.get(k) != config.get(k)}
        if config_differences:
            assert run.name in ('flatworld-hv_only-s1', 'flatworld-gnn_single-s1')
            assert set(config_differences) <= {'task_set','steps','checkpoint_every','device','output','resume'}
            assert digest(original_config['task_set']) == digest(config['task_set'])
        assert original_config['task_manifest'] == config['task_manifest']
        assert checkpoint['steps'] == data['checkpoint_steps']
        tensor_count = 0
        def check(value):
            nonlocal tensor_count
            if torch.is_tensor(value):
                tensor_count += 1
                assert bool(torch.isfinite(value).all()), run
            elif isinstance(value, dict):
                for item in value.values(): check(item)
            elif isinstance(value, (list, tuple)):
                for item in value: check(item)
        check(checkpoint)
        assert tensor_count > 0
        del checkpoint
        for formula, group in data['per_formula'].items():
            rows = [r for r in data['episodes'] if r['formula'] == formula]
            assert len(rows) == group['episodes']
            successes = [r for r in rows if r['success']]
            expected = {
                'success_rate': mean(r['success'] for r in rows) if group['finite'] else None,
                'violation_rate': mean(r['violation'] for r in rows),
                'mean_steps': mean(r['steps'] for r in rows),
                'successful_episodes': len(successes),
                'successful_episode_mean_steps': mean(r['steps'] for r in successes) if successes and group['finite'] else None,
                'mean_reward': mean(r['reward'] for r in rows),
                'mean_discounted_return': mean(r['discounted_return'] for r in rows),
                'mean_accepting_visits': mean(r['accepting_visits'] for r in rows),
                'mean_accepting_visits_per_step': mean(r['accepting_visits']/r['steps'] for r in rows),
            }
            for key, value in expected.items():
                actual = group[key]
                assert (actual is None and value is None) or (actual is not None and value is not None and math.isclose(actual, value, rel_tol=1e-10, abs_tol=1e-12)), (run, formula, key)
        episodes += len(data['episodes'])
        records.append(dict(run=run.name, checkpoint=str(Path(task['checkpoint'])),
                            checkpoint_sha256=digest(task['checkpoint']), steps=data['checkpoint_steps'],
                            evaluation_sha256=digest(run/'evaluation-minimal.json'),
                            configuration=task['configuration'], environment=task['env'], seed=task['seed'],
                            parameter_count=config['parameter_count'], tensor_count=tensor_count, original_checkpoint_config=original_config, archive_config_differences=config_differences))
        print('PASS', run.name, flush=True)
    assert len(records) == 25 and episodes == 6900
    with tempfile.TemporaryDirectory(prefix='hiltl-summary-audit-') as directory:
        output = Path(directory)/'summary.json'
        subprocess.run([sys.executable, 'scripts/summarize_paper_minimal.py', '--manifest', str(root/'manifest.json'), '--output', str(output)], check=True)
        assert json.loads(output.read_text()) == json.loads((root/'summary.json').read_text())
        assert output.with_suffix('.csv').read_bytes() == (root/'summary.csv').read_bytes()
    report = dict(checked_utc=datetime.now(timezone.utc).isoformat(), state='passed',
                  stages=51, models=25, episodes=episodes, actual_checkpoint_steps=sum(r['steps'] for r in records),
                  capped_training_steps=sum(min(200000,r['steps']) for r in records),
                  manifest_sha256=digest(root/'manifest.json'), implementation_sha256=implementation,
                  checks=['all scheduler stages and required outputs', 'frozen source hashes',
                          'checkpoint/spec SHA256', '25 CPU checkpoint loads and finite model/optimizer tensors',
                          'per-episode recomputation of all nine reported formula metrics',
                          'independent summary JSON and byte-identical CSV regeneration'], checkpoints=records)
    if export:
        export.mkdir(parents=True, exist_ok=False)
        for name in ['manifest.json','state.json','summary.json','summary.csv','events.jsonl']:
            shutil.copy2(root/name, export/name)
        shutil.copytree(root/'specs', export/'specs')
        for record in records:
            source=root/'runs'/record['run']; target=export/'runs'/record['run']; target.mkdir(parents=True)
            for name in ['evaluation-minimal.json','config.json','metrics.jsonl','progress.json']:
                shutil.copy2(source/name, target/name)
        (export/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
        hashes={str(p.relative_to(export)):digest(p) for p in sorted(export.rglob('*')) if p.is_file()}
        (export/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checkpoints'},indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('experiments/reproduction/20261021-zone-minimal-v3'))
    parser.add_argument('--export',type=Path)
    args=parser.parse_args()
    audit(args.root,args.export)


if __name__=='__main__': main()
