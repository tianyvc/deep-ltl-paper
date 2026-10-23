"""End-to-end scheduler checks with isolated synthetic artifacts, no training GPUs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def wait_for(fn, timeout=60):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        try:
            value=fn()
            if value:
                return value
        except (FileNotFoundError,json.JSONDecodeError):
            pass
        time.sleep(.5)
    raise TimeoutError('Scheduler test timed out')


def main():
    with tempfile.TemporaryDirectory(prefix='hiltl-queue-test-') as temp:
        root=Path(temp)
        worker=root/'worker.py'
        worker.write_text("import sys,json\nfrom pathlib import Path\np=Path(sys.argv[1]);p.mkdir(exist_ok=True)\n(p/'progress.json').write_text(json.dumps(dict(state='completed',steps=2,updates=2)))\n(p/'status.pt').write_bytes(b'test artifact, not a model')\n(p/'metrics.jsonl').write_text((json.dumps(dict(loss=1.,value_loss=1.,gradient_norm=1.))+'\\n')*2)\n")
        tasks=[dict(id='good',kind='pilot',output=str(root/'good'),steps=2,depends=[],argv=[str(worker),str(root/'good')]),dict(id='bad',kind='train',output=str(root/'bad'),steps=2,depends=['good'],argv=['-c','raise RuntimeError("intentional test failure")']),dict(id='blocked',kind='train',output=str(root/'blocked'),steps=2,depends=['bad'],argv=['-c','raise AssertionError("must never run")'])]
        manifest=root/'manifest.json'
        manifest.write_text(json.dumps(dict(source_root=str(Path.cwd()),gpus=[1],max_workers=1,min_free_mib=4096,poll_seconds=.25,frozen_hashes={},tasks=tasks)))
        # Reject duplicate IDs before any background worker can be started.
        bad_manifest=root/'duplicate.json'
        bad_manifest.write_text(json.dumps(dict(source_root=str(Path.cwd()),tasks=[tasks[0],tasks[0]])))
        rejected=subprocess.run([sys.executable,'scripts/paper_campaign.py','run','--manifest',str(bad_manifest)],capture_output=True,text=True)
        assert rejected.returncode != 0 and 'Duplicate task IDs' in rejected.stderr
        assert not (root/'state.json').exists()
        def launch(free):
            code="import sys;sys.path.insert(0,'scripts');import paper_campaign as p;p.free_gpus=lambda:{1:{'free_mib':"+str(free)+",'utilization':0}};sys.exit(p.run(sys.argv[1]))"
            return subprocess.Popen([sys.executable,'-c',code,str(manifest)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        process=launch(0)
        state=lambda:json.loads((root/'state.json').read_text())
        wait_for(lambda:(root/'state.json').exists() and state().get('waiting_reason'))
        assert state()['tasks']['good']['state']=='pending'
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=20)==0 and state()['state']=='paused'
        process=launch(5000)
        duplicate=launch(5000)
        assert duplicate.wait(timeout=20)!=0,'Duplicate scheduler acquired lock'
        assert process.wait(timeout=60)==1,process.stdout.read()
        result=state()
        assert result['state']=='finished_with_failures'
        assert [result['tasks'][key]['state'] for key in ['good','bad','blocked']]==['completed','failed','blocked']
        jobs=list((root/'jobs').iterdir())
        process=launch(5000)
        assert process.wait(timeout=20)==1
        assert list((root/'jobs').iterdir())==jobs,'Restart repeated completed tasks'
        print(json.dumps({'passed':['duplicate_task_id_rejected_before_launch','resource_wait_without_launch','graceful_pause','resume_pending_queue','duplicate_scheduler_lock','real_background_worker_success','worker_failure_blocks_dependents','completed_tasks_not_repeated_on_restart']}))

if __name__=='__main__':
    main()
