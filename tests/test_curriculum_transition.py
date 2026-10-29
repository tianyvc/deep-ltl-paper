"""Regression: in-flight episodes must not pollute a new curriculum stage."""
import json
import numpy as np
from sequence.samplers.curriculum import Curriculum, ExplicitCurriculumStage


def main():
    old = ExplicitCurriculumStage(.8, 'min', lambda _: ['old-a', 'old-b'])
    new = ExplicitCurriculumStage(.95, 'mean', lambda _: ['new-a', 'new-b'])
    curriculum = Curriculum([old, new])
    curriculum.sample([])
    curriculum.update_task_success({'old-a': 1., 'old-b': 1.})
    assert curriculum.stage_index == 1
    curriculum.update_task_success({'old-a': 1.})
    assert curriculum.stage_index == 1 and new._task_success is None
    curriculum.sample([])
    curriculum.update_task_success({'new-a': 0., 'old-a': 1.})
    assert set(new._task_success) == {'new-a', 'new-b'}
    # The second update triggered the production failure before this fix.
    curriculum.update_task_success({'new-b': .5, 'old-b': 1.})
    probabilities = new.compute_sampling_prob()
    assert set(new._task_success) == set(new._tasks)
    assert np.isfinite(probabilities).all() and np.isclose(probabilities.sum(), 1.)
    assert curriculum.stage_index == 1
    before = dict(new._task_success)
    curriculum.update_task_success({'old-a': 1., 'old-b': 1.})
    assert new._task_success == before and curriculum.stage_index == 1
    for _ in range(20):
        assert curriculum.sample([]) in new._tasks
    curriculum.update_task_success({'new-a': 1., 'new-b': 1.})
    assert curriculum.finished
    print(json.dumps({'passed': ['late_old_episode_before_first_new_sample',
        'mixed_stage_statistics_first_and_second_update',
        'sampling_table_closed_under_current_tasks', 'old_only_cannot_advance',
        'original_current_stage_threshold_preserved']}))


if __name__ == '__main__':
    main()
