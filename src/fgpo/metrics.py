"""Official full-split SR/TGC/SGC, with no resource-dependent success replacement."""
from __future__ import annotations

from collections import defaultdict
from statistics import fmean

from .envs import ALFWORLD_TABLE_COLUMNS, alfworld_task_type


def official_summary(benchmark, records, expected_tasks):
    """One method/seed/split, aggregated from actual official evaluator receipts.

    Records carry task_id, official_evaluation, steps, input_tokens/output_tokens.
    Missing official tasks are reported as missing and never silently scored zero.
    Scenario membership is taken from the official full task list.
    """
    expected = {str(row['task_id']): row for row in expected_tasks}
    observed = {str(row['task_id']): row for row in records}
    if len(observed) != len(records):
        raise ValueError('Official split metrics require one actual episode per task.')
    scored = {task: row['official_evaluation']['success'] for task, row in observed.items()
              if isinstance(row.get('official_evaluation', {}).get('success'), bool)}
    missing = sorted(set(expected) - set(scored))
    extra = sorted(set(observed) - set(expected))
    complete = not missing and not extra
    count = len(scored)
    rate = sum(scored.values()) / count if count else None
    result = {'benchmark': benchmark, 'official_task_count': len(expected),
              'evaluated_count': count, 'missing_tasks': missing, 'unexpected_tasks': extra,
              'dataset_complete': complete, 'success_count': sum(scored.values()),
              'mean_steps': fmean(row['steps'] for row in records) if records else None,
              'mean_input_tokens': fmean(row['input_tokens'] for row in records) if records else None,
              'mean_output_tokens': fmean(row['output_tokens'] for row in records) if records else None}
    if benchmark == 'alfworld':
        result['SR'] = 100 * rate if complete and rate is not None else None
        expected_types = {task: alfworld_task_type(task) for task in expected}
        observed_types = {task: alfworld_task_type(task) for task in observed}
        task_types = {}
        for category in ALFWORLD_TABLE_COLUMNS[:-1]:
            tasks = {task for task, kind in expected_types.items() if kind == category}
            evaluated = {task: success for task, success in scored.items()
                         if observed_types[task] == category}
            absent = sorted(tasks - set(evaluated))
            unexpected = sorted((set(evaluated) - tasks) | {
                task for task in extra if observed_types[task] == category})
            category_complete = bool(tasks) and not absent and not unexpected
            task_types[category] = {
                'SR': (100 * sum(evaluated.values()) / len(evaluated)
                       if category_complete and evaluated else None),
                'official_task_count': len(tasks), 'evaluated_count': len(evaluated),
                'success_count': sum(evaluated.values()), 'missing_tasks': absent,
                'unexpected_tasks': unexpected, 'dataset_complete': category_complete,
            }
        # All is task-weighted, exactly the existing official SR.
        task_types['All'] = {key: result[key] for key in (
            'SR', 'official_task_count', 'evaluated_count', 'success_count',
            'missing_tasks', 'unexpected_tasks', 'dataset_complete')}
        result['task_types'] = task_types
    elif benchmark == 'appworld':
        scenarios = defaultdict(list)
        for task, row in expected.items():
            scenarios[str(row['scenario_id'])].append(task)
        result['TGC'] = 100 * rate if complete and rate is not None else None
        result['SGC'] = (100 * sum(all(scored[task] for task in tasks) for tasks in scenarios.values())
                         / len(scenarios)) if complete and scenarios else None
        result['scenario_count'] = len(scenarios)
    else:
        raise ValueError(f'Unknown official benchmark: {benchmark}')
    result['score_unit'] = 'percent'
    return result
