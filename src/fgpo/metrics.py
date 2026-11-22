"""Official benchmark endpoint metrics from native episode receipts."""
from __future__ import annotations

from collections import defaultdict
from statistics import fmean

from .envs import ALFWORLD_TABLE_COLUMNS, alfworld_task_type
from .textcraft import TEXTCRAFT_COUNTS
from .webshop import VALIDATION_SESSIONS, VALIDATION_WORKERS


def webshop_score(metrics):
    score, purchased, success = metrics.get('score'), metrics.get('purchased'), metrics.get('success')
    if (type(score) not in (float, int) or not 0. <= score <= 1.
            or type(purchased) is not bool or (not purchased and score != 0.)
            or success is not (purchased and score == 1.)):
        raise ValueError('WebShop receipt has an inconsistent native score or full-success outcome.')
    return score


def textcraft_success(metrics, task):
    won = metrics.get('won')
    if (type(won) is not bool or metrics.get('success') is not won
            or metrics.get('task_score') != float(won)
            or metrics.get('goal_item') != task['goal_item']
            or type(metrics.get('depth')) is not int or metrics['depth'] != task['depth']
            or metrics.get('step_count') != metrics.get('steps')):
        raise ValueError('TextCraft receipt differs from the official worker outcome or goal depth.')
    return won


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
    elif benchmark == 'webshop':
        if len(expected) != VALIDATION_WORKERS:
            raise ValueError('WebShop requires the released 128-episode validation draw.')
        scores = [webshop_score(row['official_evaluation']) for row in records]
        result.update(SR=100 * rate if complete else None,
                      Score=100 * fmean(scores) if complete else None,
                      validation_workers=VALIDATION_WORKERS,
                      validation_session_population=VALIDATION_SESSIONS)
    elif benchmark == 'textcraft':
        if len(expected) != TEXTCRAFT_COUNTS['val'] or any(row['split'] != 'val' for row in expected.values()):
            raise ValueError('TextCraft evaluation requires every task in the original val split.')
        by_depth = defaultdict(list)
        for task, row in expected.items():
            by_depth[int(row['depth'])].append(task)
            if task in observed:
                textcraft_success(observed[task]['official_evaluation'], row)
        result['SR'] = 100 * rate if complete and rate is not None else None
        result['SR_at_70'] = result['SR']
        result['success_count_at_70'] = result['success_count']
        scored_at_30 = {task: bool(scored[task] and observed[task]['steps'] <= 30) for task in scored}
        scored_at_60 = {task: bool(scored[task] and observed[task]['steps'] <= 60) for task in scored}
        result['success_count_at_30'] = sum(scored_at_30.values())
        result['SR_at_30'] = (100 * sum(scored_at_30.values()) / count if complete and count else None)
        result['success_count_at_60'] = sum(scored_at_60.values())
        result['SR_at_60'] = (100 * sum(scored_at_60.values()) / count if complete and count else None)
        newly_31_70 = sorted(
            ({'task_id': task, 'depth': int(expected[task]['depth']),
              'first_success_step': int(observed[task]['steps'])}
             for task in scored if scored[task] and observed[task]['steps'] > 30),
            key=lambda item: (item['depth'], item['first_success_step'], item['task_id']),
        )
        result['newly_solved_31_to_70'] = newly_31_70
        result['newly_solved_31_to_70_count'] = len(newly_31_70)
        result['recipe_depths'] = {}
        result['recipe_depths_at_30'] = {}
        for depth, tasks in sorted(by_depth.items()):
            values = [scored[task] for task in tasks if task in scored]
            values_30 = [scored_at_30[task] for task in tasks if task in scored_at_30]
            absent = sorted(task for task in tasks if task not in scored)
            unexpected = sorted(task for task in extra if observed[task]['official_evaluation'].get('depth') == depth)
            depth_complete = not absent and not unexpected
            result['recipe_depths'][str(depth)] = {
                'SR': 100 * fmean(values) if depth_complete else None,
                'official_task_count': len(tasks), 'evaluated_count': len(values),
                'success_count': sum(values), 'missing_tasks': absent, 'unexpected_tasks': unexpected,
                'dataset_complete': depth_complete,
            }
            result['recipe_depths_at_30'][str(depth)] = {
                'SR': 100 * fmean(values_30) if depth_complete else None,
                'official_task_count': len(tasks), 'evaluated_count': len(values_30),
                'success_count': sum(values_30), 'missing_tasks': absent, 'unexpected_tasks': unexpected,
                'dataset_complete': depth_complete,
            }
    else:
        raise ValueError(f'Unknown official benchmark: {benchmark}')
    result['score_unit'] = 'percent'
    return result
