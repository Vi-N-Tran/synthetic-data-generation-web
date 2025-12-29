#!/usr/bin/env python3
"""Analyze generated trajectories for error cases and skipped-step scenarios"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

def analyze_trajectories(filepath: str) -> Dict[str, Any]:
    """Analyze trajectories from JSONL file"""
    trajectories = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    
    results = {
        'total_trajectories': len(trajectories),
        'trajectories_with_errors': [],
        'trajectories_with_goal_achieved_false': [],
        'trajectories_with_skipped_steps': [],
        'error_statistics': {
            'total_actions': 0,
            'actions_with_element_visible_false': 0,
            'actions_with_element_clickable_false': 0,
            'actions_with_both_false': 0
        }
    }
    
    for traj in trajectories:
        actions = traj.get('actions', [])
        results['error_statistics']['total_actions'] += len(actions)
        
        has_error = False
        error_actions = []
        
        for action in actions:
            element_visible = action.get('element_visible', True)
            element_clickable = action.get('element_clickable', True)
            
            if not element_visible:
                results['error_statistics']['actions_with_element_visible_false'] += 1
                has_error = True
                error_actions.append(action)
            
            if not element_clickable:
                results['error_statistics']['actions_with_element_clickable_false'] += 1
                has_error = True
                error_actions.append(action)
            
            if not element_visible and not element_clickable:
                results['error_statistics']['actions_with_both_false'] += 1
        
        if has_error:
            results['trajectories_with_errors'].append({
                'trajectory_id': traj.get('trajectory_id'),
                'goal': traj.get('goal'),
                'goal_achieved': traj.get('goal_achieved'),
                'error_count': traj.get('error_count', 0),
                'action_count': len(actions),
                'error_actions': [
                    {
                        'action_id': a.get('action_id'),
                        'action_type': a.get('action_type'),
                        'element_visible': a.get('element_visible'),
                        'element_clickable': a.get('element_clickable'),
                        'context': a.get('context', '')
                    }
                    for a in error_actions
                ]
            })
        
        goal_achieved = traj.get('goal_achieved', True)
        if not goal_achieved:
            results['trajectories_with_goal_achieved_false'].append({
                'trajectory_id': traj.get('trajectory_id'),
                'goal': traj.get('goal'),
                'action_count': len(actions),
                'last_action_type': actions[-1].get('action_type') if actions else None
            })
        
        # Check for skipped steps: goal_achieved=True but relatively few actions
        # (indicating user took shortcuts)
        if goal_achieved and len(actions) <= 5:
            results['trajectories_with_skipped_steps'].append({
                'trajectory_id': traj.get('trajectory_id'),
                'goal': traj.get('goal'),
                'action_count': len(actions),
                'actions': [a.get('action_type') for a in actions]
            })
    
    return results

def print_analysis(results: Dict[str, Any]):
    """Print analysis results"""
    print("=" * 80)
    print("TRAJECTORY ANALYSIS RESULTS")
    print("=" * 80)
    print(f"\nTotal trajectories: {results['total_trajectories']}")
    
    print(f"\n{'='*80}")
    print("ERROR CASES")
    print(f"{'='*80}")
    print(f"Trajectories with errors: {len(results['trajectories_with_errors'])}")
    print(f"  - Actions with element_visible=False: {results['error_statistics']['actions_with_element_visible_false']}")
    print(f"  - Actions with element_clickable=False: {results['error_statistics']['actions_with_element_clickable_false']}")
    print(f"  - Actions with both=False: {results['error_statistics']['actions_with_both_false']}")
    print(f"  - Total actions analyzed: {results['error_statistics']['total_actions']}")
    
    if results['trajectories_with_errors']:
        print("\nSample error trajectories:")
        for traj in results['trajectories_with_errors'][:3]:
            print(f"\n  Trajectory ID: {traj['trajectory_id']}")
            print(f"  Goal: {traj['goal']}")
            print(f"  Goal achieved: {traj['goal_achieved']}")
            print(f"  Error count: {traj['error_count']}")
            print(f"  Actions with errors:")
            for err_action in traj['error_actions']:
                print(f"    - {err_action['action_type']}: visible={err_action['element_visible']}, clickable={err_action['element_clickable']}")
    else:
        print("\n  ⚠️  No error cases found in trajectories!")
    
    print(f"\n{'='*80}")
    print("SKIPPED STEP TRAJECTORIES (goal_achieved=True with ≤5 actions)")
    print(f"{'='*80}")
    print(f"Trajectories with skipped steps: {len(results['trajectories_with_skipped_steps'])}")
    
    if results['trajectories_with_skipped_steps']:
        print("\nSample skipped-step trajectories:")
        for traj in results['trajectories_with_skipped_steps'][:3]:
            print(f"\n  Trajectory ID: {traj['trajectory_id']}")
            print(f"  Goal: {traj['goal']}")
            print(f"  Action count: {traj['action_count']}")
            print(f"  Actions: {' -> '.join(traj['actions'])}")
    else:
        print("\n  ⚠️  No skipped-step trajectories found!")
    
    print(f"\n{'='*80}")
    print("FAILED GOALS (goal_achieved=False)")
    print(f"{'='*80}")
    print(f"Trajectories with goal_achieved=False: {len(results['trajectories_with_goal_achieved_false'])}")
    
    if results['trajectories_with_goal_achieved_false']:
        print("\nSample failed trajectories:")
        for traj in results['trajectories_with_goal_achieved_false'][:3]:
            print(f"\n  Trajectory ID: {traj['trajectory_id']}")
            print(f"  Goal: {traj['goal']}")
            print(f"  Action count: {traj['action_count']}")
            print(f"  Last action: {traj['last_action_type']}")
    else:
        print("\n  ⚠️  No failed goals found in trajectories!")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Find most recent output file
        output_dir = Path('output')
        if output_dir.exists():
            jsonl_files = sorted(output_dir.glob('*trajectories*.jsonl'), key=lambda p: p.stat().st_mtime, reverse=True)
            if jsonl_files:
                filepath = str(jsonl_files[0])
                print(f"Using most recent file: {filepath}\n")
            else:
                print("Error: No JSONL files found in output directory")
                sys.exit(1)
        else:
            print("Error: output directory not found")
            sys.exit(1)
    else:
        filepath = sys.argv[1]
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    results = analyze_trajectories(filepath)
    print_analysis(results)

