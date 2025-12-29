"""Statistics computation for generated datasets"""

from typing import Dict, List, Any
from collections import Counter, defaultdict
from src.schema import Trajectory, BrowserAction


def compute_dataset_statistics(trajectories: List[Trajectory]) -> Dict[str, Any]:
    """
    Compute statistics over the generated dataset.
    
    Args:
        trajectories: List of trajectories
        
    Returns:
        Dictionary with computed statistics
    """
    if not trajectories:
        return {
            "n_trajectories": 0,
            "n_actions": 0
        }
    
    # Dataset-Level Statistics
    total_actions = sum(len(t.actions) for t in trajectories)
    avg_trajectory_length = total_actions / len(trajectories) if trajectories else 0
    
    # Trajectory length distribution
    length_dist = Counter(len(t.actions) for t in trajectories)
    
    # Workflow type distribution
    workflow_dist = Counter(t.workflow_type for t in trajectories)
    
    # User type distribution
    user_type_dist = Counter(t.user_type for t in trajectories)
    
    # Device type distribution
    device_type_dist = Counter(t.device_type for t in trajectories)
    
    # Browser type distribution
    browser_type_dist = Counter(t.browser_type for t in trajectories)
    
    # Goal achievement
    goal_achieved_count = sum(1 for t in trajectories if t.goal_achieved)
    goal_achievement_rate = goal_achieved_count / len(trajectories) if trajectories else 0
    
    # Average trajectory duration
    avg_duration = sum(t.duration for t in trajectories) / len(trajectories) if trajectories else 0
    
    # Action-Level Statistics
    all_actions = [action for trajectory in trajectories for action in trajectory.actions]
    
    # Action type distribution
    action_type_dist = Counter(a.action_type for a in all_actions)
    
    # Element type distribution
    element_type_dist = Counter(a.element_type for a in all_actions if a.element_type)
    
    # Average time between actions
    action_intervals = []
    for trajectory in trajectories:
        if len(trajectory.actions) >= 2:
            intervals = [
                trajectory.actions[i].timestamp - trajectory.actions[i-1].timestamp
                for i in range(1, len(trajectory.actions))
            ]
            action_intervals.extend(intervals)
    
    avg_action_interval = sum(action_intervals) / len(action_intervals) / 1000.0 if action_intervals else 0
    
    # Intentional vs exploratory
    intentional_count = sum(1 for a in all_actions if a.is_intentional)
    intentional_ratio = intentional_count / len(all_actions) if all_actions else 0
    
    # Error count (element not visible/clickable)
    error_count = sum(1 for a in all_actions if not a.element_visible or not a.element_clickable)
    error_rate = error_count / len(all_actions) if all_actions else 0
    
    # Quality Metrics
    avg_confidence = sum(a.confidence for a in all_actions) / len(all_actions) if all_actions else 0
    
    # Backtrack frequency
    backtrack_count = sum(t.backtrack_count for t in trajectories)
    avg_backtrack_per_trajectory = backtrack_count / len(trajectories) if trajectories else 0
    
    # Aggregate trajectory quality metrics
    avg_action_count = sum(t.action_count for t in trajectories) / len(trajectories) if trajectories else 0
    avg_avg_action_interval = sum(t.avg_action_interval for t in trajectories) / len(trajectories) if trajectories else 0
    avg_error_count = sum(t.error_count for t in trajectories) / len(trajectories) if trajectories else 0
    
    return {
        # Dataset-Level
        "n_trajectories": len(trajectories),
        "n_actions": total_actions,
        "avg_trajectory_length": round(avg_trajectory_length, 2),
        "trajectory_length_distribution": dict(length_dist),
        "workflow_type_distribution": dict(workflow_dist),
        "user_type_distribution": dict(user_type_dist),
        "device_type_distribution": dict(device_type_dist),
        "browser_type_distribution": dict(browser_type_dist),
        "avg_trajectory_duration_seconds": round(avg_duration, 2),
        "goal_achievement_rate": round(goal_achievement_rate, 3),
        
        # Action-Level
        "action_type_distribution": dict(action_type_dist),
        "element_type_distribution": dict(element_type_dist),
        "avg_action_interval_seconds": round(avg_action_interval, 2),
        "intentional_action_ratio": round(intentional_ratio, 3),
        "error_rate": round(error_rate, 3),
        
        # Quality Metrics
        "avg_confidence": round(avg_confidence, 3),
        "avg_backtrack_per_trajectory": round(avg_backtrack_per_trajectory, 2),
        "avg_action_count": round(avg_action_count, 2),
        "avg_action_interval_across_trajectories": round(avg_avg_action_interval, 2),
        "avg_error_count_per_trajectory": round(avg_error_count, 2)
    }

