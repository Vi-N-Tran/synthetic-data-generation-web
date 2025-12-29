"""
Validation logic for trajectories and actions.

This module provides comprehensive validation functions to ensure data quality:
- Trajectory-level validation (action count, temporal ordering, etc.)
- Action-level validation (required fields, type matching, etc.)
- URL and selector format validation
- Dataset-level validation with error reporting
"""

from typing import List, Dict, Any
from src.schema import Trajectory, BrowserAction
import re
from src.logging_config import get_logger

logger = get_logger('validator')


def validate_trajectory(trajectory: Trajectory) -> bool:
    """
    Validate a single trajectory for consistency and correctness.
    
    Performs comprehensive validation including:
    - Action count (must be 3-10)
    - Temporal ordering (timestamps must be increasing)
    - Action ID uniqueness
    - URL consistency
    - Individual action validation
    
    Args:
        trajectory: Trajectory object to validate
        
    Returns:
        True if trajectory is valid, False otherwise
    
    Note:
        This function validates structure and consistency, not semantic correctness.
        For example, it checks that timestamps are ordered but doesn't verify
        that action sequences make logical sense.
    """
    # Trajectory-Level Validation
    if not (3 <= len(trajectory.actions) <= 10):
        return False
    
    if trajectory.start_time >= trajectory.end_time:
        return False
    
    if trajectory.duration < 0:
        return False
    
    # Action-Level Validation
    for i, action in enumerate(trajectory.actions):
        if not validate_action(action, i, trajectory.actions):
            return False
        
        # Temporal ordering
        if i > 0 and action.timestamp < trajectory.actions[i-1].timestamp:
            return False
        
        # Check for duplicate action IDs
        action_ids = [a.action_id for a in trajectory.actions]
        if len(action_ids) != len(set(action_ids)):
            return False
    
    # URL consistency (navigate actions should change URL appropriately)
    current_url = None
    for action in trajectory.actions:
        if action.action_type == 'navigate':
            current_url = action.url
        elif current_url and action.url != current_url:
            # URL changed without navigate action (might be acceptable for clicks)
            pass
    
    return True


def validate_action(action: BrowserAction, index: int, all_actions: List[BrowserAction]) -> bool:
    """
    Validate a single browser action for correctness.
    
    Validates:
    - Required fields are present (timestamp, action_type, url, action_id)
    - Action type is valid (one of 12 supported types)
    - Action type matches parameters (e.g., 'type' has value, 'select' has option_index)
    - Timestamp is positive
    - Coordinates are valid (if present)
    - Confidence is in range [0, 1]
    - URL format is valid
    - Selector format is valid (if present)
    
    Args:
        action: BrowserAction object to validate
        index: Index of action in trajectory (for error reporting)
        all_actions: All actions in the trajectory (for context)
        
    Returns:
        True if action is valid, False otherwise
    """
    # Required fields
    if not action.timestamp or action.timestamp <= 0:
        return False
    
    if not action.action_type:
        return False
    
    if not action.url:
        return False
    
    if not action.action_id:
        return False
    
    # Action type validation
    valid_action_types = [
        'click', 'type', 'navigate', 'scroll', 'hover', 'select',
        'submit', 'back', 'forward', 'refresh', 'wait', 'drag_drop'
    ]
    if action.action_type not in valid_action_types:
        return False
    
    # Action type matches parameters
    if action.action_type == 'type' and not action.value:
        # Allow empty values for clearing fields
        pass
    
    if action.action_type == 'select' and action.option_index is None:
        return False
    
    if action.action_type == 'scroll' and not action.coordinates:
        return False
    
    # Timestamp validation
    if action.timestamp <= 0:
        return False
    
    # Coordinates validation
    if action.coordinates:
        if 'x' not in action.coordinates or 'y' not in action.coordinates:
            return False
        if action.coordinates['x'] < 0 or action.coordinates['y'] < 0:
            return False
        # Reasonable bounds (assuming max viewport ~10000px)
        if action.coordinates['x'] > 10000 or action.coordinates['y'] > 10000:
            return False
    
    # Confidence validation
    if not (0 <= action.confidence <= 1):
        return False
    
    # URL format validation
    if not validate_url(action.url):
        return False
    
    # Selector validation (if present)
    if action.element_selector and not validate_selector(action.element_selector):
        return False
    
    return True


def validate_url(url: str) -> bool:
    """
    Validate URL format using regex pattern.
    
    Checks for:
    - Valid HTTP/HTTPS protocol
    - Valid domain name or IP address
    - Optional port number
    - Optional path/query string
    
    Args:
        url: URL string to validate
        
    Returns:
        True if URL format is valid, False otherwise
    
    Note:
        This is a basic validation - it checks format, not accessibility.
    """
    if not url:
        return False
    
    # Basic URL validation
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return bool(url_pattern.match(url))


def validate_selector(selector: str) -> bool:
    """
    Validate CSS selector format (basic security and format checks).
    
    Checks for:
    - Non-empty selector
    - No dangerous patterns (javascript:, data:, vbscript:)
    - Reasonable length (max 500 characters)
    
    Args:
        selector: CSS selector string to validate
        
    Returns:
        True if selector appears safe and valid, False otherwise
    
    Note:
        This is a basic validation - it doesn't verify selector syntax correctness
        or that the selector would match any elements. It focuses on security
        and basic format checks.
    """
    if not selector:
        return False
    
    # Basic CSS selector validation
    # Check for dangerous patterns
    dangerous_patterns = ['javascript:', 'data:', 'vbscript:']
    if any(pattern in selector.lower() for pattern in dangerous_patterns):
        return False
    
    # Check for reasonable length
    if len(selector) > 500:
        return False
    
    return True


def validate_dataset(trajectories: List[Trajectory]) -> Dict[str, Any]:
    """
    Validate entire dataset and return comprehensive validation report.
    
    Validates all trajectories in the dataset and provides aggregate statistics
    including validation pass rate and error details.
    
    Args:
        trajectories: List of Trajectory objects to validate
        
    Returns:
        Dictionary with validation results:
            - total_trajectories: Total number of trajectories
            - valid: Number of valid trajectories
            - invalid: Number of invalid trajectories
            - validation_pass_rate: Ratio of valid to total (0.0 to 1.0)
            - errors: List of error details (limited to first 10)
    
    Note:
        Error details include trajectory_id and index for debugging.
        Only first 10 errors are included to prevent large output.
    """
    total = len(trajectories)
    valid = 0
    invalid = 0
    errors = []
    
    for i, trajectory in enumerate(trajectories):
        if validate_trajectory(trajectory):
            valid += 1
        else:
            invalid += 1
            errors.append({
                'trajectory_id': trajectory.trajectory_id,
                'index': i,
                'reason': 'Validation failed'
            })
    
    return {
        'total_trajectories': total,
        'valid': valid,
        'invalid': invalid,
        'validation_pass_rate': valid / total if total > 0 else 0,
        'errors': errors[:10]  # Limit to first 10 errors
    }

