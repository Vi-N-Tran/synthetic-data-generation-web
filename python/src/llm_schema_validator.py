"""Schema validation for LLM-generated trajectory structures"""

from typing import Dict, List, Any, Optional, Tuple
import re


# Valid action types from SPEC.md
VALID_ACTION_TYPES = [
    'click', 'type', 'navigate', 'scroll', 'hover', 'select',
    'submit', 'back', 'forward', 'refresh', 'wait', 'drag_drop'
]


def validate_trajectory_structure(structure: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that LLM-generated trajectory structure matches expected schema.
    
    Args:
        structure: Dictionary from LLM containing trajectory structure
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check top-level structure
    if not isinstance(structure, dict):
        errors.append("Trajectory structure must be a dictionary")
        return False, errors
    
    # Check required top-level fields
    if 'actions' not in structure:
        errors.append("Missing required field: 'actions'")
    
    # Validate domain (optional but if present should be string)
    if 'domain' in structure and not isinstance(structure['domain'], str):
        errors.append("'domain' must be a string")
    
    # Validate goal_achieved (optional but if present should be boolean)
    if 'goal_achieved' in structure and not isinstance(structure['goal_achieved'], bool):
        errors.append("'goal_achieved' must be a boolean")
    
    # Validate actions array
    if 'actions' in structure:
        actions = structure['actions']
        if not isinstance(actions, list):
            errors.append("'actions' must be a list")
        elif len(actions) == 0:
            errors.append("'actions' list cannot be empty")
        else:
            # Validate each action
            for i, action in enumerate(actions):
                action_errors = validate_action_structure(action, index=i)
                errors.extend(action_errors)
    
    return len(errors) == 0, errors


def validate_action_structure(action: Dict[str, Any], index: int = 0) -> List[str]:
    """
    Validate a single action structure from LLM output.
    
    Args:
        action: Dictionary containing action data
        index: Index of action in actions list (for error reporting)
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    prefix = f"Action[{index}]"
    
    if not isinstance(action, dict):
        return [f"{prefix}: Action must be a dictionary"]
    
    # Required fields
    if 'action_type' not in action:
        errors.append(f"{prefix}: Missing required field 'action_type'")
    else:
        action_type = action['action_type']
        if not isinstance(action_type, str):
            errors.append(f"{prefix}: 'action_type' must be a string")
        elif action_type not in VALID_ACTION_TYPES:
            errors.append(f"{prefix}: Invalid 'action_type' '{action_type}'. Must be one of: {VALID_ACTION_TYPES}")
    
    # Validate URL (required for most actions)
    if 'url' not in action:
        errors.append(f"{prefix}: Missing recommended field 'url'")
    elif not isinstance(action['url'], str):
        errors.append(f"{prefix}: 'url' must be a string")
    elif action['url'] and not _validate_url_format(action['url']):
        errors.append(f"{prefix}: 'url' has invalid format")
    
    # Validate page_title (recommended)
    if 'page_title' in action and not isinstance(action['page_title'], str):
        errors.append(f"{prefix}: 'page_title' must be a string")
    
    # Validate element_type (optional, but if present should be string)
    if 'element_type' in action and not isinstance(action['element_type'], str):
        errors.append(f"{prefix}: 'element_type' must be a string")
    
    # Validate optional fields based on action type
    action_type = action.get('action_type')
    
    if action_type == 'type':
        # Type actions should have value or value_hint
        if 'value' not in action and 'value_hint' not in action:
            errors.append(f"{prefix}: 'type' action should have 'value' or 'value_hint'")
    
    if action_type == 'select':
        # Select actions should have option_index
        if 'option_index' in action and not isinstance(action['option_index'], int):
            errors.append(f"{prefix}: 'select' action 'option_index' must be an integer")
    
    if action_type == 'scroll':
        # Scroll actions should have coordinates
        if 'coordinates' in action:
            coords = action['coordinates']
            if not isinstance(coords, dict):
                errors.append(f"{prefix}: 'scroll' action 'coordinates' must be a dictionary")
            elif 'x' not in coords or 'y' not in coords:
                errors.append(f"{prefix}: 'scroll' action 'coordinates' must have 'x' and 'y'")
    
    # Validate is_intentional (optional, but if present should be boolean)
    if 'is_intentional' in action and not isinstance(action['is_intentional'], bool):
        errors.append(f"{prefix}: 'is_intentional' must be a boolean")
    
    # Validate user_intent (optional, but if present should be string)
    if 'user_intent' in action and not isinstance(action['user_intent'], str):
        errors.append(f"{prefix}: 'user_intent' must be a string")
    
    # Validate context (optional, but if present should be string)
    if 'context' in action and not isinstance(action['context'], str):
        errors.append(f"{prefix}: 'context' must be a string")
    
    return errors


def _validate_url_format(url: str) -> bool:
    """Basic URL format validation"""
    if not url:
        return False
    
    # Basic URL pattern
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)?$', re.IGNORECASE)
    
    return bool(url_pattern.match(url))


def normalize_trajectory_structure(structure: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and fix common issues in LLM-generated trajectory structure.
    
    Args:
        structure: Dictionary from LLM
        
    Returns:
        Normalized dictionary with fixes applied
    """
    normalized = structure.copy()
    
    # Ensure actions is a list
    if 'actions' not in normalized:
        normalized['actions'] = []
    elif not isinstance(normalized['actions'], list):
        normalized['actions'] = []
    
    # Normalize each action
    normalized['actions'] = [normalize_action_structure(action) for action in normalized['actions']]
    
    # Ensure domain is string if present
    if 'domain' in normalized and not isinstance(normalized['domain'], str):
        normalized['domain'] = str(normalized['domain'])
    
    # Ensure goal_achieved is boolean if present
    if 'goal_achieved' in normalized and not isinstance(normalized['goal_achieved'], bool):
        normalized['goal_achieved'] = bool(normalized['goal_achieved'])
    
    return normalized


def normalize_action_structure(action: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single action structure.
    
    Args:
        action: Action dictionary
        
    Returns:
        Normalized action dictionary
    """
    normalized = action.copy()
    
    # Ensure action_type is string
    if 'action_type' in normalized:
        normalized['action_type'] = str(normalized['action_type']).lower()
    
    # Ensure URL is string (add default if missing)
    if 'url' not in normalized or not normalized['url']:
        normalized['url'] = 'https://example.com'
    else:
        normalized['url'] = str(normalized['url'])
    
    # Ensure page_title is string (add default if missing)
    if 'page_title' not in normalized or not normalized['page_title']:
        normalized['page_title'] = 'Page'
    else:
        normalized['page_title'] = str(normalized['page_title'])
    
    # Ensure is_intentional is boolean (default True)
    if 'is_intentional' not in normalized:
        normalized['is_intentional'] = True
    else:
        normalized['is_intentional'] = bool(normalized['is_intentional'])
    
    # Ensure element_type is string if present
    if 'element_type' in normalized:
        normalized['element_type'] = str(normalized['element_type'])
    
    # Ensure user_intent is string if present
    if 'user_intent' in normalized and normalized['user_intent'] is not None:
        normalized['user_intent'] = str(normalized['user_intent'])
    
    # Ensure context is string if present
    if 'context' in normalized and normalized['context'] is not None:
        normalized['context'] = str(normalized['context'])
    
    # Ensure option_index is integer if present
    if 'option_index' in normalized and normalized['option_index'] is not None:
        try:
            normalized['option_index'] = int(normalized['option_index'])
        except (ValueError, TypeError):
            # Remove invalid option_index
            del normalized['option_index']
    
    return normalized

