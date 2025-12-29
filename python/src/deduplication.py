"""Trajectory deduplication to ensure dataset diversity"""

import hashlib
import json
from typing import List, Dict, Tuple, Any
from src.schema import Trajectory, BrowserAction
from src.logging_config import get_logger

logger = get_logger('deduplication')


def compute_trajectory_fingerprint(trajectory: Trajectory, normalize: bool = True) -> str:
    """
    Compute a content-based fingerprint for a trajectory.
    
    The fingerprint is based on the action sequence (action types, elements, URLs)
    but excludes temporal and ID information to detect content duplicates.
    
    Args:
        trajectory: Trajectory to fingerprint
        normalize: If True, normalize values (lowercase, trim) for better matching
        
    Returns:
        SHA256 hash string of the trajectory content
    """
    # Extract content features (exclude IDs, timestamps, metadata)
    content = {
        "workflow_type": trajectory.workflow_type,
        "goal": trajectory.goal,
        "actions": []
    }
    
    for action in trajectory.actions:
        action_content = {
            "action_type": action.action_type,
            "element_type": action.element_type,
            "element_selector": action.element_selector,
            "url": action.url,
            "page_title": action.page_title,
            "value": action.value,  # Include value for type actions
            "option_index": action.option_index,  # Include for select actions
        }
        
        # Normalize if requested
        if normalize:
            action_content["url"] = _normalize_url(action_content["url"])
            action_content["page_title"] = _normalize_text(action_content.get("page_title", ""))
            action_content["element_selector"] = _normalize_selector(action_content["element_selector"])
            if action_content["value"]:
                action_content["value"] = _normalize_text(action_content["value"])
        
        content["actions"].append(action_content)
    
    # Create JSON string and hash it
    content_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
    fingerprint = hashlib.sha256(content_str.encode('utf-8')).hexdigest()
    
    return fingerprint


def _normalize_url(url: str) -> str:
    """Normalize URL for comparison (remove protocol, trailing slashes, etc.)"""
    if not url:
        return ""
    url = url.lower().strip()
    # Remove protocol
    url = url.replace("https://", "").replace("http://", "")
    # Remove trailing slash
    url = url.rstrip("/")
    return url


def _normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    if not text:
        return ""
    return text.lower().strip()


def _normalize_selector(selector: str) -> str:
    """Normalize CSS selector for comparison"""
    if not selector:
        return ""
    # Lowercase and remove extra whitespace
    return " ".join(selector.lower().strip().split())


def detect_exact_duplicates(trajectories: List[Trajectory]) -> Tuple[List[Trajectory], List[Dict[str, Any]]]:
    """
    Detect and remove exact duplicate trajectories.
    
    Args:
        trajectories: List of trajectories to check
        
    Returns:
        Tuple of (deduplicated_trajectories, duplicate_info_list)
        duplicate_info contains trajectory_id and fingerprint for each duplicate
    """
    seen_fingerprints: Dict[str, Trajectory] = {}
    deduplicated: List[Trajectory] = []
    duplicates_info: List[Dict[str, any]] = []
    
    for trajectory in trajectories:
        fingerprint = compute_trajectory_fingerprint(trajectory)
        
        if fingerprint in seen_fingerprints:
            # This is a duplicate
            original = seen_fingerprints[fingerprint]
            duplicates_info.append({
                "trajectory_id": trajectory.trajectory_id,
                "duplicate_of": original.trajectory_id,
                "fingerprint": fingerprint,
                "workflow_type": trajectory.workflow_type,
                "goal": trajectory.goal
            })
        else:
            # First occurrence - keep it
            seen_fingerprints[fingerprint] = trajectory
            deduplicated.append(trajectory)
    
    return deduplicated, duplicates_info


# Near-duplicate detection functions removed - see SPEC.md for future enhancement plans
# Helper function for URL path extraction (kept for potential future use in near-duplicate detection)
def _extract_url_path(url: str) -> str:
    """Extract path from URL for comparison (for future near-duplicate detection)"""
    if not url:
        return ""
    try:
        # Remove protocol and domain, keep path
        url = url.lower().strip()
        url = url.replace("https://", "").replace("http://", "")
        if "/" in url:
            path = "/" + "/".join(url.split("/")[1:])  # Keep path and query
            return path.rstrip("/")
        return ""
    except Exception:
        return url


def deduplicate_trajectories(
    trajectories: List[Trajectory]
) -> Tuple[List[Trajectory], Dict[str, Any]]:
    """
    Main deduplication function that removes exact duplicate trajectories.
    
    Args:
        trajectories: List of trajectories to deduplicate
    
    Returns:
        Tuple of (deduplicated_trajectories, stats_dict)
        stats_dict contains counts and duplicate info
    """
    stats = {
        "original_count": len(trajectories),
        "exact_duplicates_count": 0,
        "near_duplicates_count": 0,  # Always 0 (feature removed)
        "final_count": 0,
        "exact_duplicates_info": [],
        "near_duplicates_info": []  # Always empty (feature removed)
    }
    
    # Remove exact duplicates
    logger.debug(f"Detecting exact duplicates in {len(trajectories)} trajectories...")
    deduplicated, exact_duplicates = detect_exact_duplicates(trajectories)
    stats["exact_duplicates_count"] = len(exact_duplicates)
    stats["exact_duplicates_info"] = exact_duplicates
    stats["final_count"] = len(deduplicated)
    
    if len(exact_duplicates) > 0:
        logger.debug(f"Found {len(exact_duplicates)} exact duplicates")
        for dup in exact_duplicates[:5]:  # Log first 5
            logger.debug(f"  Duplicate: {dup['trajectory_id']} -> {dup['duplicate_of']}")
    
    logger.debug(f"Deduplication complete: {stats['original_count']} -> {stats['final_count']} trajectories")
    return deduplicated, stats

