"""Data schema definitions for browser trajectories"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class BrowserAction:
    """Represents a single browser interaction."""
    
    # Temporal
    timestamp: float
    
    # Action Identity
    action_type: str
    action_id: str
    
    # Target Element
    element_type: str
    element_selector: str
    
    # Context (required fields before optional)
    url: str
    page_title: str
    
    # Optional Target Element fields
    element_text: Optional[str] = None
    element_id: Optional[str] = None
    element_classes: Optional[List[str]] = None
    
    # Action Parameters
    value: Optional[str] = None
    option_index: Optional[int] = None
    coordinates: Optional[Dict[str, int]] = None
    
    # Intent Signals
    is_intentional: bool = True
    confidence: float = 1.0
    user_intent: Optional[str] = None
    
    # DOM State
    dom_snapshot: Optional[str] = None
    element_visible: bool = True
    element_clickable: bool = True
    
    # Metadata
    session_id: str = ""
    tab_id: str = ""
    frame_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary for serialization"""
        return asdict(self)


@dataclass
class Trajectory:
    """Represents a sequence of interactions forming a user trajectory."""
    
    # Identity
    trajectory_id: str
    session_id: str
    
    # Actions
    actions: List[BrowserAction]
    
    # Metadata
    workflow_type: str
    domain: str
    start_time: float
    end_time: float
    duration: float
    
    # User Profile
    user_type: str
    device_type: str
    browser_type: str
    
    # Goal Information
    goal: Optional[str] = None
    goal_achieved: bool = False
    success_indicators: List[str] = field(default_factory=list)
    
    @property
    def action_count(self) -> int:
        """Number of actions in trajectory"""
        return len(self.actions)
    
    @property
    def avg_action_interval(self) -> float:
        """Average time between actions in seconds"""
        if len(self.actions) < 2:
            return 0.0
        intervals = [
            self.actions[i].timestamp - self.actions[i-1].timestamp 
            for i in range(1, len(self.actions))
        ]
        return (sum(intervals) / len(intervals)) / 1000.0  # Convert ms to seconds
    
    @property
    def backtrack_count(self) -> int:
        """Number of back/forward navigations"""
        return sum(1 for a in self.actions if a.action_type in ['back', 'forward'])
    
    @property
    def error_count(self) -> int:
        """Number of failed actions"""
        return sum(1 for a in self.actions 
                  if not a.element_visible or not a.element_clickable)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary for serialization"""
        result = asdict(self)
        # Add computed properties
        result['action_count'] = self.action_count
        result['avg_action_interval'] = self.avg_action_interval
        result['backtrack_count'] = self.backtrack_count
        result['error_count'] = self.error_count
        return result

