"""
Data schema definitions for browser trajectories.

This module defines the core data structures used throughout the trajectory generation
pipeline: BrowserAction (single interaction) and Trajectory (sequence of interactions).

These schemas are designed to capture:
- User interactions (clicks, typing, navigation, etc.)
- Temporal relationships between actions
- User intent and context signals
- DOM state and element properties
- Metadata for ML model training
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class BrowserAction:
    """
    Represents a single browser interaction event.
    
    This is the atomic unit of a trajectory, capturing one user action such as
    clicking a button, typing text, or navigating to a URL. All fields are designed
    to support downstream ML tasks like next-action prediction and element selection.
    
    Attributes:
        timestamp: Unix timestamp in milliseconds when the action occurred
        action_type: Type of action (e.g., 'click', 'type', 'navigate', 'scroll')
        action_id: Unique identifier for this action within a trajectory
        element_type: HTML element type (e.g., 'button', 'input', 'link')
        element_selector: CSS selector or XPath to locate the element
        url: Current page URL when action occurred
        page_title: Title of the current page
        element_text: Visible text content of the element (optional)
        element_id: HTML id attribute of the element (optional)
        element_classes: List of CSS classes on the element (optional)
        value: Text value for 'type' actions (optional)
        option_index: Selected option index for 'select' actions (optional)
        coordinates: X,Y coordinates for 'click' or 'scroll' actions (optional)
        is_intentional: Whether this action was goal-directed (True) or exploratory
        confidence: Model confidence score for this action (0.0 to 1.0)
        user_intent: Semantic intent label (e.g., 'add_to_cart', 'search', 'submit_form')
        dom_snapshot: Lightweight DOM representation at action time (optional)
        element_visible: Whether element was visible when action attempted
        element_clickable: Whether element was clickable when action attempted
        session_id: Browser session identifier
        tab_id: Browser tab identifier
        frame_id: Frame identifier for iframe interactions (optional)
    
    Examples:
        >>> action = BrowserAction(
        ...     timestamp=1704067200000,
        ...     action_type='click',
        ...     action_id='action_001',
        ...     element_type='button',
        ...     element_selector='#add-to-cart',
        ...     url='https://example.com/product',
        ...     page_title='Product Page',
        ...     user_intent='add_to_cart',
        ...     is_intentional=True
        ... )
    """
    
    # Temporal
    timestamp: float  # Unix timestamp in milliseconds
    
    # Action Identity
    action_type: str  # One of: click, type, navigate, scroll, hover, select, submit, back, forward, refresh, wait, drag_drop
    action_id: str  # Unique identifier within trajectory (e.g., 'action_001')
    
    # Target Element
    element_type: str  # HTML element type (button, input, link, select, etc.)
    element_selector: str  # CSS selector or XPath to locate element
    
    # Context (required fields before optional)
    url: str  # Current page URL
    page_title: str  # Page title
    
    # Optional Target Element fields
    element_text: Optional[str] = None  # Visible text content
    element_id: Optional[str] = None  # HTML id attribute
    element_classes: Optional[List[str]] = None  # CSS classes list
    
    # Action Parameters
    value: Optional[str] = None  # Text value for 'type' actions
    option_index: Optional[int] = None  # Selected option for 'select' actions
    coordinates: Optional[Dict[str, int]] = None  # {'x': int, 'y': int} for click/scroll
    
    # Intent Signals
    is_intentional: bool = True  # True for goal-directed actions, False for exploration
    confidence: float = 1.0  # Model confidence (0.0 to 1.0)
    user_intent: Optional[str] = None  # Semantic intent label
    
    # DOM State
    dom_snapshot: Optional[str] = None  # Lightweight DOM representation
    element_visible: bool = True  # Element visibility state
    element_clickable: bool = True  # Element clickability state
    
    # Metadata
    session_id: str = ""  # Browser session identifier
    tab_id: str = ""  # Browser tab identifier
    frame_id: Optional[str] = None  # Frame identifier for iframe interactions
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert action to dictionary for serialization.
        
        Returns:
            Dictionary representation of the action, suitable for JSON serialization
        """
        return asdict(self)


@dataclass
class Trajectory:
    """
    Represents a sequence of browser interactions forming a complete user trajectory.
    
    A trajectory captures a user's complete workflow session, from initial navigation
    to goal completion (or abandonment). It includes all actions, temporal information,
    user profile data, and goal achievement status.
    
    Attributes:
        trajectory_id: Unique identifier for this trajectory
        session_id: Browser session identifier (shared across trajectories in same session)
        actions: Ordered list of BrowserAction objects (3-10 actions per trajectory)
        workflow_type: Type of workflow (e.g., 'e_commerce', 'form_filling', 'research')
        domain: Website domain (e.g., 'example-store.com')
        start_time: Unix timestamp in milliseconds when trajectory started
        end_time: Unix timestamp in milliseconds when trajectory ended
        duration: Total duration in seconds (end_time - start_time) / 1000
        user_type: User behavior pattern ('power_user', 'casual', 'first_time')
        device_type: Device type ('desktop', 'mobile', 'tablet')
        browser_type: Browser type ('chrome', 'firefox', 'safari', 'edge')
        goal: High-level goal description (e.g., 'purchase_product', 'submit_contact_form')
        goal_achieved: Whether the user successfully completed their goal
        success_indicators: List of indicators that goal was achieved (optional)
    
    Properties:
        action_count: Number of actions in the trajectory
        avg_action_interval: Average time between actions in seconds
        backtrack_count: Number of back/forward navigations
        error_count: Number of actions where element was not visible or clickable
    
    Examples:
        >>> trajectory = Trajectory(
        ...     trajectory_id='traj_001',
        ...     session_id='session_abc',
        ...     actions=[action1, action2, action3],
        ...     workflow_type='e_commerce',
        ...     domain='example-store.com',
        ...     start_time=1704067200000,
        ...     end_time=1704067265000,
        ...     duration=65.0,
        ...     user_type='casual',
        ...     device_type='desktop',
        ...     browser_type='chrome',
        ...     goal='purchase_product',
        ...     goal_achieved=True
        ... )
    """
    
    # Identity
    trajectory_id: str  # Unique identifier (e.g., 'traj_001')
    session_id: str  # Browser session identifier
    
    # Actions
    actions: List[BrowserAction]  # Ordered sequence of 3-10 actions
    
    # Metadata
    workflow_type: str  # 'e_commerce', 'form_filling', or 'research'
    domain: str  # Website domain
    start_time: float  # Unix timestamp in milliseconds
    end_time: float  # Unix timestamp in milliseconds
    duration: float  # Duration in seconds
    
    # User Profile
    user_type: str  # 'power_user', 'casual', or 'first_time'
    device_type: str  # 'desktop', 'mobile', or 'tablet'
    browser_type: str  # 'chrome', 'firefox', 'safari', or 'edge'
    
    # Goal Information
    goal: Optional[str] = None  # High-level goal description
    goal_achieved: bool = False  # Whether goal was successfully completed
    success_indicators: List[str] = field(default_factory=list)  # Indicators of success
    
    @property
    def action_count(self) -> int:
        """
        Get the number of actions in this trajectory.
        
        Returns:
            Number of actions (always between 3 and 10)
        """
        return len(self.actions)
    
    @property
    def avg_action_interval(self) -> float:
        """
        Calculate average time between consecutive actions.
        
        Returns:
            Average interval in seconds, or 0.0 if fewer than 2 actions
        """
        if len(self.actions) < 2:
            return 0.0
        intervals = [
            self.actions[i].timestamp - self.actions[i-1].timestamp 
            for i in range(1, len(self.actions))
        ]
        return (sum(intervals) / len(intervals)) / 1000.0  # Convert ms to seconds
    
    @property
    def backtrack_count(self) -> int:
        """
        Count the number of back/forward navigations in the trajectory.
        
        Returns:
            Number of 'back' or 'forward' actions
        """
        return sum(1 for a in self.actions if a.action_type in ['back', 'forward'])
    
    @property
    def error_count(self) -> int:
        """
        Count the number of failed actions (element not visible or not clickable).
        
        Returns:
            Number of actions where element_visible=False or element_clickable=False
        """
        return sum(1 for a in self.actions 
                  if not a.element_visible or not a.element_clickable)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert trajectory to dictionary for serialization.
        
        Includes computed properties (action_count, avg_action_interval, etc.)
        in addition to all dataclass fields.
        
        Returns:
            Dictionary representation suitable for JSON serialization
        """
        result = asdict(self)
        # Add computed properties
        result['action_count'] = self.action_count
        result['avg_action_interval'] = self.avg_action_interval
        result['backtrack_count'] = self.backtrack_count
        result['error_count'] = self.error_count
        return result

