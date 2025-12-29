"""
Utility functions for trajectory generation.

This module provides helper functions for:
- ID generation
- Timestamp calculations
- Temporal realism (typing speed, decision time, reading time)
- User behavior patterns
- Fallback text generation (with optional faker support)
"""

import random
import time
from typing import List, Tuple, Any

# Faker is optional - only used for fallback text generation
try:
    from faker import Faker
    fake = Faker()
except ImportError:
    fake = None

# Domains for different workflow types
DOMAINS = {
    'e_commerce': ['example-store.com', 'shop-example.com', 'retail-test.com'],
    'form_filling': ['signup-example.com', 'contact-example.com', 'form-test.com'],
    'research': ['search-example.com', 'wiki-example.com', 'article-test.com']
}

# User behavior types
USER_TYPES = ['power_user', 'casual', 'first_time']
# Device types
DEVICE_TYPES = ['desktop', 'mobile', 'tablet']
# Browser types
BROWSER_TYPES = ['chrome', 'firefox', 'safari']


def generate_id(prefix: str = 'id') -> str:
    """
    Generate a unique identifier with a given prefix.
    
    Args:
        prefix: Prefix for the ID (e.g., 'traj', 'session', 'action')
    
    Returns:
        String ID in format '{prefix}_{random_6_digit_number}'
    
    Example:
        >>> generate_id('traj')
        'traj_123456'
    """
    return f"{prefix}_{random.randint(100000, 999999)}"


def get_timestamp() -> float:
    """
    Get current Unix timestamp in milliseconds.
    
    Returns:
        Current time as Unix timestamp in milliseconds (float)
    """
    return time.time() * 1000


def get_timestamp_offset(base: float, seconds: float) -> float:
    """
    Calculate a timestamp offset from a base timestamp.
    
    Args:
        base: Base timestamp in milliseconds
        seconds: Offset in seconds (can be negative)
    
    Returns:
        New timestamp in milliseconds (base + seconds * 1000)
    
    Example:
        >>> base = 1704067200000
        >>> get_timestamp_offset(base, 5.5)
        1704067205500.0
    """
    return base + (seconds * 1000)


def random_choice_weighted(choices: List[Tuple[Any, float]]) -> Any:
    """
    Choose an item from a list of (item, weight) tuples based on weights.
    
    Args:
        choices: List of tuples (item, weight) where weight is a probability
    
    Returns:
        Randomly selected item based on weights
    
    Example:
        >>> choices = [('a', 0.5), ('b', 0.3), ('c', 0.2)]
        >>> random_choice_weighted(choices)  # 'a' has 50% chance
    """
    items, weights = zip(*choices)
    return random.choices(items, weights=weights)[0]


def get_trajectory_length_distribution() -> int:
    """
    Get trajectory length based on realistic distribution.
    
    Distribution:
        - 3-4 actions: 20% (quick tasks)
        - 5-6 actions: 40% (typical workflows)
        - 7-8 actions: 30% (complex tasks)
        - 9-10 actions: 10% (multi-step processes)
    
    Returns:
        Random trajectory length between 3 and 10 actions
    """
    rand = random.random()
    if rand < 0.2:
        return random.randint(3, 4)
    elif rand < 0.6:
        return random.randint(5, 6)
    elif rand < 0.9:
        return random.randint(7, 8)
    else:
        return random.randint(9, 10)


def get_typing_speed_wpm(user_type: str) -> float:
    """
    Get realistic typing speed (words per minute) based on user type.
    
    Args:
        user_type: One of 'power_user', 'casual', or 'first_time'
    
    Returns:
        Random WPM value within realistic range for user type:
        - power_user: 60-80 WPM
        - casual: 30-50 WPM
        - first_time: 20-40 WPM
    """
    if user_type == 'power_user':
        return random.uniform(60, 80)
    elif user_type == 'casual':
        return random.uniform(30, 50)
    else:  # first_time
        return random.uniform(20, 40)


def calculate_typing_time(text: str, wpm: float) -> float:
    """
    Calculate time required to type text based on words-per-minute speed.
    
    Args:
        text: Text to be typed
        wpm: Typing speed in words per minute
    
    Returns:
        Time in seconds required to type the text
    """
    words = len(text.split())
    minutes = words / wpm
    return minutes * 60  # Return seconds


def get_decision_time(user_type: str, is_important: bool = False) -> float:
    """
    Get realistic decision time before taking an action.
    
    Simulates the time a user takes to think before clicking, typing, etc.
    Important actions (like checkout, submit) take longer.
    
    Args:
        user_type: One of 'power_user', 'casual', or 'first_time'
        is_important: Whether this is an important action (default: False)
    
    Returns:
        Random decision time in seconds:
        - power_user: 0.5-2.0s (1.5-4.0s if important)
        - casual: 1.0-4.0s (2.0-8.0s if important)
        - first_time: 2.0-8.0s (3.0-16.0s if important)
    """
    base_time = {
        'power_user': (0.5, 2.0),
        'casual': (1.0, 4.0),
        'first_time': (2.0, 8.0)
    }[user_type]
    
    if is_important:
        base_time = (base_time[0] * 1.5, base_time[1] * 2.0)
    
    return random.uniform(*base_time)


def get_reading_time(user_type: str, content_complexity: str = 'medium') -> float:
    """
    Get realistic reading time based on user type and content complexity.
    
    Args:
        user_type: One of 'power_user', 'casual', or 'first_time'
        content_complexity: 'low', 'medium', or 'high' (default: 'medium')
    
    Returns:
        Random reading time in seconds (multiplied by complexity factor)
    """
    base_time = {
        'power_user': (5, 15),
        'casual': (10, 30),
        'first_time': (15, 45)
    }[user_type]
    
    multiplier = {'low': 0.7, 'medium': 1.0, 'high': 1.5}[content_complexity]
    return random.uniform(base_time[0] * multiplier, base_time[1] * multiplier)


def generate_email() -> str:
    """
    Generate a fake email address for testing.
    
    Uses faker if available, otherwise generates a simple test email.
    
    Returns:
        Fake email address (e.g., 'user_1234@test.example.com')
    """
    if fake:
        return fake.email().replace('@', '@test.')
    return f"user_{random.randint(1000, 9999)}@test.example.com"


def generate_name() -> str:
    """
    Generate a fake name for testing.
    
    Uses faker if available, otherwise generates a simple name.
    
    Returns:
        Fake name (e.g., 'John Doe' or 'User 42')
    """
    if fake:
        return fake.name()
    return f"User {random.randint(1, 100)}"


def generate_text(length: int = 50) -> str:
    """
    Generate fake text of specified length.
    
    Args:
        length: Desired text length in characters (default: 50)
    
    Returns:
        Fake text string of approximately the specified length
    """
    if fake:
        return fake.text(max_nb_chars=length)
    return f"Sample text with {length} characters. " * (length // 20)


def get_domain(workflow_type: str) -> str:
    """
    Get a random domain name for a given workflow type.
    
    Args:
        workflow_type: One of 'e_commerce', 'form_filling', or 'research'
    
    Returns:
        Random domain from the appropriate list, or 'example.com' if unknown
    """
    return random.choice(DOMAINS.get(workflow_type, ['example.com']))

