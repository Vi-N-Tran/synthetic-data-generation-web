"""Utility functions for trajectory generation"""

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

USER_TYPES = ['power_user', 'casual', 'first_time']
DEVICE_TYPES = ['desktop', 'mobile', 'tablet']
BROWSER_TYPES = ['chrome', 'firefox', 'safari']


def generate_id(prefix: str = 'id') -> str:
    """Generate a unique identifier"""
    return f"{prefix}_{random.randint(100000, 999999)}"


def get_timestamp() -> float:
    """Get current timestamp in milliseconds"""
    return time.time() * 1000


def get_timestamp_offset(base: float, seconds: float) -> float:
    """Get timestamp offset from base by seconds"""
    return base + (seconds * 1000)


def random_choice_weighted(choices: List[Tuple[Any, float]]) -> Any:
    """Choose an item from choices based on weights"""
    items, weights = zip(*choices)
    return random.choices(items, weights=weights)[0]


def get_trajectory_length_distribution() -> int:
    """Get trajectory length based on distribution:
    - 3-4 actions: 20%
    - 5-6 actions: 40%
    - 7-8 actions: 30%
    - 9-10 actions: 10%
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
    """Get typing speed based on user type"""
    if user_type == 'power_user':
        return random.uniform(60, 80)
    elif user_type == 'casual':
        return random.uniform(30, 50)
    else:  # first_time
        return random.uniform(20, 40)


def calculate_typing_time(text: str, wpm: float) -> float:
    """Calculate time to type text based on WPM"""
    words = len(text.split())
    minutes = words / wpm
    return minutes * 60  # Return seconds


def get_decision_time(user_type: str, is_important: bool = False) -> float:
    """Get decision time before action in seconds"""
    base_time = {
        'power_user': (0.5, 2.0),
        'casual': (1.0, 4.0),
        'first_time': (2.0, 8.0)
    }[user_type]
    
    if is_important:
        base_time = (base_time[0] * 1.5, base_time[1] * 2.0)
    
    return random.uniform(*base_time)


def get_reading_time(user_type: str, content_complexity: str = 'medium') -> float:
    """Get reading time in seconds"""
    base_time = {
        'power_user': (5, 15),
        'casual': (10, 30),
        'first_time': (15, 45)
    }[user_type]
    
    multiplier = {'low': 0.7, 'medium': 1.0, 'high': 1.5}[content_complexity]
    return random.uniform(base_time[0] * multiplier, base_time[1] * multiplier)


def generate_email() -> str:
    """Generate a fake email address"""
    if fake:
        return fake.email().replace('@', '@test.')
    return f"user_{random.randint(1000, 9999)}@test.example.com"


def generate_name() -> str:
    """Generate a fake name"""
    if fake:
        return fake.name()
    return f"User {random.randint(1, 100)}"


def generate_text(length: int = 50) -> str:
    """Generate fake text"""
    if fake:
        return fake.text(max_nb_chars=length)
    return f"Sample text with {length} characters. " * (length // 20)


def get_domain(workflow_type: str) -> str:
    """Get a domain for the workflow type"""
    return random.choice(DOMAINS.get(workflow_type, ['example.com']))

