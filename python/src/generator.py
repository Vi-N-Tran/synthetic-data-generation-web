"""Trajectory generator using LLM for realistic data generation"""

import random
import time
from typing import Dict, List, Optional, Any
from src.schema import BrowserAction, Trajectory
from src.llm_generator import LLMDataGenerator
from src.actions import (
    create_navigate_action, create_click_action, create_type_action,
    create_scroll_action, create_select_action, create_submit_action,
    create_wait_action
)
from src.utils import (
    generate_id, get_timestamp, get_timestamp_offset,
    get_trajectory_length_distribution, get_decision_time,
    get_reading_time, calculate_typing_time, get_typing_speed_wpm,
    get_domain, USER_TYPES, DEVICE_TYPES, BROWSER_TYPES
)


class TrajectoryGenerator:
    """Generates synthetic trajectories using LLM"""
    
    # Workflow types and goals
    WORKFLOW_GOALS = {
        'e_commerce': [
            'purchase_product',
            'browse_products',
            'add_to_cart',
            'compare_products'
        ],
        'form_filling': [
            'submit_contact_form',
            'create_account',
            'subscribe_newsletter',
            'request_quote'
        ],
        'research': [
            'find_information',
            'read_article',
            'compare_options',
            'learn_topic'
        ]
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, api_key: Optional[str] = None, use_openrouter: bool = False):
        """
        Initialize the generator.
        
        Args:
            config: Configuration dict for generation parameters
            api_key: API key (OpenAI or OpenRouter)
            use_openrouter: If True, use OpenRouter API instead of OpenAI
        """
        self.config = config or {}
        self.llm_generator = LLMDataGenerator(api_key=api_key, use_openrouter=use_openrouter)
        
        # Set random seed if provided
        seed = self.config.get('generator', {}).get('seed')
        if seed is not None:
            random.seed(seed)
    
    def _select_workflow_type(self) -> str:
        """Select workflow type based on distribution"""
        distribution = self.config.get('generator', {}).get(
            'workflow_distribution',
            {'e_commerce': 0.4, 'form_filling': 0.3, 'research': 0.3}
        )
        workflow_types = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(workflow_types, weights=weights)[0]
    
    def _select_goal(self, workflow_type: str) -> str:
        """Select a goal for the workflow type"""
        goals = self.WORKFLOW_GOALS.get(workflow_type, ['complete_task'])
        return random.choice(goals)
    
    def _convert_action_to_browser_action(
        self,
        action_data: Dict[str, Any],
        timestamp: float,
        session_id: str,
        tab_id: str,
        action_id: str,
        workflow_type: str,
        user_type: str
    ) -> BrowserAction:
        """Convert LLM-generated action data to BrowserAction object"""
        action_type = action_data['action_type']
        url = action_data.get('url', 'https://example.com')
        page_title = action_data.get('page_title', 'Page')
        context = action_data.get('context', '')
        user_intent = action_data.get('user_intent')
        is_intentional = action_data.get('is_intentional', True)
        
        # Calculate next timestamp based on action type and user behavior
        if action_type == 'navigate':
            time_offset = random.uniform(0.3, 3.0)  # Page load time
        elif action_type == 'type':
            value = action_data.get('value') or action_data.get('value_hint', 'text')
            wpm = get_typing_speed_wpm(user_type)
            time_offset = calculate_typing_time(value, wpm) + get_decision_time(user_type)
        elif action_type == 'scroll':
            time_offset = random.uniform(0.5, 2.0)
        elif action_type == 'wait':
            time_offset = action_data.get('duration', get_reading_time(user_type))
        else:
            time_offset = get_decision_time(user_type, action_type in ['submit', 'click'] and 'cart' in context.lower())
        
        next_timestamp = get_timestamp_offset(timestamp, time_offset)
        
        # Generate action based on type
        if action_type == 'navigate':
            return create_navigate_action(
                timestamp=next_timestamp,
                url=url,
                page_title=page_title,
                session_id=session_id,
                tab_id=tab_id,
                action_id=action_id,
                is_intentional=is_intentional,
                llm_generator=self.llm_generator
            )
        elif action_type == 'click':
            element_type = action_data.get('element_type', 'button')
            return create_click_action(
                timestamp=next_timestamp,
                element_type=element_type,
                url=url,
                page_title=page_title,
                session_id=session_id,
                tab_id=tab_id,
                action_id=action_id,
                workflow_type=workflow_type,
                context=context,
                user_intent=user_intent,
                is_intentional=is_intentional,
                llm_generator=self.llm_generator
            )
        elif action_type == 'type':
            field_type = action_data.get('field_type', 'text')
            value = action_data.get('value') or action_data.get('value_hint', '')
            return create_type_action(
                timestamp=next_timestamp,
                url=url,
                page_title=page_title,
                session_id=session_id,
                tab_id=tab_id,
                action_id=action_id,
                workflow_type=workflow_type,
                field_type=field_type,
                context=context,
                user_intent=user_intent,
                is_intentional=is_intentional,
                llm_generator=self.llm_generator,
                value=value
            )
        elif action_type == 'scroll':
            return create_scroll_action(
                timestamp=next_timestamp,
                url=url,
                page_title=page_title,
                session_id=session_id,
                tab_id=tab_id,
                action_id=action_id,
                coordinates={'x': 0, 'y': random.randint(100, 500)},
                is_intentional=is_intentional
            )
        elif action_type == 'select':
            option_index = action_data.get('option_index', random.randint(0, 4))
            return create_select_action(
                timestamp=next_timestamp,
                url=url,
                page_title=page_title,
                session_id=session_id,
                tab_id=tab_id,
                action_id=action_id,
                workflow_type=workflow_type,
                context=context,
                option_index=option_index,
                user_intent=user_intent,
                is_intentional=is_intentional,
                llm_generator=self.llm_generator
            )
        elif action_type == 'submit':
            return create_submit_action(
                timestamp=next_timestamp,
                url=url,
                page_title=page_title,
                session_id=session_id,
                tab_id=tab_id,
                action_id=action_id,
                user_intent=user_intent,
                is_intentional=is_intentional
            )
        elif action_type == 'wait':
            duration = action_data.get('duration', get_reading_time(user_type))
            return create_wait_action(
                timestamp=next_timestamp,
                url=url,
                page_title=page_title,
                session_id=session_id,
                tab_id=tab_id,
                action_id=action_id,
                duration=duration,
                is_intentional=is_intentional
            )
        else:
            # Fallback for other action types
            return BrowserAction(
                timestamp=next_timestamp,
                action_type=action_type,
                action_id=action_id,
                element_type=action_data.get('element_type', ''),
                element_selector=action_data.get('selector', ''),
                url=url,
                page_title=page_title,
                session_id=session_id,
                tab_id=tab_id,
                is_intentional=is_intentional,
                user_intent=user_intent
            )
    
    def generate_trajectory(self, **kwargs) -> Trajectory:
        """
        Generate a single trajectory using LLM.
        
        Returns:
            Generated Trajectory object
        """
        # Select workflow type and goal
        workflow_type = kwargs.get('workflow_type') or self._select_workflow_type()
        goal = kwargs.get('goal') or self._select_goal(workflow_type)
        
        # Select user characteristics
        user_type = kwargs.get('user_type') or random.choice(USER_TYPES)
        device_type = kwargs.get('device_type') or random.choice(DEVICE_TYPES)
        browser_type = kwargs.get('browser_type') or random.choice(BROWSER_TYPES)
        
        # Determine number of actions
        min_actions = self.config.get('generator', {}).get('min_actions', 3)
        max_actions = self.config.get('generator', {}).get('max_actions', 10)
        num_actions = kwargs.get('num_actions') or random.randint(min_actions, max_actions)
        
        # Generate trajectory structure using LLM
        trajectory_structure = self.llm_generator.generate_trajectory_structure(
            workflow_type=workflow_type,
            goal=goal,
            user_type=user_type,
            num_actions=num_actions
        )
        
        # Validate structure (additional check after LLM validation)
        if not trajectory_structure or 'actions' not in trajectory_structure:
            raise ValueError("LLM returned invalid trajectory structure: missing 'actions' field")
        
        # Extract domain and goal info
        domain = trajectory_structure.get('domain') or get_domain(workflow_type)
        goal_achieved = trajectory_structure.get('goal_achieved', True)
        actions_data = trajectory_structure.get('actions', [])
        
        # Ensure we have at least some actions
        if not actions_data or len(actions_data) == 0:
            raise ValueError("LLM returned trajectory structure with no actions")
        
        # Limit to num_actions to respect configuration
        if len(actions_data) > num_actions:
            actions_data = actions_data[:num_actions]
            print(f"Warning: LLM returned {len(actions_data)} actions, limiting to {num_actions}")
        
        # Generate IDs
        trajectory_id = generate_id('traj')
        session_id = generate_id('session')
        tab_id = generate_id('tab')
        
        # Convert actions to BrowserAction objects
        start_time = get_timestamp()
        timestamp = start_time
        browser_actions = []
        
        for i, action_data in enumerate(actions_data):
            try:
                action_id = f"action_{i+1:03d}"
                browser_action = self._convert_action_to_browser_action(
                    action_data=action_data,
                    timestamp=timestamp,
                    session_id=session_id,
                    tab_id=tab_id,
                    action_id=action_id,
                    workflow_type=workflow_type,
                    user_type=user_type
                )
                browser_actions.append(browser_action)
                timestamp = browser_action.timestamp  # Use the timestamp from the action
            except Exception as e:
                print(f"Warning: Failed to convert action {i+1}: {e}. Skipping action.")
                continue
        
        # Ensure we have at least 3 actions (minimum required)
        if len(browser_actions) < 3:
            raise ValueError(f"Too few valid actions after conversion: {len(browser_actions)}. Need at least 3.")
        
        end_time = timestamp
        duration = (end_time - start_time) / 1000.0  # Convert to seconds
        
        # Create trajectory
        trajectory = Trajectory(
            trajectory_id=trajectory_id,
            session_id=session_id,
            actions=browser_actions,
            workflow_type=workflow_type,
            domain=domain,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            user_type=user_type,
            device_type=device_type,
            browser_type=browser_type,
            goal=goal,
            goal_achieved=goal_achieved,
            success_indicators=[]
        )
        
        return trajectory
    
    def generate_dataset(self, n_trajectories: int) -> List[Trajectory]:
        """
        Generate multiple trajectories to form a dataset.
        
        Args:
            n_trajectories: Number of trajectories to generate
            
        Returns:
            List of generated trajectories
        """
        trajectories = []
        for i in range(n_trajectories):
            try:
                trajectory = self.generate_trajectory()
                trajectories.append(trajectory)
                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{n_trajectories} trajectories...")
            except Exception as e:
                print(f"Warning: Failed to generate trajectory {i+1}: {e}")
                continue
        return trajectories

