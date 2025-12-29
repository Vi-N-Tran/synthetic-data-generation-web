"""
Trajectory generator using LLM for realistic data generation.

This module implements the TrajectoryGenerator class, which orchestrates the
creation of complete browser interaction trajectories. It uses LLMDataGenerator
to create realistic action sequences and then converts them into BrowserAction
and Trajectory objects with proper temporal relationships.
"""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Tuple
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
from src.deduplication import deduplicate_trajectories
from src.logging_config import get_logger

logger = get_logger('generator')


class TrajectoryGenerator:
    """
    Generates synthetic browser interaction trajectories using LLM.
    
    This class orchestrates the generation of complete trajectories by:
    1. Selecting workflow type and goal based on configuration
    2. Using LLM to generate realistic action sequences
    3. Converting LLM output to BrowserAction objects with temporal relationships
    4. Creating Trajectory objects with metadata and computed properties
    5. Applying deduplication if enabled
    
    Attributes:
        config: Configuration dictionary with generation parameters
        llm_generator: LLMDataGenerator instance for generating action data
        WORKFLOW_GOALS: Dictionary mapping workflow types to available goals
    
    Example:
        >>> config = {'generator': {'n_trajectories': 10, 'seed': 42}}
        >>> generator = TrajectoryGenerator(config, api_key='...')
        >>> trajectories, stats = generator.generate_dataset(10)
    """
    
    # Workflow types and their associated goals
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
        Initialize the trajectory generator.
        
        Args:
            config: Configuration dictionary with generation parameters:
                - generator.seed: Random seed for reproducibility
                - generator.workflow_distribution: Dict of workflow type probabilities
                - generator.deduplication.enabled: Whether to deduplicate trajectories
            api_key: API key for OpenAI or OpenRouter (if None, reads from env vars)
            use_openrouter: If True, use OpenRouter API instead of OpenAI
        
        Note:
            If a seed is provided in config, it will be set for random number generation
            to ensure reproducible results.
        """
        self.config = config or {}
        self.llm_generator = LLMDataGenerator(api_key=api_key, use_openrouter=use_openrouter)
        
        # Set random seed if provided
        seed = self.config.get('generator', {}).get('seed')
        if seed is not None:
            random.seed(seed)
    
    def _select_workflow_type(self) -> str:
        """
        Select a workflow type based on configured distribution.
        
        Returns:
            One of 'e_commerce', 'form_filling', or 'research' based on
            weighted random selection from config.workflow_distribution
        
        Note:
            Default distribution is 40% e_commerce, 30% form_filling, 30% research
        """
        distribution = self.config.get('generator', {}).get(
            'workflow_distribution',
            {'e_commerce': 0.4, 'form_filling': 0.3, 'research': 0.3}
        )
        workflow_types = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(workflow_types, weights=weights)[0]
    
    def _select_goal(self, workflow_type: str) -> str:
        """
        Select a random goal for the given workflow type.
        
        Args:
            workflow_type: One of 'e_commerce', 'form_filling', or 'research'
        
        Returns:
            A goal string from WORKFLOW_GOALS, or 'complete_task' if unknown
        """
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
        """
        Convert LLM-generated action data dictionary to BrowserAction object.
        
        This method:
        1. Extracts action data from LLM response
        2. Calculates realistic timestamp offsets based on action type and user behavior
        3. Creates appropriate BrowserAction using action factory functions
        4. Preserves element_visible and element_clickable flags from LLM if provided
        
        Args:
            action_data: Dictionary from LLM containing action structure
            timestamp: Base timestamp in milliseconds
            session_id: Browser session identifier
            tab_id: Browser tab identifier
            action_id: Unique identifier for this action
            workflow_type: Type of workflow (for context)
            user_type: User behavior type (for temporal calculations)
        
        Returns:
            BrowserAction object with proper temporal relationships
        
        Raises:
            ValueError: If action_type is not recognized
        """
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
            # Extract element data from LLM response if provided (avoids additional LLM call)
            selector = action_data.get('element_selector')
            element_text = action_data.get('element_text')
            element_id = action_data.get('element_id')
            element_classes = action_data.get('element_classes')
            # Ensure element_classes is a list if provided
            if element_classes and isinstance(element_classes, str):
                element_classes = [element_classes]
            
            click_action = create_click_action(
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
                llm_generator=self.llm_generator if not selector else None,  # Skip LLM call if data provided
                selector=selector,
                element_text=element_text,
                element_id=element_id,
                element_classes=element_classes
            )
            # Override element_visible and element_clickable if provided by LLM
            if 'element_visible' in action_data:
                click_action.element_visible = bool(action_data['element_visible'])
            if 'element_clickable' in action_data:
                click_action.element_clickable = bool(action_data['element_clickable'])
            return click_action
        elif action_type == 'type':
            field_type = action_data.get('field_type', 'text')
            value = action_data.get('value') or action_data.get('value_hint', '')
            # Extract element data from LLM response if provided (avoids additional LLM call)
            selector = action_data.get('element_selector')
            element_id = action_data.get('element_id')
            
            type_action = create_type_action(
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
                llm_generator=self.llm_generator if not (selector and value) else None,  # Skip LLM call if data provided
                selector=selector,
                value=value,
                element_id=element_id
            )
            # Override element_visible and element_clickable if provided by LLM
            if 'element_visible' in action_data:
                type_action.element_visible = bool(action_data['element_visible'])
            if 'element_clickable' in action_data:
                type_action.element_clickable = bool(action_data['element_clickable'])
            return type_action
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
            # Extract element data from LLM response if provided (avoids additional LLM call)
            selector = action_data.get('element_selector')
            element_id = action_data.get('element_id')
            
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
                llm_generator=self.llm_generator if not selector else None,  # Skip LLM call if data provided
                selector=selector,
                element_id=element_id
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
        Generate a single complete trajectory using LLM.
        
        This is the main trajectory generation method. It:
        1. Selects workflow type and goal
        2. Determines trajectory length (3-10 actions)
        3. Calls LLM to generate action sequence structure
        4. Converts LLM output to BrowserAction objects with temporal relationships
        5. Creates Trajectory object with metadata
        
        Args:
            **kwargs: Optional keyword arguments:
                - workflow_type: Override workflow type selection
                - goal: Override goal selection
                - user_type: Override user type selection
                - num_actions: Override number of actions
        
        Returns:
            Trajectory object with 3-10 actions, proper timestamps, and metadata
        
        Raises:
            ValueError: If LLM returns invalid structure or too few actions
        
        Note:
            Temporal relationships are calculated based on:
            - Action type (navigate takes longer than click)
            - User type (power_user is faster than first_time)
            - Action context (important actions take longer)
        """
        # Select workflow type and goal
        workflow_type = kwargs.get('workflow_type') or self._select_workflow_type()
        goal = kwargs.get('goal') or self._select_goal(workflow_type)
        logger.debug(f"Generating trajectory: workflow={workflow_type}, goal={goal}")
        
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
            logger.error("LLM returned invalid trajectory structure: missing 'actions' field")
            raise ValueError("LLM returned invalid trajectory structure: missing 'actions' field")
        
        # Extract domain and goal info
        domain = trajectory_structure.get('domain') or get_domain(workflow_type)
        goal_achieved = trajectory_structure.get('goal_achieved', True)
        actions_data = trajectory_structure.get('actions', [])
        logger.debug(f"Trajectory structure: domain={domain}, {len(actions_data)} actions")
        
        # Ensure we have at least some actions
        if not actions_data or len(actions_data) == 0:
            logger.error("LLM returned trajectory structure with no actions")
            raise ValueError("LLM returned trajectory structure with no actions")
        
        # Limit to num_actions to respect configuration
        if len(actions_data) > num_actions:
            actions_data = actions_data[:num_actions]
            logger.debug(f"Limited actions from {len(actions_data)} to {num_actions}")
        
        # Generate IDs
        trajectory_id = generate_id('traj')
        session_id = generate_id('session')
        tab_id = generate_id('tab')
        logger.debug(f"Generated IDs: trajectory={trajectory_id}, session={session_id}, tab={tab_id}")
        
        # Convert actions to BrowserAction objects
        start_time = get_timestamp()
        timestamp = start_time
        browser_actions = []
        failed_actions = 0
        
        for i, action_data in enumerate(actions_data):
            try:
                action_id = f"action_{i+1:03d}"
                logger.debug(f"Converting action {i+1}/{len(actions_data)}: {action_data.get('action_type', 'unknown')}")
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
                failed_actions += 1
                logger.warning(f"Failed to convert action {i+1}: {e}")
                logger.debug(f"Action conversion error details:", exc_info=True)
                continue
        
        if failed_actions > 0:
            logger.warning(f"{failed_actions} actions failed conversion")
        
        # Ensure we have at least 3 actions (minimum required)
        if len(browser_actions) < 3:
            logger.error(f"Too few valid actions after conversion: {len(browser_actions)}. Need at least 3.")
            raise ValueError(f"Too few valid actions after conversion: {len(browser_actions)}. Need at least 3.")
        
        logger.debug(f"Successfully converted {len(browser_actions)} actions")
        
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
    
    def generate_dataset(self, n_trajectories: int) -> Tuple[List[Trajectory], Dict[str, Any]]:
        """
        Generate multiple trajectories to form a complete dataset.
        
        This method generates the specified number of trajectories in parallel,
        applies deduplication if enabled, and returns both the trajectories and
        deduplication statistics.
        
        Args:
            n_trajectories: Number of trajectories to generate (typically 100+)
        
        Returns:
            Tuple containing:
                - List of Trajectory objects (after deduplication if enabled)
                - Dictionary with deduplication statistics:
                    - total_generated: Number of trajectories generated
                    - duplicates_removed: Number of exact duplicates removed
                    - final_count: Number of unique trajectories
        
        Note:
            - Failed trajectory generations are logged but don't stop the process
            - Progress is logged every 10 trajectories
            - Deduplication is applied if enabled in config
            - Uses parallel processing with ThreadPoolExecutor for faster generation
        """
        logger.info(f"Starting generation of {n_trajectories} trajectories (with parallelization)...")
        trajectories = []
        failed_count = 0
        
        # Get max workers from config or use default (5 workers for parallel LLM calls)
        max_workers = self.config.get('generator', {}).get('max_workers', 5)
        logger.debug(f"Using {max_workers} parallel workers for trajectory generation")
        
        # Generate trajectories in parallel batches
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all trajectory generation tasks
            future_to_index = {
                executor.submit(self.generate_trajectory): i 
                for i in range(n_trajectories)
            }
            
            # Process completed futures as they finish
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    trajectory = future.result()
                    trajectories.append(trajectory)
                    completed += 1
                    logger.debug(f"Trajectory {index + 1} generated: {trajectory.trajectory_id} ({len(trajectory.actions)} actions)")
                    
                    if completed % 10 == 0:
                        logger.info(f"Progress: {completed}/{n_trajectories} trajectories generated")
                except Exception as e:
                    failed_count += 1
                    logger.warning(f"Failed to generate trajectory {index+1}: {e}", exc_info=False)
                    logger.debug(f"Trajectory {index+1} generation error details:", exc_info=True)
        
        if failed_count > 0:
            logger.warning(f"{failed_count} trajectory generation attempts failed")
        
        logger.info(f"Generated {len(trajectories)} trajectories (target: {n_trajectories})")
        
        # Apply deduplication if enabled
        dedup_config = self.config.get('generator', {}).get('deduplication', {})
        if dedup_config.get('enabled', True):  # Enabled by default
            logger.info("Applying deduplication (exact duplicates only)...")
            deduplicated, dedup_stats = deduplicate_trajectories(trajectories)
            
            if dedup_stats['exact_duplicates_count'] > 0:
                logger.info(f"Removed {dedup_stats['exact_duplicates_count']} exact duplicates")
                logger.debug(f"Deduplication reduced from {dedup_stats['original_count']} to {dedup_stats['final_count']} trajectories")
            
            return deduplicated, dedup_stats
        else:
            logger.debug("Deduplication disabled")
            # No deduplication
            return trajectories, {
                "original_count": len(trajectories),
                "exact_duplicates_count": 0,
                "near_duplicates_count": 0,
                "final_count": len(trajectories),
                "exact_duplicates_info": [],
                "near_duplicates_info": []
            }

