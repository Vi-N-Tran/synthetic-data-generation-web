"""Action generation utilities using LLM for realistic data"""

import random
from typing import Dict, List, Optional
from src.schema import BrowserAction
from src.utils import (
    generate_id, get_timestamp_offset,
    get_decision_time, calculate_typing_time,
    get_typing_speed_wpm, get_reading_time
)
from src.llm_generator import LLMDataGenerator


def create_navigate_action(
    timestamp: float,
    url: str,
    page_title: str,
    session_id: str,
    tab_id: str,
    action_id: str,
    is_intentional: bool = True,
    llm_generator: Optional[LLMDataGenerator] = None
) -> BrowserAction:
    """Create a navigate action"""
    return BrowserAction(
        timestamp=timestamp,
        action_type='navigate',
        action_id=action_id,
        element_type='',
        element_selector='',
        url=url,
        page_title=page_title,
        session_id=session_id,
        tab_id=tab_id,
        is_intentional=is_intentional,
        user_intent='navigate' if is_intentional else None
    )


def create_click_action(
    timestamp: float,
    element_type: str,
    url: str,
    page_title: str,
    session_id: str,
    tab_id: str,
    action_id: str,
    workflow_type: str,
    context: str,
    user_intent: Optional[str] = None,
    is_intentional: bool = True,
    llm_generator: Optional[LLMDataGenerator] = None,
    selector: Optional[str] = None,
    element_text: Optional[str] = None,
    element_id: Optional[str] = None,
    element_classes: Optional[List[str]] = None
) -> BrowserAction:
    """Create a click action with LLM-generated element data"""
    # Use LLM to generate element data if generator provided
    if llm_generator and not selector:
        element_data = llm_generator.generate_element_data(
            element_type=element_type,
            context=context,
            workflow_type=workflow_type
        )
        selector = element_data["selector"]
        element_text = element_data.get("element_text") or element_text
        element_id = element_data.get("element_id") or element_id
        element_classes = element_data.get("element_classes") or element_classes
    elif not selector:
        # Fallback if no LLM generator and no selector provided
        selector = f"#{element_type}-{generate_id('el')}"
    
    return BrowserAction(
        timestamp=timestamp,
        action_type='click',
        action_id=action_id,
        element_type=element_type,
        element_selector=selector,
        element_text=element_text,
        element_id=element_id,
        element_classes=element_classes,
        url=url,
        page_title=page_title,
        session_id=session_id,
        tab_id=tab_id,
        is_intentional=is_intentional,
        user_intent=user_intent,
        element_visible=True,
        element_clickable=True
    )


def create_type_action(
    timestamp: float,
    url: str,
    page_title: str,
    session_id: str,
    tab_id: str,
    action_id: str,
    workflow_type: str,
    field_type: str,
    context: Optional[str] = None,
    user_intent: Optional[str] = None,
    is_intentional: bool = True,
    llm_generator: Optional[LLMDataGenerator] = None,
    selector: Optional[str] = None,
    value: Optional[str] = None,
    element_id: Optional[str] = None
) -> BrowserAction:
    """Create a type action with LLM-generated data"""
    # Use LLM to generate input value and element data
    if llm_generator:
        if not value:
            value = llm_generator.generate_text_input_value(
                field_type=field_type,
                workflow_type=workflow_type,
                context=context
            )
        if not selector:
            element_data = llm_generator.generate_element_data(
                element_type='input',
                context=f"{field_type} input field",
                workflow_type=workflow_type
            )
            selector = element_data["selector"]
            element_id = element_data.get("element_id") or element_id
    elif not selector:
        selector = f"input[name='{field_type}']"
    if not value:
        value = f"test_{field_type}"
    return BrowserAction(
        timestamp=timestamp,
        action_type='type',
        action_id=action_id,
        element_type='input',
        element_selector=selector,
        value=value,
        element_id=element_id,
        url=url,
        page_title=page_title,
        session_id=session_id,
        tab_id=tab_id,
        is_intentional=is_intentional,
        user_intent=user_intent,
        element_visible=True,
        element_clickable=True
    )


def create_scroll_action(
    timestamp: float,
    url: str,
    page_title: str,
    session_id: str,
    tab_id: str,
    action_id: str,
    coordinates: Dict[str, int],
    is_intentional: bool = True
) -> BrowserAction:
    """Create a scroll action"""
    return BrowserAction(
        timestamp=timestamp,
        action_type='scroll',
        action_id=action_id,
        element_type='body',
        element_selector='body',
        coordinates=coordinates,
        url=url,
        page_title=page_title,
        session_id=session_id,
        tab_id=tab_id,
        is_intentional=is_intentional,
        element_visible=True,
        element_clickable=False
    )


def create_select_action(
    timestamp: float,
    url: str,
    page_title: str,
    session_id: str,
    tab_id: str,
    action_id: str,
    workflow_type: str,
    context: str,
    option_index: int,
    user_intent: Optional[str] = None,
    is_intentional: bool = True,
    llm_generator: Optional[LLMDataGenerator] = None,
    selector: Optional[str] = None,
    element_id: Optional[str] = None
) -> BrowserAction:
    """Create a select action with LLM-generated element data"""
    if llm_generator and not selector:
        element_data = llm_generator.generate_element_data(
            element_type='select',
            context=context,
            workflow_type=workflow_type
        )
        selector = element_data["selector"]
        element_id = element_data.get("element_id") or element_id
    elif not selector:
        selector = f"select[name='{context.lower().replace(' ', '_')}']"
    
    return BrowserAction(
        timestamp=timestamp,
        action_type='select',
        action_id=action_id,
        element_type='select',
        element_selector=selector,
        option_index=option_index,
        element_id=element_id,
        url=url,
        page_title=page_title,
        session_id=session_id,
        tab_id=tab_id,
        is_intentional=is_intentional,
        user_intent=user_intent,
        element_visible=True,
        element_clickable=True
    )


def create_submit_action(
    timestamp: float,
    url: str,
    page_title: str,
    session_id: str,
    tab_id: str,
    action_id: str,
    user_intent: Optional[str] = None,
    is_intentional: bool = True
) -> BrowserAction:
    """Create a submit action"""
    return BrowserAction(
        timestamp=timestamp,
        action_type='submit',
        action_id=action_id,
        element_type='form',
        element_selector='form',
        url=url,
        page_title=page_title,
        session_id=session_id,
        tab_id=tab_id,
        is_intentional=is_intentional,
        user_intent=user_intent,
        element_visible=True,
        element_clickable=True
    )


def create_wait_action(
    timestamp: float,
    url: str,
    page_title: str,
    session_id: str,
    tab_id: str,
    action_id: str,
    duration: float,
    is_intentional: bool = True
) -> BrowserAction:
    """Create a wait action"""
    return BrowserAction(
        timestamp=timestamp,
        action_type='wait',
        action_id=action_id,
        element_type='',
        element_selector='',
        url=url,
        page_title=page_title,
        session_id=session_id,
        tab_id=tab_id,
        is_intentional=is_intentional,
        element_visible=True,
        element_clickable=False
    )

