"""LLM-based generation for realistic browser interaction data"""

import json
import os
from typing import Dict, List, Optional, Any
import openai
from src.llm_schema_validator import (
    validate_trajectory_structure,
    normalize_trajectory_structure,
    VALID_ACTION_TYPES
)


class LLMDataGenerator:
    """Generate realistic browser interaction data using OpenAI API or OpenRouter"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", use_openrouter: bool = False):
        """
        Initialize LLM generator.
        
        Args:
            api_key: API key (defaults to OPENAI_API_KEY or OPENROUTER_API_KEY env var)
            model: Model to use
            use_openrouter: If True, use OpenRouter API instead of OpenAI
        """
        self.use_openrouter = use_openrouter
        
        if use_openrouter:
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass api_key parameter.")
            # OpenRouter uses OpenAI-compatible API
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter.")
            self.client = openai.OpenAI(api_key=self.api_key)
        
        self.model = model
    
    def generate_element_data(
        self, 
        element_type: str, 
        context: str,
        workflow_type: str
    ) -> Dict[str, Any]:
        """
        Generate realistic element data (selector, text, id, classes) for a given element type.
        
        Args:
            element_type: Type of element (button, input, link, select, etc.)
            context: Context of the element (e.g., "search button on homepage")
            workflow_type: Type of workflow (e-commerce, form_filling, research)
        
        Returns:
            Dict with selector, element_text, element_id, element_classes
        """
        prompt = f"""Generate realistic HTML element data for a {element_type} element in a {workflow_type} website.

Context: {context}
Element type: {element_type}

Generate realistic:
1. CSS selector (use common patterns like id, class, data attributes, semantic selectors)
2. Element text (visible text content, or null if not applicable)
3. Element ID (HTML id attribute, or null if not applicable)
4. Element classes (array of CSS class names, or empty array if none)

Return JSON only, no markdown:
{{
    "selector": "realistic CSS selector",
    "element_text": "text content or null",
    "element_id": "id value or null",
    "element_classes": ["class1", "class2"]
}}

Make it realistic for modern web applications."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a web development expert. Generate realistic HTML element data in JSON format only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            data = json.loads(content)
            return {
                "selector": data.get("selector", f"#{element_type}"),
                "element_text": data.get("element_text"),
                "element_id": data.get("element_id"),
                "element_classes": data.get("element_classes", [])
            }
        except Exception as e:
            # Fallback to simple selector on error
            print(f"Warning: LLM generation failed: {e}. Using fallback.")
            return {
                "selector": f"#{element_type}-{context.lower().replace(' ', '-')}",
                "element_text": None,
                "element_id": f"{element_type}-id",
                "element_classes": []
            }
    
    def generate_url_and_title(
        self,
        page_type: str,
        workflow_type: str,
        domain: str,
        context: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate realistic URL and page title.
        
        Args:
            page_type: Type of page (homepage, product, search_results, etc.)
            workflow_type: Type of workflow
            domain: Domain name
            context: Additional context (e.g., product name)
        
        Returns:
            Dict with url and page_title
        """
        prompt = f"""Generate a realistic URL and page title for a {page_type} page on a {workflow_type} website.

Domain: {domain}
Context: {context or "N/A"}

Return JSON only, no markdown:
{{
    "url": "https://example.com/path",
    "page_title": "Realistic Page Title"
}}

Make it realistic for modern web applications."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a web development expert. Generate realistic URLs and page titles in JSON format only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            data = json.loads(content)
            return {
                "url": data.get("url", f"https://{domain}"),
                "page_title": data.get("page_title", f"{page_type.replace('_', ' ').title()} - {domain}")
            }
        except Exception as e:
            print(f"Warning: LLM generation failed: {e}. Using fallback.")
            return {
                "url": f"https://{domain}/{page_type}",
                "page_title": f"{page_type.replace('_', ' ').title()} - {domain}"
            }
    
    def generate_text_input_value(
        self,
        field_type: str,
        workflow_type: str,
        context: Optional[str] = None
    ) -> str:
        """
        Generate realistic text input value.
        
        Args:
            field_type: Type of field (email, name, search_query, address, etc.)
            workflow_type: Type of workflow
            context: Additional context
        
        Returns:
            Realistic text value for the field
        """
        prompt = f"""Generate a realistic {field_type} value for a {workflow_type} website form.

Context: {context or "N/A"}

Return only the value, no explanation, no quotes, no JSON. Just the text value.

Examples:
- email: user@example.com
- name: John Doe
- search_query: laptop
- address: 123 Main St, New York, NY 10001"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Generate realistic form field values. Return only the value, nothing else."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=50
            )
            
            return response.choices[0].message.content.strip().strip('"').strip("'")
        except Exception as e:
            print(f"Warning: LLM generation failed: {e}. Using fallback.")
            # Fallback using faker would go here, but keeping simple for now
            return f"test_{field_type}@example.com" if field_type == "email" else f"test {field_type}"
    
    def generate_trajectory_structure(
        self,
        workflow_type: str,
        goal: str,
        user_type: str,
        num_actions: int
    ) -> Dict[str, Any]:
        """
        Generate a complete trajectory structure with sequence of actions.
        
        Args:
            workflow_type: Type of workflow (e-commerce, form_filling, research)
            goal: High-level goal (e.g., "purchase_product", "submit_contact_form")
            user_type: User behavior type (power_user, casual, first_time)
            num_actions: Target number of actions (3-10)
        
        Returns:
            Dict with trajectory structure including actions list
        """
        # Enhanced prompt with explicit schema requirements
        prompt = f"""Generate a realistic browser interaction trajectory for a {user_type} user on a {workflow_type} website.

Goal: {goal}
Target number of actions: {num_actions}

SCHEMA REQUIREMENTS:

Required top-level fields:
- "domain": string (e.g., "example-store.com")
- "goal": string (same as goal parameter: "{goal}")
- "goal_achieved": boolean
- "actions": array of action objects

Each action object MUST have:
- "action_type": string, one of {VALID_ACTION_TYPES}
- "url": string (valid HTTP/HTTPS URL)
- "page_title": string

Each action object SHOULD have:
- "element_type": string (HTML element type: button, input, link, select, etc.)
- "context": string (description of what user is doing)
- "user_intent": string (semantic intent: "add_to_cart", "search", "submit_form", etc.)
- "is_intentional": boolean (usually true for workflow actions)

Action-type-specific fields:
- For "type" actions: include "value" (string) or "value_hint" (string), and "field_type" (string)
- For "select" actions: include "option_index" (integer)
- For "scroll" actions: include "coordinates" (object with "x" and "y" integers)

Example structure:
{{
    "domain": "example-store.com",
    "goal": "{goal}",
    "goal_achieved": true,
    "actions": [
        {{
            "action_type": "navigate",
            "url": "https://example-store.com",
            "page_title": "Example Store - Home",
            "context": "Navigate to homepage",
            "user_intent": "navigate",
            "is_intentional": true
        }},
        {{
            "action_type": "type",
            "element_type": "input",
            "url": "https://example-store.com",
            "page_title": "Example Store - Home",
            "context": "Type search query in search box",
            "user_intent": "search",
            "is_intentional": true,
            "field_type": "search_query",
            "value_hint": "product name"
        }},
        {{
            "action_type": "click",
            "element_type": "button",
            "url": "https://example-store.com",
            "page_title": "Example Store - Home",
            "context": "Click search button",
            "user_intent": "search",
            "is_intentional": true
        }}
    ]
}}

IMPORTANT:
- Return ONLY valid JSON (no markdown, no code blocks, no explanations)
- Ensure all URLs are valid HTTP/HTTPS URLs
- All action_type values must be from the valid list above
- Generate exactly {num_actions} actions (or close to it)
- Make sequences realistic for a {user_type} user
- Match user behavior patterns for {user_type} (speed, exploration, etc.)"""

        try:
            # Try to use response_format for structured output if available
            # (OpenAI API supports response_format={"type": "json_object"})
            request_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert in web user behavior analysis. Generate realistic browser interaction trajectories. Return ONLY valid JSON, no markdown or code blocks."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 2000
            }
            
            # Add JSON mode if model supports it (GPT-4o, GPT-4-turbo, etc.)
            # Check if model name suggests JSON mode support
            if any(x in self.model.lower() for x in ['gpt-4', 'gpt-3.5', 'o1']):
                try:
                    request_params["response_format"] = {"type": "json_object"}
                except Exception:
                    # If response_format not supported, continue without it
                    pass
            
            response = self.client.chat.completions.create(**request_params)
            
            content = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present (backup in case model ignores instruction)
            if content.startswith("```"):
                parts = content.split("```")
                if len(parts) >= 2:
                    content = parts[1]
                    if content.startswith("json"):
                        content = content[4:]
            content = content.strip()
            
            # Parse JSON
            trajectory_structure = json.loads(content)
            
            # Validate structure before returning
            is_valid, errors = validate_trajectory_structure(trajectory_structure)
            if not is_valid:
                print(f"Warning: LLM response validation found {len(errors)} issues:")
                for error in errors[:5]:  # Show first 5 errors
                    print(f"  - {error}")
                if len(errors) > 5:
                    print(f"  ... and {len(errors) - 5} more issues")
                
                # Try to normalize and fix issues
                try:
                    trajectory_structure = normalize_trajectory_structure(trajectory_structure)
                    # Re-validate after normalization
                    is_valid_after, errors_after = validate_trajectory_structure(trajectory_structure)
                    if is_valid_after:
                        print("✓ Normalization fixed validation issues")
                    elif len(errors_after) < len(errors):
                        print(f"Warning: Normalization fixed some issues ({len(errors)} -> {len(errors_after)} remaining)")
                        # Continue with normalized structure even if some issues remain
                        # (non-critical issues may be handled by conversion layer)
                    else:
                        print(f"Warning: Normalization did not improve validation. {len(errors_after)} issues remain")
                        # Still continue - conversion layer may handle missing fields
                except Exception as norm_error:
                    print(f"Warning: Normalization failed: {norm_error}. Proceeding with original structure.")
            
            return trajectory_structure
            
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse LLM response as JSON: {e}")
            print("Response preview:", content[:200] if 'content' in locals() else "N/A")
        except ValueError as e:
            print(f"Warning: LLM response validation failed: {e}")
        except Exception as e:
            print(f"Warning: LLM trajectory generation failed: {e}")
        
        # Fallback to simple structure
        return {
            "domain": f"example-{workflow_type}.com",
            "goal": goal,
            "goal_achieved": True,
            "actions": [
                {
                    "action_type": "navigate",
                    "url": f"https://example-{workflow_type}.com",
                    "page_title": f"{workflow_type.replace('_', ' ').title()} - Home",
                    "context": "Navigate to homepage",
                    "user_intent": "navigate",
                    "is_intentional": True
                }
            ]
        }

