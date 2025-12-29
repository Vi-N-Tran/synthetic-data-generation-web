# Browser Trajectory Dataset Generation - Implementation Specification

**Author:** Senior ML/AI Engineer  
**Date:** 2024  
**Status:** Design Specification

---

## Executive Summary

This document outlines the design and implementation approach for a synthetic browser trajectory dataset generation pipeline. The solution will generate realistic user interaction sequences suitable for training browser automation models, with emphasis on data quality, ML-readiness, and scalability.

---

## 1. System Architecture Overview

### 1.1 High-Level Design


### 1.2 Component Responsibilities


---

## 2. Data Schema Design

### 2.1 BrowserAction Schema

**Core Fields:**
```python
@dataclass
class BrowserAction:
    # Temporal
    timestamp: float              # Unix timestamp (milliseconds)
    
    # Action Identity
    action_type: str              # click, type, navigate, scroll, hover, select, etc.
    action_id: str                # Unique identifier within trajectory
    
    # Target Element
    element_type: str             # button, input, link, select, canvas, iframe, etc.
    element_selector: str         # CSS selector or XPath
    element_text: Optional[str]   # Visible text content
    element_id: Optional[str]     # HTML id attribute
    element_classes: Optional[str]    # CSS classes
    
    # Action Parameters
    value: Optional[str]          # For type actions: text input
    option_index: Optional[int]   # For select actions
    coordinates: Optional[Dict[str, int]]  # x, y for click/scroll
    
    # Context
    url: str                      # Current page URL
    page_title: str               # Page title
    
    # Intent Signals
    is_intentional: bool          # True for goal-directed actions
    confidence: float             # Model confidence in action (0-1)
    user_intent: Optional[str]    # Semantic intent (e.g., "add_to_cart", "search")
    
    # DOM State
    dom_snapshot: Optional[str]   # Lightweight DOM representation
    element_visible: bool         # Whether element was visible
    element_clickable: bool       # Whether element was clickable
    
    # Metadata
    session_id: str               # Browser session identifier
    tab_id: str                   # Tab identifier
    frame_id: Optional[str]       # For iframe interactions
```

**Action Types:**
- `click`: Mouse click on element
- `type`: Text input
- `navigate`: URL navigation
- `scroll`: Page scrolling
- `hover`: Mouse hover
- `select`: Dropdown selection
- `submit`: Form submission
- `back`: Browser back button
- `forward`: Browser forward button
- `refresh`: Page refresh
- `wait`: Intentional wait/delay
- `drag_drop`: Drag and drop operation

**Why Each Action Type is Needed:**

**1. `click` - Mouse click on element**
- **Purpose**: Most common interaction - clicking buttons, links, checkboxes, etc.
- **ML Value**: Core action for element selection models - predicts which element to click
- **Frequency**: ~40-50% of all actions in typical browsing
- **Examples**: Click "Add to Cart", click navigation links, click form buttons

**2. `type` - Text input**
- **Purpose**: Entering text into input fields, textareas, search boxes
- **ML Value**: Models need to predict what text to type and where
- **Frequency**: ~20-30% of actions in form-filling workflows
- **Examples**: Enter email, search query, form data
- **Note**: Requires `value` parameter to store the typed text

**3. `navigate` - URL navigation**
- **Purpose**: Direct URL navigation (address bar, bookmarks, external links)
- **ML Value**: Models learn navigation patterns and page transitions
- **Frequency**: ~5-10% of actions (less common than clicking links)
- **Examples**: Type URL in address bar, open bookmark, navigate to new domain
- **Note**: Different from clicking a link (which is a `click` that happens to change URL)

**4. `scroll` - Page scrolling**
- **Purpose**: Scrolling to view content, reveal elements, infinite scroll
- **ML Value**: Models learn when scrolling is needed to reveal target elements
- **Frequency**: ~10-15% of actions (very common in modern web)
- **Examples**: Scroll to see "Add to Cart" button, scroll through product list
- **Note**: Requires `coordinates` parameter for scroll position

**5. `hover` - Mouse hover**
- **Purpose**: Hovering over elements to reveal tooltips, dropdowns, previews
- **ML Value**: Models learn hover patterns that precede clicks (e.g., hover menu → click item)
- **Frequency**: ~5-10% of actions (common in complex UIs)
- **Examples**: Hover over product image for zoom, hover menu to see options
- **Note**: Often precedes `click` actions, creating temporal dependencies

**6. `select` - Dropdown selection**
- **Purpose**: Selecting options from `<select>` dropdowns
- **ML Value**: Models predict which option to select based on context
- **Frequency**: ~5% of actions (common in forms)
- **Examples**: Select country, shipping method, product size
- **Note**: Requires `option_index` parameter to specify selected option

**7. `submit` - Form submission**
- **Purpose**: Submitting forms (distinct from clicking submit button)
- **ML Value**: Models learn form completion patterns and submission timing
- **Frequency**: ~2-5% of actions (end of form workflows)
- **Examples**: Submit login form, checkout form, contact form
- **Note**: Semantically different from `click` on submit button - represents form submission event

**8. `back` - Browser back button**
- **Purpose**: Navigating back in browser history
- **ML Value**: Models learn backtracking patterns and error recovery
- **Frequency**: ~2-5% of actions (common in error recovery)
- **Examples**: Go back after clicking wrong link, undo navigation mistake
- **Note**: Important for understanding user intent changes and corrections

**9. `forward` - Browser forward button**
- **Purpose**: Navigating forward in browser history
- **ML Value**: Models learn forward navigation patterns (less common than back)
- **Frequency**: ~1-2% of actions (less common)
- **Examples**: Return to page after going back, redo navigation
- **Note**: Less frequent than `back` but important for complete trajectory modeling

**10. `refresh` - Page refresh**
- **Purpose**: Reloading the current page
- **ML Value**: Models learn when users refresh (e.g., after errors, to get updates)
- **Frequency**: ~1-2% of actions (uncommon but important)
- **Examples**: Refresh after error, reload for new content, retry failed action
- **Note**: Important for error recovery and retry patterns

**11. `wait` - Intentional wait/delay**
- **Purpose**: Explicit pauses (distinct from natural delays between actions)
- **ML Value**: Models learn when to wait for page loads, animations, or user reading time
- **Frequency**: ~5-10% of actions (varies by workflow)
- **Examples**: Wait for page load, wait for animation, intentional reading pause
- **Note**: Helps distinguish intentional waits from natural action intervals

**12. `drag_drop` - Drag and drop operation**
- **Purpose**: Dragging elements and dropping them elsewhere
- **ML Value**: Models learn drag-drop patterns (file uploads, reordering, etc.)
- **Frequency**: ~1-3% of actions (specialized but important)
- **Examples**: Drag file to upload, reorder list items, drag slider
- **Note**: Requires `coordinates` for both drag start and drop end positions

### 2.2 Trajectory Schema

```python
@dataclass
class Trajectory:
    # Identity
    trajectory_id: str            # Unique trajectory identifier
    session_id: str               # Browser session
    
    # Actions
    actions: List[BrowserAction]  # Ordered sequence of actions
    
    # Metadata
    workflow_type: str            # e-commerce, form_filling, research, etc.
    domain: str                   # Primary domain (e.g., "amazon.com")
    start_time: float             # Trajectory start timestamp
    end_time: float               # Trajectory end timestamp
    duration: float               # Total duration (seconds)
    
    # User Profile
    user_type: str                # power_user, casual, first_time, etc.
    device_type: str              # desktop, mobile, tablet
    browser_type: str             # chrome, firefox, safari
    
    # Goal Information
    goal: Optional[str]           # High-level goal description
    goal_achieved: bool           # Whether goal was completed
    success_indicators: List[str]  # Signals of goal completion
    
    # Quality Metrics (computed properties, not stored fields)
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
        """Number of back/forward navigations (indicates corrections/exploration)"""
        return sum(1 for a in self.actions if a.action_type in ['back', 'forward'])
    
    @property
    def error_count(self) -> int:
        """Number of failed actions (element not visible/clickable, validation errors)"""
        return sum(1 for a in self.actions 
                  if not a.element_visible or not a.element_clickable)
```
---

## 3. Data Storage Format

### 3.1 Format Choice: JSON Lines (JSONL)

**Rationale:**
- **Human Readable**: Easy to inspect and debug
- **Streaming Friendly**: Can process one trajectory at a time
- **ML Framework Compatible**: Works with PyTorch/TensorFlow data loaders
- **Efficient**: No need to load entire dataset into memory
- **Version Control Friendly**: Line-by-line diffs

**Optional Parquet Conversion:**
- After generating JSONL, optionally convert to Parquet format
- Pros: Better compression (~70% smaller), columnar storage, faster reads for ML pipelines
- Cons: Less human-readable, requires pyarrow dependency
- **Decision**: Generate JSONL first (human-readable, debuggable), then optionally convert to Parquet for production/ML use

### 3.2 File Structure

```
output/
├── trajectories.jsonl          # Main dataset file (always generated)
├── trajectories.parquet        # Parquet format (optional, if conversion enabled)
├── metadata.json               # Dataset-level metadata
├── statistics.json             # Computed statistics
└── schema.json                 # JSON schema for validation
```

**Note**: Parquet file is only generated if `convert_to_parquet: true` in config.

**JSONL Format:**
- One trajectory per line
- Each line is valid JSON
- Trajectory objects serialized via `to_dict()` method

**Metadata File:**
```json
{
  "dataset_version": "1.0",
  "generation_date": "2024-01-01T00:00:00Z",
  "generator_config": {...},
  "total_trajectories": 100,
  "total_actions": 500,
  "schema_version": "1.0"
}
```

### 3.3 Optional Parquet Conversion

**Purpose**: Convert JSONL to Parquet format for better compression and faster ML pipeline processing.

**Implementation**:
1. After writing JSONL file, check if `convert_to_parquet: true` in config
2. Read JSONL file using pandas
3. Convert nested structures (actions list) appropriately
4. Write to Parquet format using pyarrow
5. Preserve all trajectory metadata

**Conversion Strategy**:
- **Option 1 (Recommended)**: Flatten actions into separate rows
  - Each row = one action with trajectory_id as foreign key
  - Enables efficient columnar storage and filtering
  - Structure: `trajectory_id | action_id | timestamp | action_type | ...`
  
- **Option 2**: Keep nested structure
  - Store actions as nested arrays/structs in Parquet
  - Preserves trajectory-level grouping
  - Less efficient for ML pipelines

**Benefits**:
- **Compression**: ~70% smaller file size
- **Performance**: Faster reads for ML frameworks (PyTorch, TensorFlow)
- **Columnar**: Efficient filtering and aggregation
- **Compatibility**: Works with pandas, Dask, Spark

**When to Use**:
- Production datasets (large scale)
- ML training pipelines
- When storage/bandwidth is a concern
- When human readability is not required

**When to Skip**:
- Development/debugging (keep JSONL for readability)
- Small datasets (< 1000 trajectories)
- When pyarrow/pandas dependencies are not available

---

## 4. Synthetic Data Generation Strategy

### 4.1 LLM-Based Trajectory Generation

**Approach**: Use OpenAI API to generate complete trajectory structures based on the defined schemas. The LLM generates realistic sequences of browser interactions that match user goals and behavior patterns.

**Workflow Types Supported:**
- **E-commerce**: Product browsing, searching, purchasing workflows
- **Form Filling**: Contact forms, signups, subscriptions
- **Research**: Information finding, article reading, comparison tasks

**Generation Process**:
1. LLM receives workflow type, goal, user type, and target action count
2. LLM generates complete trajectory structure with:
   - Sequence of actions (action_type, element_type, context)
   - Realistic URLs and page titles for each step
   - User intent labels for each action
   - Goal achievement status
3. System converts LLM-generated structure into `BrowserAction` objects
4. LLM generates element-level details (selectors, element text, IDs, classes) for each action
5. Temporal sequencing applied based on user type and action characteristics

**Benefits**:
- More realistic and varied trajectories (not limited to templates)
- LLM understands context and generates coherent sequences
- Natural variation in action sequences
- Adapts to different user types and goals automatically

### 4.2 LLM-Based Action Data Generation

**Element Data Generation**:
- LLM generates realistic CSS selectors based on context (e.g., `button[data-testid="add-to-cart"]`, `.product-card__title`)
- LLM provides element text, IDs, and CSS classes matching modern web patterns
- Supports semantic HTML5 elements, React/Vue patterns, and accessibility attributes
- Each element generation includes context about the workflow and action purpose

**URL and Page Title Generation**:
- LLM generates realistic URLs for each page in the trajectory
- Page titles match the context and workflow type
- URLs follow realistic patterns (paths, query parameters, etc.)

**Text Input Values**:
- LLM generates realistic text values for form fields (emails, names, addresses, search queries)
- Values are contextually appropriate for the workflow and field type
- Typing speed simulated via timestamps based on user type (30-80 WPM)

**Temporal Realism** (applied post-LLM generation):
- Reading time: 5-45 seconds per page (varies by content complexity and user intent)
- Typing speed: 30-80 WPM (average user range, varies by user type)
- Decision time: 1-10 seconds before clicks (longer for important actions like purchases)
- Page load time: 0.3-3 seconds (network and rendering time)

### 4.3 LLM-Based Variability

**1. Trajectory Structure Variation:**
- LLM generates different action sequences for the same goal
- Natural variation in workflow steps (some users skip optional steps)
- Multiple paths to same goal emerge organically from LLM generation
- Workflow type selection via configuration distribution

**2. Action-Level Variation:**
- LLM generates unique selectors for each action (not template-based)
- Varying action sequences (hover before click, scroll patterns, etc.)
- Different element types and contexts for similar actions
- Natural temporal variations in action sequences

**3. User Behavior Patterns** (prompted to LLM):
- **Power User**: Fast actions, minimal scrolling, direct navigation
- **Casual User**: Slower actions, more scrolling, exploratory behavior
- **First-time User**: Longer pauses, more back/forward, trial and error patterns
- LLM adapts generated trajectories to match user type characteristics

**4. Error Simulation**:
- Can be added via prompts or post-processing
- LLM can generate error recovery patterns when requested
- Failed actions, validation errors, and retry behaviors

### 4.4 Trajectory Length Distribution

- **Target**: 3-10 actions per trajectory (as specified)
- **Distribution**: 
  - 3-4 actions: 20% (quick tasks)
  - 5-6 actions: 40% (typical workflows)
  - 7-8 actions: 30% (complex tasks)
  - 9-10 actions: 10% (multi-step processes)

---

## 5. Data Quality & Validation

### 5.1 Validation Rules

**Trajectory-Level:**
1. ✅ At least 3 actions, at most 10 actions
2. ✅ Actions are temporally ordered (timestamps increasing)
3. ✅ All actions have valid action_type
4. ✅ URL consistency (navigate actions change URL appropriately)
5. ✅ No orphaned actions (all actions belong to valid trajectory)

**Action-Level:**
1. ✅ Required fields present (timestamp, action_type, url)
2. ✅ Action type matches parameters (e.g., `type` has `value`)
3. ✅ Selectors are valid CSS/XPath format
4. ✅ Timestamps are positive and reasonable
5. ✅ Coordinates within viewport bounds (if provided)

**Semantic Validation:**
1. ✅ Navigate actions have valid URL format
2. ✅ Type actions have non-empty value (or intentional empty)
3. ✅ Select actions have valid option_index
4. ✅ Form submissions follow form filling actions

**Business Logic Validations:**
1. ✅ Workflow state transitions are logical (e.g., can't checkout before adding to cart)
2. ✅ Goal achievement matches actual trajectory completion
3. ✅ Domain consistency within trajectory (single domain unless explicit navigation)
4. ✅ Action-element compatibility (type actions only on input elements)
5. ✅ Temporal consistency (realistic timing for action types)

**Workflow Completeness:**
1. ✅ Critical workflow steps present (e.g., e-commerce has product selection)
2. ✅ User intent matches action sequence
3. ✅ Intentional actions align with workflow goals

### 5.2 Edge Case Handling

### 5.2 Edge Case Handling

**1. Temporal Anomalies:**
- Detect and fix negative time intervals
- Cap maximum time intervals (e.g., > 1 hour = error)
- Handle concurrent actions (same timestamp) - add minimal offset (1ms)

**2. Invalid Selectors:**
- Validate selector syntax
- Fallback to element_id or element_text if selector invalid

**3. Missing Context:**
- Infer missing fields where possible
- Mark inferred fields in metadata

**4. Duplicate Action IDs:**
- Ensure action_id is unique within trajectory
- Generate new ID if duplicate detected
- Log warning for ID collisions

**5. Trajectory-Level Deduplication:**
- Detect exact duplicate trajectories (same action sequences)
- Optional near-duplicate detection with configurable similarity threshold
- Content-based fingerprinting using action sequences, URLs, and element data
- Excludes temporal and ID information from fingerprint (focuses on content)
- Removes duplicates to ensure dataset diversity
- Reports deduplication statistics (counts, duplicate pairs)
- Configurable deduplication mode (exact only, or exact + near-duplicates)

**5. Circular Navigation Loops:**
- Detect excessive back/forward patterns (e.g., > 5 consecutive back/forward)
- Break loops or flag as exploratory behavior
- Prevent infinite navigation cycles

**7. Type Mismatches:**
- Ensure timestamp is float (not string)
- Validate option_index is integer
- Ensure coordinates dict has int values (not strings)
- Validate boolean fields are actually boolean

**8. Text Input Edge Cases:**
- Extremely long text values (> 10K characters) - truncate or reject

### 5.3 Trajectory Deduplication

**Purpose:**
Ensure dataset diversity by removing duplicate and near-duplicate trajectories that provide no additional information for ML training.

**Exact Duplicate Detection:**
- Content-based fingerprinting using SHA256 hash of:
  - Action sequence (action types, element types, selectors)
  - URLs and page titles (normalized)
  - Action parameters (values, option_index)
  - Workflow type and goal
- Excludes from fingerprint:
  - Timestamps (temporal information)
  - Trajectory/action IDs (metadata)
  - Session/tab IDs (metadata)
- Two trajectories with identical fingerprints are considered exact duplicates
- First occurrence is kept, subsequent duplicates are removed

**Near-Duplicate Detection (Optional):**
- Similarity scoring based on:
  - Action sequence alignment (action types and order)
  - Element type matching
  - URL path similarity
  - Goal matching
- Configurable similarity threshold (default: 0.9 = 90% similar)
- More computationally expensive than exact duplicate detection
- Applied after exact duplicate removal

**Implementation Details:**
- Deduplication runs after trajectory generation but before final validation
- Statistics reported: original count, exact duplicates removed, near-duplicates removed
- Duplicate information logged (trajectory IDs, similarity scores) for analysis
- Configurable via generator config:
  ```json
  {
    "generator": {
      "deduplication": {
        "enabled": true,
        "mode": "exact_only",  // or "exact_and_near"
        "similarity_threshold": 0.9  // for near-duplicate detection
      }
    }
  }
  ```

**Benefits:**
- Prevents over-representation of common patterns
- Ensures dataset diversity for better ML generalization
- Reduces dataset size while maintaining information content
- Identifies LLM generation patterns (if many duplicates appear, may indicate prompt issues)

**9. Serialization Issues:**
- Circular references in nested structures
- Non-serializable objects
- Very large objects causing memory issues
- Encoding issues with special characters

**10. Resource Exhaustion:**
- Extremely long selectors (> 500 chars) - validate or truncate
- Memory issues with large nested structures
- Generation timeouts for complex workflows

---

## 6. Implementation Plan

### 6.1 File Structure

```
python/
├── main.py                      # Entry point (existing)
├── requirements.txt            # Dependencies
├── src/
│   ├── __init__.py
│   ├── schema.py               # BrowserAction, Trajectory dataclasses
│   ├── generator.py            # TrajectoryGenerator implementation
│   ├── llm_generator.py        # LLM-based data generation
│   ├── actions.py              # Action creation utilities (uses LLM generator)
│   ├── validator.py            # Validation logic
│   ├── statistics.py           # Statistics computation
│   ├── writer.py               # DatasetWriter implementation
│   ├── deduplication.py        # Trajectory deduplication logic
│   └── utils.py                # Helper functions
├── config/
│   └── generator_config.json   # Configuration file
└── output/                     # Generated dataset (gitignored)
    ├── trajectories.jsonl
    ├── trajectories.parquet    # Optional, if conversion enabled
    ├── metadata.json
    └── statistics.json
```

### 6.2 Implementation Phases

**Phase 1: Schema Implementation**
- Complete `BrowserAction` dataclass with all fields
- Complete `Trajectory` dataclass with all fields
- Implement `to_dict()` methods for serialization
- Add JSON schema validation

**Phase 2: LLM Integration**
- Implement `LLMDataGenerator` class for OpenAI API integration
- Implement trajectory structure generation via LLM
- Implement element data generation (selectors, URLs, titles, text values)
- Add error handling and fallback mechanisms

**Phase 3: Action Creation Utilities**
- Implement action creation functions (click, type, navigate, etc.)
- Integrate LLM generator for realistic element data
- Add temporal sequencing logic based on user type
- Handle action type-specific parameters

**Phase 4: Trajectory Generation**
- Implement `TrajectoryGenerator.generate_trajectory()` method
- Use LLM to generate complete trajectory structures
- Convert LLM-generated data into `BrowserAction` objects
- Apply temporal sequencing and user behavior patterns
- Implement trajectory length distribution

**Phase 5: Validation**
- Implement `validate_trajectory()` function
- Add action-level validation
- Implement edge case handling
- Add validation logging

**Phase 6: Statistics & Writing**
- Implement `compute_dataset_statistics()` function
- Implement `DatasetWriter.write()` method
- Add metadata generation
- Implement JSONL writing
- Implement optional Parquet conversion (if enabled in config)

**Phase 7: Integration & Testing**
- Integrate all components in `main()`
- Generate sample dataset (100 trajectories)
- Verify output format
- Test edge cases

### 6.3 Key Implementation Details

**Random Seed Management:**
- Use configurable random seed for reproducibility
- Separate seeds for different variability aspects

**Configuration System:**
- YAML/JSON config file for generator parameters
- Command-line argument support
- Environment variable overrides

**Error Handling:**
- Graceful degradation (skip invalid trajectories)
- Comprehensive logging
- Error reporting in statistics

**Parquet Conversion Implementation:**
- Add `convert_to_parquet()` method to `DatasetWriter` class
- Check config flag: `if config.get('output', {}).get('convert_to_parquet', False)`
- Read JSONL using pandas: `pd.read_json(jsonl_path, lines=True)`
- Flatten nested actions structure (each action becomes a row with trajectory_id)
- Write Parquet: `df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')`
- Handle missing pyarrow gracefully (warn if conversion requested but dependency missing)

**LLM Generation Implementation:**
- Use OpenAI API with appropriate model (e.g., gpt-4o-mini for cost efficiency)
- Temperature settings: configurable
- Prompt engineering: Clear instructions with schema examples
- Error handling: Graceful fallback if LLM calls fail
- Cost optimization: Batch requests where possible, use efficient models (future implementation)

**Performance Considerations:**
- Lazy generation (generate on-demand)
- Streaming writes (don't hold all trajectories in memory)
- LLM API rate limiting and retry logic (future implementation)
- Cache LLM responses for similar requests (optional optimization)

---







## 7. Technical Considerations

### 7.1 Website Type Handling

**E-commerce:**
- Product pages with structured data
- Shopping cart interactions
- Checkout flows
- Search and filtering

**Social Media:**
- Infinite scroll patterns
- Dynamic content loading
- Modal interactions
- Real-time updates

**Productivity Tools:**
- Form-heavy interfaces
- Multi-step wizards
- Drag-and-drop interactions
- Keyboard shortcuts

**Implementation:** LLM generates trajectories with appropriate patterns for each website type. Website type is specified in the prompt, and LLM adapts the action sequences accordingly. Adding new website types requires updating the prompt or configuration.

### 7.2 Element Type Handling

**Canvas Elements:**
- Store canvas context and coordinates
- Include drawing/click coordinates
- Preserve canvas state in metadata

**Iframes:**
- Track frame_id in actions
- Handle cross-frame navigation
- Maintain frame hierarchy

**Single Page Applications (SPAs):**
- Track virtual navigation (hash changes)
- Handle dynamic DOM updates
- Include route information

**Implementation:** Element type is stored in `element_type` field, with type-specific metadata in action parameters.

### 7.3 Intentional vs Exploratory Actions

**Intentional Actions:**
- Direct path to goal
- High confidence scores
- Minimal backtracking
- Clear user_intent labels

**Exploratory Actions:**
- Random browsing
- Multiple back/forward
- Low confidence scores
- No clear user_intent

**Implementation:** `is_intentional` flag and `user_intent` field distinguish these. LLM-generated trajectories include intentional actions by default, with user type influencing the pattern (e.g., first-time users may have more exploratory behavior).

### 7.4 Realism vs Efficiency Trade-off 

**Realism Techniques:**
- LLM-generated trajectories capture realistic user behavior patterns
- LLM generates contextually appropriate URLs, page titles, and element data
- Human-like timing distributions applied post-generation
- Semantic action sequences emerge from LLM understanding of user goals
- Natural variation in action sequences without hardcoded templates

**Efficiency Techniques:**
- Use cost-effective LLM models (e.g., gpt-4o-mini)
- Optimize prompts to reduce token usage
- Batch similar requests where possible
- Streaming writes to avoid memory issues
- Efficient data structures for trajectory conversion

**Balance:** Use LLM for trajectory structure and element data generation to maximize realism. Apply temporal sequencing and validation post-generation for efficiency and consistency.

### 7.5 Real User Data Adaptation

**Changes Needed:**
1. **Privacy**: Anonymize PII in text inputs
2. **Sanitization**: Remove sensitive data (passwords, tokens)
3. **Normalization**: Standardize selectors across browsers
4. **Validation**: More robust validation for real-world edge cases
5. **Storage**: Consider encryption for sensitive trajectories

**Implementation:** Validation layer can be extended with privacy filters and sanitization rules.

---

## 8. ML Training Considerations

### 8.1 Model Training Scenarios

**1. Next Action Prediction:**
- Input: Previous N actions + current context
- Output: Next action type and target element
- Use: Autocomplete, suggestion systems

**2. Element Selection:**
- Input: Current page state + user intent
- Output: Best element to interact with
- Use: Automated task execution

**3. Intent Classification:**
- Input: Action sequence
- Output: User intent label
- Use: Goal understanding, workflow recognition

**4. Timing Prediction:**
- Input: Action history + current state
- Output: Time until next action
- Use: Wait time optimization

### 8.2 Dataset Splits (added based on common training split)

- **Train**: 70% (70 trajectories)
- **Validation**: 15% (15 trajectories)
- **Test**: 15% (15 trajectories)

Split by trajectory (not action) to prevent data leakage.

---

## 9. Dependencies

### 9.1 Required Packages

```txt
# Core
dataclasses>=0.8          # Already in stdlib (Python 3.7+)
typing>=3.7.0             # Already in stdlib
json                      # Already in stdlib
random                    # Already in stdlib
datetime                  # Already in stdlib

# Data Generation
faker>=18.0.0             # Fallback text generation
numpy>=1.24.0             # Numerical operations
openai>=1.0.0             # OpenAI API for LLM-based generation

# Validation
jsonschema>=4.17.0        # JSON schema validation

# Optional (for Parquet conversion)
pyarrow>=12.0.0           # Parquet support (required if convert_to_parquet enabled)
pandas>=2.0.0             # Data analysis and Parquet conversion (required if convert_to_parquet enabled)
```

### 9.2 Development Dependencies

```txt
pytest>=7.0.0             # Testing
black>=23.0.0             # Code formatting
mypy>=1.0.0               # Type checking
```

---

## 10. Success Criteria

### 10.1 Functional Requirements

- ✅ Generate at least 100 trajectories
- ✅ Trajectory lengths between 3-10 actions
- ✅ Multiple workflow types represented
- ✅ Realistic action sequences
- ✅ Valid data format (JSONL)
- ✅ Optional Parquet conversion (if enabled in config)
- ✅ Comprehensive statistics

### 10.2 Quality Requirements

- ✅ All trajectories pass validation
- ✅ Realistic timing and sequences
- ✅ Varied user behavior patterns
- ✅ Edge cases handled gracefully
- ✅ Clear documentation

### 10.3 ML-Readiness Requirements

- ✅ Rich feature set for model training
- ✅ Temporal relationships preserved
- ✅ Intent signals included
- ✅ Scalable data format
- ✅ Reproducible generation

---

## 11. Future Enhancements

1. **Enhanced LLM Integration**: Expand LLM usage for more complex scenarios (multi-step workflows, error recovery patterns, advanced user behaviors)
2. **Visual Features**: Add screenshot embeddings for visual element selection
3. **Multi-modal**: Include image/audio/voice interaction patterns
4. **A/B Testing**: Generate variants for model comparison
5. **Domain Adaptation**: Fine-tune generation for specific websites

### 11.1 Real User Data Integration

**Objective:** Allow seamless ingestion of anonymized real user interaction data into the synthetic generation pipeline.

**Considerations:**

- **Privacy**: Strip personally identifiable information (PII) and sensitive fields (emails, passwords, tokens).
- **Sanitization & Normalization**: Standardize selectors, URLs, and DOM structure across sessions.
- **Hybrid Datasets**: Mix synthetic and real trajectories to improve model robustness.
- **Schema Alignment**: Ensure real user data maps cleanly to `BrowserAction` and `Trajectory` schemas.

### 11.2 Enhanced Validation

**Additional Checks:**

- **Cross-trajectory consistency**: Validate that related workflows don't contradict each other (e.g., cart never emptied mid-transaction).
- **Temporal realism**: Check for unrealistic action intervals or simultaneous conflicting actions.
- **Domain-specific rules**: Workflow-specific constraints (e.g., e-commerce: checkout only after add-to-cart).
- **Selector sanity**: Validate element selectors against real or reference DOM structures.
- **LLM output verification**: Automatically detect hallucinated URLs, page titles, or action sequences.

### 11.3 Prevention of Model Overfitting

**Synthetic Variability:**

- Ensure multiple paths to the same goal exist (trajectory diversification).
- Increase action sequence diversity for similar workflows.
- Vary element selectors, URLs, and page titles even for similar actions.
- Introduce controlled randomness in user behavior patterns.

### 11.6 Feedback Loop for Data Quality

**Objective:** Implement mechanisms to monitor ML model performance on generated datasets and iteratively improve data quality.

**Components:**

- **Performance Monitoring**: Track ML model performance metrics (accuracy, loss, generalization) on generated datasets.
- **Iterative Refinement**: Use model performance signals to iteratively refine generation parameters and LLM prompts.
- **Human-in-the-Loop Validation**: Enable human validation for high-value workflows to identify quality issues and improve generation.
- **Quality Metrics**: Define and track data quality metrics that correlate with model performance.
- **Automated Feedback**: Automatically adjust generation parameters based on model performance feedback.

---

## 12. Risk Mitigation

**Risk 1: Unrealistic Data**
- **Mitigation**: Use real-world patterns from Web Arena and similar datasets as reference
- **Validation**: Manual review of sample trajectories

**Risk 2: Overfitting to Templates**
- **Mitigation**: High variability in action sequences and parameters
- **Validation**: Check action type diversity in statistics

**Risk 3: Performance Issues**
- **Mitigation**: Streaming writes, efficient data structures
- **Validation**: Profile generation time for 100 trajectories

**Risk 4: Schema Evolution**
- **Mitigation**: Versioned schema, backward compatibility
- **Validation**: Schema validation on write

---

## Appendix A: Example Trajectory

```json
{
  "trajectory_id": "traj_001",
  "session_id": "session_abc123",
  "workflow_type": "e_commerce",
  "domain": "example-store.com",
  "start_time": 1704067200000,
  "end_time": 1704067265000,
  "duration": 65.0,
  "user_type": "casual",
  "goal": "purchase_product",
  "goal_achieved": true,
  "actions": [
    {
      "timestamp": 1704067200000,
      "action_type": "navigate",
      "url": "https://example-store.com",
      "page_title": "Example Store - Home",
      "action_id": "action_001"
    },
    {
      "timestamp": 1704067202000,
      "action_type": "type",
      "element_type": "input",
      "element_selector": "#search-input",
      "value": "laptop",
      "url": "https://example-store.com",
      "action_id": "action_002"
    }
    // ... more actions
  ]
}
```

---

## Appendix B: Configuration Example

```json
{
  "generator": {
    "seed": 42,
    "n_trajectories": 100,
    "min_actions": 3,
    "max_actions": 10,
    "workflow_distribution": {
      "e_commerce": 0.4,
      "form_filling": 0.3,
      "research": 0.3
    }
  },
  "actions": {
    "typing_speed_wpm": [150, 300],
    "reading_time_seconds": [2, 5],
    "decision_time_seconds": [0.5, 3.0]
  },
  "output": {
    "format": "jsonl",
    "path": "output/trajectories.jsonl",
    "include_dom_snapshot": false,
    "convert_to_parquet": false  # Optional: convert JSONL to Parquet after generation
  }
}
```

---

**End of Specification**

