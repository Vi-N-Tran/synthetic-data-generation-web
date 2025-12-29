# Browser Trajectory Dataset Generator

LLM-based synthetic browser interaction trajectory generator for training browser automation models.

## Overview

This pipeline generates realistic browser interaction trajectories using OpenAI's API. The LLM generates complete trajectory structures (sequences of actions) and detailed element data (selectors, URLs, page titles) based on the defined schemas.

## Features

- **LLM-Based Generation**: Uses OpenAI API to generate realistic trajectories and element data
- **Multiple Workflow Types**: E-commerce, form filling, and research workflows
- **Multiple Output Formats**: JSONL (default) or Parquet with optional sample JSONL
- **Comprehensive Validation**: Validates trajectories for consistency and correctness
- **Rich Statistics**: Computes detailed statistics on generated datasets
- **User Behavior Patterns**: Supports power_user, casual, and first_time user types

## Setup

### Prerequisites

- Python 3.7+
- OpenAI API key

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Set your OpenAI API key:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Optional: Edit `config/generator_config.json` to customize generation parameters.

## Usage

### Basic Usage

```bash
python main.py
```

This will:
1. Generate 100 trajectories (configurable)
2. Validate all trajectories
3. Compute statistics
4. Save to `output/` directory

### Configuration Options

Edit `config/generator_config.json`:

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
  "output": {
    "format": "jsonl",  // or "parquet"
    "path": "output",
    "sample_size": 10
  }
}
```

## Output Format

### JSONL Format (default)

- `output/trajectories.jsonl`: One trajectory per line
- `output/metadata.json`: Dataset metadata
- `output/statistics.json`: Computed statistics

### Parquet Format

- `output/trajectories.parquet`: Flattened actions in Parquet format
- `output/trajectories_sample.jsonl`: Sample for human inspection
- `output/metadata.json`: Dataset metadata
- `output/statistics.json`: Computed statistics

## Project Structure

```
python/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── config/
│   └── generator_config.json
├── src/
│   ├── schema.py          # BrowserAction and Trajectory dataclasses
│   ├── llm_generator.py   # LLM-based data generation
│   ├── actions.py         # Action creation utilities
│   ├── generator.py       # TrajectoryGenerator using LLM
│   ├── validator.py       # Validation logic
│   ├── statistics.py      # Statistics computation
│   ├── writer.py          # DatasetWriter (JSONL/Parquet)
│   └── utils.py           # Helper functions
└── output/                # Generated datasets (gitignored)
```

## Data Schema

See `SPEC.md` for detailed schema documentation. Key components:

- **BrowserAction**: Single browser interaction (click, type, navigate, etc.)
- **Trajectory**: Sequence of actions forming a user session
- **Quality Metrics**: Computed properties (action_count, avg_action_interval, etc.)

## Notes

- Generation time depends on dataset size and API response times
- LLM API costs apply (uses OpenAI API)
- For large datasets, consider using Parquet format for efficiency
- All trajectories are validated before saving

