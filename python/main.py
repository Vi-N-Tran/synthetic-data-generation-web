"""
Browser Trajectory Dataset Generator

Main entry point for generating synthetic browser interaction trajectories.
"""

import json
import os
import sys
from typing import Dict, Any, Optional

# Import from src modules
from src.generator import TrajectoryGenerator
from src.writer import DatasetWriter
from src.validator import validate_trajectory, validate_dataset
from src.statistics import compute_dataset_statistics
from src.schema import Trajectory


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to config JSON file
        
    Returns:
        Configuration dictionary
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    # Default configuration
    return {
        "generator": {
            "seed": 42,
            "n_trajectories": 1, # 100
            "min_actions": 3,
            "max_actions": 10,
            "workflow_distribution": {
                "e_commerce": 0.4,
                "form_filling": 0.3,
                "research": 0.3
            }
        },
        "output": {
            "format": "jsonl",
            "path": "output/trajectories",
            "sample_size": 1, # 10
            "include_dom_snapshot": False
        }
    }


def main():
    """
    Main execution function.
    Generates synthetic browser trajectory dataset using LLM-based generation.
    """
    print("=" * 60)
    print("Browser Trajectory Dataset Generator")
    print("=" * 60)
    
    # Load configuration
    config_path = os.path.join("config", "generator_config.json") if os.path.exists("config") else None
    config = load_config(config_path)
    
    # Get API key (OpenAI or OpenRouter) from env var or config file
    use_openrouter = config.get('generator', {}).get('use_openrouter', False)
    
    if use_openrouter:
        api_key = os.getenv("OPENROUTER_API_KEY")
        api_key_path = os.path.join("config", "openrouter_api_key.txt")
        key_name = "OPENROUTER_API_KEY"
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        api_key_path = os.path.join("config", "openai_api_key.txt")
        key_name = "OPENAI_API_KEY"
    
    # Try to load from config file if not in environment
    if not api_key and os.path.exists(api_key_path):
        with open(api_key_path, 'r') as f:
            api_key = f.read().strip()
    
    if not api_key:
        print(f"Error: {key_name} not found.")
        print("Please set it either:")
        print(f"  1. Environment variable: export {key_name}='your-api-key'")
        print(f"  2. Or create {api_key_path} with your API key")
        sys.exit(1)
    
    # Initialize generator
    print("\n1. Initializing trajectory generator...")
    generator = TrajectoryGenerator(config=config, api_key=api_key, use_openrouter=use_openrouter)
    
    # Generate dataset
    n_trajectories = config.get('generator', {}).get('n_trajectories', 100)
    print(f"\n2. Generating {n_trajectories} trajectories using LLM...")
    print("   (This may take several minutes depending on dataset size)")
    
    try:
        trajectories = generator.generate_dataset(n_trajectories=n_trajectories)
        print(f"   ✓ Generated {len(trajectories)} trajectories")
    except Exception as e:
        print(f"   ✗ Error generating trajectories: {e}")
        sys.exit(1)
    
    # Validate data
    print("\n3. Validating trajectories...")
    valid_trajectories = [t for t in trajectories if validate_trajectory(t)]
    invalid_count = len(trajectories) - len(valid_trajectories)
    
    if invalid_count > 0:
        print(f"   ⚠ {invalid_count} trajectories failed validation and were excluded")
    else:
        print(f"   ✓ All {len(valid_trajectories)} trajectories passed validation")
    
    if not valid_trajectories:
        print("   ✗ No valid trajectories generated. Exiting.")
        sys.exit(1)
    
    # Compute statistics
    print("\n4. Computing dataset statistics...")
    stats = compute_dataset_statistics(valid_trajectories)
    print(f"   Statistics:")
    print(f"   - Total trajectories: {stats['n_trajectories']}")
    print(f"   - Total actions: {stats['n_actions']}")
    print(f"   - Avg trajectory length: {stats['avg_trajectory_length']} actions")
    print(f"   - Workflow distribution: {stats['workflow_type_distribution']}")
    print(f"   - Goal achievement rate: {stats['goal_achievement_rate']:.1%}")
    
    # Save dataset
    print("\n5. Saving dataset...")
    output_config = config.get('output', {})
    output_path = output_config.get('path', 'output/trajectories')
    format_type = output_config.get('format', 'jsonl')
    
    writer = DatasetWriter(output_path=output_path, config=config)
    
    try:
        writer.write(valid_trajectories)
        
        # Write metadata
        writer.write_metadata(valid_trajectories, config)
        
        # Write statistics
        writer.write_statistics(stats)
        
        print(f"   ✓ Dataset saved in {format_type.upper()} format")
    except Exception as e:
        print(f"   ✗ Error saving dataset: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    output_files = writer.get_output_files()
    for file_type, file_path in output_files.items():
        if file_type == 'sample':
            print(f"  - {file_path} (sample for inspection)")
        elif file_type == 'main':
            print(f"  - {file_path}")
        else:
            print(f"  - {file_path}")


if __name__ == "__main__":
    main()
