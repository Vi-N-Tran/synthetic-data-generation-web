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
from src.logging_config import setup_logging, get_logger


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
    # Set up logging
    verbose = os.getenv("VERBOSE", "false").lower() == "true"
    logger = setup_logging(level="INFO", verbose=verbose)
    
    logger.info("=" * 60)
    logger.info("Browser Trajectory Dataset Generator")
    logger.info("=" * 60)
    
    # Load configuration
    logger.info("Loading configuration...")
    config_path = os.path.join("config", "generator_config.json") if os.path.exists("config") else None
    config = load_config(config_path)
    logger.debug(f"Config loaded: {json.dumps(config, indent=2)}")
    
    # Get API key (OpenAI or OpenRouter) from env var or config file
    use_openrouter = config.get('generator', {}).get('use_openrouter', False)
    api_source = "OpenRouter" if use_openrouter else "OpenAI"
    logger.info(f"API provider: {api_source}")
    
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
        logger.debug(f"Loading API key from file: {api_key_path}")
        with open(api_key_path, 'r') as f:
            api_key = f.read().strip()
    
    if not api_key:
        logger.error(f"{key_name} not found")
        logger.error("Please set it either:")
        logger.error(f"  1. Environment variable: export {key_name}='your-api-key'")
        logger.error(f"  2. Or create {api_key_path} with your API key")
        sys.exit(1)
    
    logger.debug("API key loaded successfully")
    
    # Initialize generator
    logger.info("Step 1/5: Initializing trajectory generator...")
    generator = TrajectoryGenerator(config=config, api_key=api_key, use_openrouter=use_openrouter)
    logger.debug("TrajectoryGenerator initialized")
    
    # Generate dataset
    n_trajectories = config.get('generator', {}).get('n_trajectories', 100)
    logger.info(f"Step 2/5: Generating {n_trajectories} trajectories using LLM...")
    logger.info("(This may take several minutes depending on dataset size)")
    
    try:
        trajectories, dedup_stats = generator.generate_dataset(n_trajectories=n_trajectories)
        logger.info(f"✓ Generated {len(trajectories)} unique trajectories")
        if dedup_stats.get('exact_duplicates_count', 0) > 0:
            logger.info(f"  Removed {dedup_stats.get('exact_duplicates_count', 0)} exact duplicates")
            logger.debug(f"Deduplication stats: {dedup_stats}")
    except Exception as e:
        logger.error(f"✗ Error generating trajectories: {e}", exc_info=True)
        sys.exit(1)
    
    # Validate data
    logger.info(f"Step 3/5: Validating {len(trajectories)} trajectories...")
    valid_trajectories = []
    invalid_trajectories = []
    
    for i, trajectory in enumerate(trajectories):
        if validate_trajectory(trajectory):
            valid_trajectories.append(trajectory)
        else:
            invalid_trajectories.append(trajectory)
            logger.debug(f"Trajectory {trajectory.trajectory_id} failed validation")
    
    invalid_count = len(invalid_trajectories)
    
    if invalid_count > 0:
        logger.warning(f"⚠ {invalid_count} trajectories failed validation and were excluded")
        logger.debug(f"Invalid trajectory IDs: {[t.trajectory_id for t in invalid_trajectories[:10]]}")
    else:
        logger.info(f"✓ All {len(valid_trajectories)} trajectories passed validation")
    
    if not valid_trajectories:
        logger.error("✗ No valid trajectories generated. Exiting.")
        sys.exit(1)
    
    # Compute statistics
    logger.info(f"Step 4/5: Computing dataset statistics...")
    stats = compute_dataset_statistics(valid_trajectories)
    logger.info("Dataset statistics:")
    logger.info(f"  - Total trajectories: {stats['n_trajectories']}")
    logger.info(f"  - Total actions: {stats['n_actions']}")
    logger.info(f"  - Avg trajectory length: {stats['avg_trajectory_length']} actions")
    logger.info(f"  - Workflow distribution: {stats['workflow_type_distribution']}")
    logger.info(f"  - Goal achievement rate: {stats['goal_achievement_rate']:.1%}")
    logger.debug(f"Full statistics: {json.dumps(stats, indent=2)}")
    
    # Save dataset
    logger.info(f"Step 5/5: Saving dataset...")
    output_config = config.get('output', {})
    output_path = output_config.get('path', 'output/trajectories')
    format_type = output_config.get('format', 'jsonl')
    logger.debug(f"Output path: {output_path}, Format: {format_type}")
    
    writer = DatasetWriter(output_path=output_path, config=config)
    
    try:
        logger.debug(f"Writing {len(valid_trajectories)} trajectories to {format_type} format...")
        writer.write(valid_trajectories)
        
        # Write metadata
        logger.debug("Writing metadata file...")
        writer.write_metadata(valid_trajectories, config)
        
        # Write statistics
        logger.debug("Writing statistics file...")
        writer.write_statistics(stats)
        
        logger.info(f"✓ Dataset saved in {format_type.upper()} format")
    except Exception as e:
        logger.error(f"✗ Error saving dataset: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Generation complete!")
    logger.info("=" * 60)
    logger.info("Output files:")
    output_files = writer.get_output_files()
    for file_type, file_path in output_files.items():
        if file_type == 'sample':
            logger.info(f"  - {file_path} (sample for inspection)")
        elif file_type == 'main':
            logger.info(f"  - {file_path}")
        else:
            logger.info(f"  - {file_path}")


if __name__ == "__main__":
    main()
