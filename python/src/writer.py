"""Dataset writer for JSONL and optional Parquet formats"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.schema import Trajectory
from src.logging_config import get_logger

logger = get_logger('writer')


def _get_timestamped_filename(base_path: str, base_name: str, extension: str) -> str:
    """Generate a timestamped filename to avoid overwriting existing files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if os.path.isdir(base_path):
        return os.path.join(base_path, f"{base_name}_{timestamp}{extension}")
    else:
        # base_path is a file path
        dir_name = os.path.dirname(base_path) or "."
        base_file = os.path.basename(base_path).rsplit('.', 1)[0] if '.' in os.path.basename(base_path) else os.path.basename(base_path)
        return os.path.join(dir_name, f"{base_file}_{timestamp}{extension}")


class DatasetWriter:
    """Handles saving trajectories to disk in JSONL or Parquet format"""
    
    def __init__(self, output_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize dataset writer.
        
        Args:
            output_path: Base output path (directory or file path)
            config: Configuration dict with output settings
        """
        self.output_path = output_path
        self.config = config or {}
        self._actual_output_file = None  # Will be set when writing
        self._sample_file = None  # Will be set when writing Parquet
        self._metadata_file = None  # Will be set when writing metadata
        self._stats_file = None  # Will be set when writing stats
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path) if os.path.isfile(output_path) or '.' in os.path.basename(output_path) else output_path
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    def write(self, trajectories: List[Trajectory]) -> None:
        """
        Write trajectories to disk in configured format.
        
        Args:
            trajectories: List of trajectories to write
        """
        output_config = self.config.get('output', {})
        format_type = output_config.get('format', 'jsonl')
        
        if format_type == 'parquet':
            self._write_parquet(trajectories, output_config)
        else:
            self._write_jsonl(trajectories, output_config)
    
    def _write_jsonl(self, trajectories: List[Trajectory], config: Dict[str, Any]) -> None:
        """Write trajectories to JSONL format"""
        output_file = config.get('path', self.output_path)
        if not output_file.endswith('.jsonl'):
            if os.path.isdir(output_file) or os.path.isdir(self.output_path):
                base_dir = output_file if os.path.isdir(output_file) else self.output_path
                output_file = _get_timestamped_filename(base_dir, 'trajectories', '.jsonl')
            else:
                base_name = output_file.rsplit('.', 1)[0] if '.' in output_file else output_file
                output_file = _get_timestamped_filename(os.path.dirname(output_file) or '.', os.path.basename(base_name), '.jsonl')
        else:
            # Add timestamp to existing .jsonl path
            base_name = output_file.rsplit('.jsonl', 1)[0]
            output_file = _get_timestamped_filename(os.path.dirname(base_name) or '.', os.path.basename(base_name), '.jsonl')
        
        # Ensure directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Store the actual output file path for metadata/stats files
        self._actual_output_file = output_file
        
        logger.info(f"Writing {len(trajectories)} trajectories to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, trajectory in enumerate(trajectories):
                trajectory_dict = trajectory.to_dict()
                json_line = json.dumps(trajectory_dict, ensure_ascii=False)
                f.write(json_line + '\n')
                if (i + 1) % 100 == 0:
                    logger.debug(f"Written {i + 1}/{len(trajectories)} trajectories...")
        
        logger.info(f"✓ Written {len(trajectories)} trajectories to {output_file}")
    
    def _write_parquet(self, trajectories: List[Trajectory], config: Dict[str, Any]) -> None:
        """Write trajectories to Parquet format with optional JSONL sample"""
        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            print("Warning: pyarrow and pandas required for Parquet format. Falling back to JSONL.")
            self._write_jsonl(trajectories, config)
            return
        
        output_file = config.get('path', self.output_path)
        if not output_file.endswith('.parquet'):
            if os.path.isdir(output_file) or os.path.isdir(self.output_path):
                base_path = output_file if os.path.isdir(output_file) else self.output_path
                output_file = _get_timestamped_filename(base_path, 'trajectories', '.parquet')
            else:
                base_name = output_file.rsplit('.', 1)[0] if '.' in output_file else output_file
                output_file = _get_timestamped_filename(os.path.dirname(output_file) or '.', os.path.basename(base_name), '.parquet')
        else:
            # Add timestamp to existing .parquet path
            base_name = output_file.rsplit('.parquet', 1)[0]
            output_file = _get_timestamped_filename(os.path.dirname(base_name) or '.', os.path.basename(base_name), '.parquet')
        
        # Ensure directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Store the actual output file path
        self._actual_output_file = output_file
        
        logger.info(f"Writing {len(trajectories)} trajectories to Parquet format...")
        
        # Flatten trajectories into actions with trajectory metadata
        logger.debug("Flattening trajectories into action rows...")
        rows = []
        for trajectory in trajectories:
            trajectory_dict = trajectory.to_dict()
            for action in trajectory.actions:
                action_dict = action.to_dict()
                # Add trajectory metadata to each action
                row = {
                    'trajectory_id': trajectory.trajectory_id,
                    'trajectory_workflow_type': trajectory.workflow_type,
                    'trajectory_domain': trajectory.domain,
                    'trajectory_user_type': trajectory.user_type,
                    'trajectory_goal': trajectory.goal,
                    'trajectory_goal_achieved': trajectory.goal_achieved,
                    **action_dict
                }
                rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Write to Parquet
        logger.debug(f"Writing DataFrame with {len(df)} rows to Parquet file...")
        df.to_parquet(
            output_file,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        logger.info(f"✓ Written {len(trajectories)} trajectories ({len(df)} actions) to {output_file}")
        
        # Generate sample JSONL file for human inspection
        sample_size = config.get('sample_size', 10)
        sample_trajectories = trajectories[:sample_size] if len(trajectories) >= sample_size else trajectories
        
        # Use same timestamp for sample file
        base_name = output_file.rsplit('.parquet', 1)[0]
        sample_file = f"{base_name}_sample.jsonl"
        self._sample_file = sample_file
        logger.info(f"Generating sample file with {len(sample_trajectories)} trajectories...")
        # Temporarily override the write_jsonl to use the exact sample filename
        with open(sample_file, 'w', encoding='utf-8') as f:
            for trajectory in sample_trajectories:
                trajectory_dict = trajectory.to_dict()
                json_line = json.dumps(trajectory_dict, ensure_ascii=False)
                f.write(json_line + '\n')
        logger.info(f"✓ Sample written to {sample_file}")
    
    def write_metadata(self, trajectories: List[Trajectory], generator_config: Dict[str, Any]) -> None:
        """Write dataset metadata file"""
        # Use timestamped filename based on actual output file
        if self._actual_output_file:
            base_name = self._actual_output_file.rsplit('.', 1)[0]
            metadata_file = f"{base_name}_metadata.json"
        else:
            # Fallback to timestamped name
            base_path = self.output_path if os.path.isdir(self.output_path) else os.path.dirname(self.output_path) or '.'
            metadata_file = _get_timestamped_filename(base_path, 'metadata', '.json')
        
        self._metadata_file = metadata_file
        
        metadata = {
            "dataset_version": "1.0",
            "generation_date": datetime.utcnow().isoformat() + "Z",
            "generator_config": generator_config,
            "total_trajectories": len(trajectories),
            "total_actions": sum(len(t.actions) for t in trajectories),
            "schema_version": "1.0"
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Metadata written to {metadata_file}")
    
    def write_statistics(self, statistics: Dict[str, Any]) -> None:
        """Write computed statistics to file"""
        # Use timestamped filename based on actual output file
        if self._actual_output_file:
            base_name = self._actual_output_file.rsplit('.', 1)[0]
            stats_file = f"{base_name}_statistics.json"
        else:
            # Fallback to timestamped name
            base_path = self.output_path if os.path.isdir(self.output_path) else os.path.dirname(self.output_path) or '.'
            stats_file = _get_timestamped_filename(base_path, 'statistics', '.json')
        
        self._stats_file = stats_file
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Statistics written to {stats_file}")
    
    def get_output_files(self) -> Dict[str, str]:
        """Get the actual output file paths that were written"""
        files = {}
        if self._actual_output_file:
            files['main'] = self._actual_output_file
        if self._sample_file:
            files['sample'] = self._sample_file
        if self._metadata_file:
            files['metadata'] = self._metadata_file
        if self._stats_file:
            files['statistics'] = self._stats_file
        return files

