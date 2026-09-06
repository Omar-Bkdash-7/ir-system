"""
Path resolution for dataset directories and files.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from common.config import get_config


logger = logging.getLogger(__name__)


class DatasetPathResolver:
    """
    Resolves and manages IR datasets home directory paths.
    
    Implements path resolution strategy for dataset storage locations,
    supporting both environment-based and configuration-based paths.
    """
    
    def __init__(self):
        """Initialize path resolver with configuration."""
        self._ir_home_cache: Optional[Path] = None
        self._metadata_path_cache: Optional[Path] = None
    
    def get_ir_home(self) -> Path:
        """
        Get IR datasets home directory from environment or config.
        
        Re-reads the env var at call time for container/test compatibility.
        Caches the result for performance optimization.
        
        Returns:
            Path to IR_DATASETS_HOME directory
        """
        home_path = os.environ.get(
            "IR_DATASETS_HOME",
            str(Path(get_config("paths.data_root", "data")) / "datasets" / ".ir_datasets")
        )
        return Path(home_path)
    
    def get_metadata_path(self) -> Path:
        """
        Get path to metadata directory.
        
        Returns:
            Path object pointing to metadata directory location
        """
        return Path(get_config("paths.metadata_dir", "data/metadata"))
    
    def ensure_directory_exists(self, path: Path) -> None:
        """
        Ensure that directory exists, create if necessary.
        
        Args:
            path: Directory path to ensure exists
        """
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")
