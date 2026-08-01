"""
Local filesystem dataset scanning.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any

from services.indexing.dataset.path_resolver import DatasetPathResolver
from services.indexing.dataset.constants import DATA_FILE_EXTENSIONS, MAX_RECURSION_DEPTH


logger = logging.getLogger(__name__)


class LocalDatasetScanner:
    """
    Scans local filesystem for downloaded IR datasets.
    
    Implements recursive directory scanning to discover and catalog
    locally downloaded datasets with their metadata.
    """
    
    def __init__(self):
        """Initialize scanner with path resolver."""
        self.path_resolver = DatasetPathResolver()
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all local datasets with size and file information.
        
        Scans IR_DATASETS_HOME directory recursively and collects
        information about each dataset found.
        
        Returns:
            List of dictionaries containing dataset info (id, path, size, files)
            Sorted by dataset_id for consistent ordering
        """
        home = self.path_resolver.get_ir_home()
        
        logger.info("=" * 80)
        logger.info(f"Starting dataset scan")
        logger.info(f"IR_DATASETS_HOME: {home}")
        logger.info(f"Directory exists: {home.exists()}")
        logger.info(f"Directory is_dir: {home.is_dir() if home.exists() else 'N/A'}")
        
        if not home.exists():
            logger.warning(f"IR datasets home directory not found: {home}")
            return []
        
        # Log immediate children for debugging
        try:
            children = list(home.iterdir())
            logger.info(f"Found {len(children)} items in IR_DATASETS_HOME")
            
            # Separate directories and files
            dirs = [c for c in children if c.is_dir()]
            files = [c for c in children if c.is_file()]
            
            logger.info(f"  Directories: {len(dirs)}")
            for d in dirs[:20]:  # Log first 20 directories
                try:
                    subcount = len(list(d.iterdir()))
                    logger.info(f"    - {d.name}/ ({subcount} items)")
                except Exception as e:
                    logger.info(f"    - {d.name}/ (error: {e})")
            
            if len(dirs) > 20:
                logger.info(f"    ... and {len(dirs) - 20} more directories")
            
            logger.info(f"  Files: {len(files)}")
            for f in files[:10]:  # Log first 10 files
                logger.info(f"    - {f.name}")
            
            if len(files) > 10:
                logger.info(f"    ... and {len(files) - 10} more files")
                
        except Exception as e:
            logger.error(f"Error listing IR_DATASETS_HOME contents: {e}", exc_info=True)
        
        results: List[Dict[str, Any]] = []
        self._collect_datasets(home, [], results, depth=0)
        
        sorted_results = sorted(results, key=lambda x: x["dataset_id"])
        logger.info(f"Scan complete: found {len(sorted_results)} datasets")
        logger.info("=" * 80)
        
        return sorted_results
    
    def _collect_datasets(
        self, 
        base: Path, 
        parts: List[str], 
        acc: List[Dict[str, Any]], 
        depth: int
    ) -> None:
        """
        Recursively collect dataset information from directory tree.
        
        Uses depth-first search with maximum depth limit to prevent
        infinite recursion and handle deep directory structures.
        
        Args:
            base: Current directory being scanned
            parts: Path components forming dataset ID
            acc: Accumulator list for results
            depth: Current recursion depth
        """
        if not base.is_dir():
            logger.debug(f"Skipping {base}: not a directory")
            return
            
        if depth > MAX_RECURSION_DEPTH:
            logger.warning(f"Max recursion depth ({MAX_RECURSION_DEPTH}) reached at {base}")
            return
        
        path_str = "/".join(parts) if parts else "<root>"
        logger.debug(f"[Depth {depth}] Scanning: {path_str} -> {base}")
        
        try:
            children = list(base.iterdir())
        except (PermissionError, OSError) as e:
            logger.warning(f"Cannot access directory {base}: {e}")
            return
        
        subdirs = sorted(
            [c for c in children if c.is_dir()], 
            key=lambda p: p.name
        )
        files = [c for c in children if c.is_file()]
        
        data_files = self._filter_data_files(files)
        
        # Log detailed info for debugging
        logger.debug(
            f"[{path_str}] subdirs={len(subdirs)}, "
            f"files={len(files)}, data_files={len(data_files)}"
        )
        
        # If we have path parts and data files, this is a dataset
        if parts and data_files:
            dataset_id = "/".join(parts)
            logger.info(
                f"✓ Found dataset: {dataset_id} "
                f"({len(data_files)} data files in {base})"
            )
            self._add_dataset_entry(base, parts, acc)
            return
        
        # Continue searching subdirectories
        for sub in subdirs:
            self._collect_datasets(sub, parts + [sub.name], acc, depth + 1)
    
    def _add_dataset_entry(
        self, 
        base: Path, 
        parts: List[str], 
        acc: List[Dict[str, Any]]
    ) -> None:
        """
        Add dataset entry to accumulator with computed statistics.
        
        Args:
            base: Dataset base directory
            parts: Path components forming dataset ID
            acc: Accumulator list
        """
        try:
            all_files = [f for f in base.rglob("*") if f.is_file()]
            all_data = self._filter_data_files(all_files)
            
            if all_data:
                size_bytes = sum(f.stat().st_size for f in all_files)
                
                acc.append({
                    "dataset_id": "/".join(parts),
                    "path": str(base),
                    "size_mb": round(size_bytes / (1024 * 1024), 2),
                    "file_count": len(all_data),
                })
        except (PermissionError, OSError) as e:
            logger.warning(f"Cannot compute stats for {base}: {e}")
    
    @staticmethod
    def _filter_data_files(files: List[Path]) -> List[Path]:
        """
        Filter files to only include actual data files.
        
        Args:
            files: List of file paths to filter
            
        Returns:
            List of paths that are data files
        """
        return [f for f in files if f.suffix in DATA_FILE_EXTENSIONS]
