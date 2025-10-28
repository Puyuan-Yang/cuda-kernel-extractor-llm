import os
import json
from pathlib import Path
from typing import List, Dict
import logging

from config_project import SOURCE_DIRECTORY, FILE_INVENTORY_PATH, CUDA_EXTENSIONS, OUTPUT_ROOT


class FileCollector:
    
    def __init__(self, source_dir: str, output_path: str):
        self.source_dir = Path(source_dir)
        self.output_path = Path(output_path)
        self.logger = logging.getLogger(__name__)
        
        if not self.source_dir.exists():
            raise ValueError(f"Source directory does not exist: {self.source_dir}")
    
    def collect_cuda_files(self) -> List[str]:
        cuda_files = []
        
        self.logger.info(f"Start scanning directory: {self.source_dir}")
        
        all_cuda_files = []
        for ext in CUDA_EXTENSIONS:
            pattern = f"**/*{ext}"
            files = list(self.source_dir.rglob(pattern))
            
            for file_path in files:
                abs_path = str(file_path.absolute())
                all_cuda_files.append(abs_path)
                self.logger.debug(f"Found file: {abs_path}")
        
        all_cuda_files.sort()
        self.logger.info(f"Total found {len(all_cuda_files)} CUDA files")
        
        for file_path in all_cuda_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '__global__' in content:
                        cuda_files.append(file_path)
                        self.logger.debug(f"Keep file with __global__: {file_path}")
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                        if '__global__' in content:
                            cuda_files.append(file_path)
                            self.logger.debug(f"Keep file with __global__: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Cannot read file, skip: {file_path}, error: {e}")
            except Exception as e:
                self.logger.warning(f"Cannot read file, skip: {file_path}, error: {e}")
        
        self.logger.info(f"After filtering, kept {len(cuda_files)}/{len(all_cuda_files)} CUDA files with __global__")
        
        return cuda_files
    
    def generate_inventory(self) -> Dict:
        cuda_files = self.collect_cuda_files()
        
        inventory = {
            "source_directory": str(self.source_dir),
            "total_files": len(cuda_files),
            "filtered_by_global": True,
            "files": cuda_files
        }
        
        return inventory
    
    def save_inventory(self) -> str:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        inventory = self.generate_inventory()
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(inventory, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"File inventory saved to: {self.output_path}")
        self.logger.info(f"Total files: {inventory['total_files']}")
        self.logger.info("Filtered: only files with __global__ keyword")
        
        return str(self.output_path)
    
    @staticmethod
    def load_inventory(inventory_path: str) -> Dict:
        with open(inventory_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Step 1: Start collecting CUDA files")
    logger.info("=" * 60)
    
    try:
        collector = FileCollector(SOURCE_DIRECTORY, FILE_INVENTORY_PATH)
        
        output_path = collector.save_inventory()
        
        logger.info("=" * 60)
        logger.info(f"✓ Step 1 completed! File inventory saved: {output_path}")
        logger.info("✓ Filtered: only kept CUDA files with __global__ keyword")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"✗ Step 1 failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

