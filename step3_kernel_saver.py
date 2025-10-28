import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

from config_project import EXTRACTION_RESULTS_DIR, EXTRACTED_KERNELS_DIR


class KernelSaver:
    
    def __init__(self, extraction_dir: str, output_dir: str):
        self.extraction_dir = Path(extraction_dir)
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        if not self.extraction_dir.exists():
            raise ValueError(f"Extraction results directory does not exist: {self.extraction_dir}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def sanitize_filename(self, name: str) -> str:
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
        sanitized = re.sub(r'_+', '_', sanitized)
        return sanitized
    
    def check_name_conflicts(self, extraction_results: List[Dict]) -> Dict[str, List[str]]:
        name_to_sources = defaultdict(list)
        
        for result in extraction_results:
            source_file = result.get('source_file', 'unknown')
            for kernel in result.get('kernels', []):
                func_name = kernel.get('func_name', '')
                if func_name:
                    name_to_sources[func_name].append(source_file)
        
        conflicts = {
            name: sources for name, sources in name_to_sources.items()
            if len(sources) > 1
        }
        
        if conflicts:
            self.logger.warning(f"Found {len(conflicts)} kernel name conflicts")
            for name, sources in conflicts.items():
                self.logger.warning(f"  - {name}: appears in {len(sources)} files")
        
        return conflicts
    
    def generate_unique_filename(self, kernel_name: str, source_file: str, 
                                 conflicts: Dict[str, List[str]]) -> str:
        if kernel_name in conflicts:
            source_name = Path(source_file).stem
            unique_name = f"{source_name}_{kernel_name}"
            self.logger.debug(f"Conflict detected, generate unique name: {unique_name}")
        else:
            unique_name = kernel_name
        
        return self.sanitize_filename(unique_name)
    
    def load_extraction_results(self) -> List[Dict]:
        results = []
        
        for json_file in self.extraction_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    results.append(result)
                    self.logger.debug(f"Load extraction result: {json_file}")
            except Exception as e:
                self.logger.error(f"Failed to load extraction result: {json_file}, error: {e}")
        
        self.logger.info(f"Loaded {len(results)} extraction result files")
        return results
    
    def save_kernels(self) -> Dict[str, any]:
        extraction_results = self.load_extraction_results()
        
        if not extraction_results:
            self.logger.warning("No extraction results found, skip saving")
            return {
                'total_files': 0,
                'total_kernels': 0,
                'saved_kernels': 0,
                'conflicts': 0
            }
        
        conflicts = self.check_name_conflicts(extraction_results)
        
        total_files = len(extraction_results)
        total_kernels = 0
        saved_kernels = 0
        saved_files = []
        
        for result in extraction_results:
            source_file = result.get('source_file', 'unknown')
            kernels = result.get('kernels', [])
            total_kernels += len(kernels)
            
            for kernel in kernels:
                func_name = kernel.get('func_name', '')
                func_content = kernel.get('func_content', '')
                
                if not func_name or not func_content:
                    self.logger.warning(f"Skip invalid kernel: {func_name or '(unnamed)'}")
                    continue
                
                try:
                    unique_filename = self.generate_unique_filename(
                        func_name, source_file, conflicts
                    )
                    output_filename = f"{unique_filename}.cu"
                    output_path = self.output_dir / output_filename

                    processed_content = re.sub(r'#include\s+.*', '', func_content, flags=re.MULTILINE)

                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(processed_content)
                    
                    saved_kernels += 1
                    saved_files.append(str(output_path))
                    self.logger.info(f"✓ Saved kernel: {output_filename}")
                    
                except Exception as e:
                    self.logger.error(f"✗ Failed to save kernel: {func_name}, error: {e}")
        
        manifest = {
            'total_source_files': total_files,
            'total_kernels_extracted': total_kernels,
            'successfully_saved': saved_kernels,
            'conflicts_detected': len(conflicts),
            'saved_files': saved_files
        }
        
        manifest_path = self.output_dir / "kernel_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Kernel manifest saved: {manifest_path}")
        
        return {
            'total_files': total_files,
            'total_kernels': total_kernels,
            'saved_kernels': saved_kernels,
            'conflicts': len(conflicts)
        }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Step 3: Start saving individual kernel files")
    logger.info("=" * 60)
    
    try:
        saver = KernelSaver(EXTRACTION_RESULTS_DIR, EXTRACTED_KERNELS_DIR)
        
        stats = saver.save_kernels()
        
        logger.info("=" * 60)
        logger.info(f"✓ Step 3 completed!")
        logger.info(f"  - Processed source files: {stats['total_files']}")
        logger.info(f"  - Total extracted kernels: {stats['total_kernels']}")
        logger.info(f"  - Successfully saved: {stats['saved_kernels']}")
        logger.info(f"  - Name conflicts: {stats['conflicts']}")
        logger.info(f"  - Output directory: {EXTRACTED_KERNELS_DIR}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"✗ Step 3 failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

