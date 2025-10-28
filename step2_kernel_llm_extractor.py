import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config_project import (
    FILE_INVENTORY_PATH, EXTRACTION_RESULTS_DIR, MAX_WORKERS,
    SYSTEM_PROMPT_PATH, TASK_PROMPT_PATH
)
from template import prompt_loader
from step1_cu_file_collector import FileCollector
from llm_generator import LLMGenerator


class LLMExtractor:
    
    def __init__(self, llm_config: Dict):
        self.logger = logging.getLogger(__name__)
        
        self.generator = LLMGenerator(llm_config)
        self.system_prompt = prompt_loader.load_prompt(SYSTEM_PROMPT_PATH)
        self.task_prompt_template = prompt_loader.load_prompt(TASK_PROMPT_PATH)
        
        self.logger.info(f"LLM extractor initialized, model: {llm_config.get('model_id')}")
    
    def read_file_content(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def _filter_files_with_kernels(self, file_paths: List[str]) -> List[str]:
        filtered = []
        for file_path in file_paths:
            try:
                content = self.read_file_content(file_path)
                if '__global__' in content:
                    filtered.append(file_path)
                else:
                    self.logger.debug(f"Skip (no kernel): {file_path}")
            except Exception as e:
                self.logger.warning(f"Read failed, skip: {file_path}, error: {e}")
        
        return filtered
    
    def extract_kernels_from_file(self, file_path: str) -> Optional[Dict]:
        self.logger.info(f"Start processing file: {file_path}")
        
        try:
            code_content = self.read_file_content(file_path)
            
            prompt = self.task_prompt_template.format(
                file_path=file_path,
                code_content=code_content
            )
            
            self.logger.debug(f"Call LLM API, file: {file_path}")
            
            result_text = self.generator.generate(prompt, self.system_prompt)
            
            if not result_text:
                self.logger.error(f"✗ Empty response: {file_path}")
                return None
            
            if result_text.startswith("```"):
                lines = result_text.split('\n')
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                result_text = '\n'.join(lines)
            
            result = json.loads(result_text)
            
            self.logger.info(f"✓ Successfully extracted {len(result.get('kernels', []))} kernels: {file_path}")
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"✗ JSON parse failed: {file_path}, error: {e}")
            self.logger.debug(f"Raw response: {result_text[:500]}...")
            return None
        except Exception as e:
            self.logger.error(f"✗ Extraction failed: {file_path}, error: {e}", exc_info=True)
            return None
    
    def extract_batch(self, file_paths: List[str], output_dir: str) -> Dict[str, Dict]:
        os.makedirs(output_dir, exist_ok=True)
        
        filtered_paths = self._filter_files_with_kernels(file_paths)
        
        self.logger.info(f"Pre-filter result: {len(filtered_paths)}/{len(file_paths)} files contain kernels")
        
        results = {}
        success_count = 0
        fail_count = 0
        
        self.logger.info(f"Start batch extraction, total {len(filtered_paths)} files, max workers: {MAX_WORKERS}")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {
                executor.submit(self.extract_kernels_from_file, file_path): file_path
                for file_path in filtered_paths
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                
                try:
                    result = future.result()
                    
                    if result is not None:
                        results[file_path] = result
                        success_count += 1
                        
                        output_filename = Path(file_path).stem + ".json"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                        
                        self.logger.debug(f"Extraction result saved: {output_path}")
                    else:
                        fail_count += 1
                        
                except Exception as e:
                    self.logger.error(f"Processing failed: {file_path}, error: {e}")
                    fail_count += 1
        
        self.logger.info(f"Batch extraction completed: success {success_count}, failed {fail_count}")
        return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Step 2: Start extracting kernels using LLM")
    logger.info("=" * 60)
    
    try:
        with open("config_llm.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        llm_config = config["providers"]["openai"]
        logger.info(f"Loaded LLM config: {llm_config.get('model_id')}")
        
        logger.info(f"Load file inventory: {FILE_INVENTORY_PATH}")
        inventory = FileCollector.load_inventory(FILE_INVENTORY_PATH)
        file_paths = inventory['files']
        # TODO: remove this after testing
        # file_paths = file_paths[:3]  # only test the first 3 files

        logger.info(f"Total {len(file_paths)} files to process")
        
        extractor = LLMExtractor(llm_config)
        
        start_time = time.time()
        results = extractor.extract_batch(file_paths, EXTRACTION_RESULTS_DIR)
        elapsed_time = time.time() - start_time
        
        total_kernels = sum(len(r.get('kernels', [])) for r in results.values())
        
        logger.info("=" * 60)
        logger.info(f"✓ Step 2 completed!")
        logger.info(f"  - Processed files: {len(results)}/{len(file_paths)}")
        logger.info(f"  - Total extracted kernels: {total_kernels}")
        logger.info(f"  - Time elapsed: {elapsed_time:.2f} seconds")
        logger.info(f"  - Results saved to: {EXTRACTION_RESULTS_DIR}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"✗ Step 2 failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

