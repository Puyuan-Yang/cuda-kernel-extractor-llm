import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SOURCE_DIRECTORY = r"/group/ossdphi_algo_scratch_15/puyuyang/cuda-kernel-extractor-llm/source_projects"
# SOURCE_DIRECTORY = r"/group/ossdphi_algo_scratch_15/puyuyang/cuda-kernel-extractor-llm/source_projects/mmcv/mmcv/ops/csrc/common/cuda/spconv"
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")

FILE_INVENTORY_PATH = os.path.join(OUTPUT_ROOT, "cuda_files_inventory.json")
EXTRACTION_RESULTS_DIR = os.path.join(OUTPUT_ROOT, "extraction_results")
EXTRACTED_KERNELS_DIR = os.path.join(OUTPUT_ROOT, "extracted_kernels")

PROMPT_TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "template", "EN", "v1")
SYSTEM_PROMPT_PATH = os.path.join(PROMPT_TEMPLATE_DIR, "system_prompt.txt")
TASK_PROMPT_PATH = os.path.join(PROMPT_TEMPLATE_DIR, "task_prompt.txt")

CUDA_EXTENSIONS = [".cu", ".cuh"]

MAX_WORKERS = 8

LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(OUTPUT_ROOT, "extractor.log")
