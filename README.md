# CUDA Kernel Extractor LLM

LLM-powered CUDA Kernel extraction tool

## 📋 Overview

CUDA Kernel Extractor LLM is an intelligent tool that uses OpenAI's GPT models to automatically extract and reconstruct CUDA kernel functions from open-source CUDA code repositories. This tool simplifies complex CUDA code into independent, compilable kernel code snippets.


## 🚀 Quick Start

### Requirements

- Python 3.8+
- OpenAI API key
- CUDA development environment (optional, for validating extraction results)

### Installation

```bash
pip install -r requirements.txt
```

## 📖 Usage Guide

### Complete Processing Pipeline

```bash
# 1. File Collection: Scan CUDA files and generate inventory
python step1_cu_file_collector.py

# 2. LLM Extraction: Use AI to analyze and extract kernels
python step2_kernel_llm_extractor.py

# 3. File Saving: Generate independent kernel files
python step3_kernel_saver.py

# 4. Header Cleanup: Remove PyTorch-related dependencies (optional)
python step4_clean_pytorch_headers.py
```


## 🏗️ Project Structure

```
cuda-kernel-extractor-llm/
├── 📁 llm_providers/          # LLM provider interfaces
│   ├── base_provider.py       # Base interface definition
│   └── openai_provider.py     # OpenAI implementation
├── 📁 template/               # Prompt templates
├── 📁 output/                 # Output directory (auto-generated)
│   ├── cuda_files_inventory.json    # File inventory
│   ├── extraction_results/          # LLM extraction results
│   └── extracted_kernels/           # Final kernel files
├── 📁 source_projects/        # Source code directory
├── config_llm.json           # LLM configuration
├── config_project.py         # Project configuration
├── llm_generator.py          # LLM generator
├── step1_cu_file_collector.py      # Step 1: File collection
├── step2_kernel_llm_extractor.py   # Step 2: LLM extraction
├── step3_kernel_saver.py           # Step 3: File saving
├── step4_clean_pytorch_headers.py  # Step 4: Header cleanup
├── test_model.py             # API test script
└── requirements.txt          # Python dependencies
```
