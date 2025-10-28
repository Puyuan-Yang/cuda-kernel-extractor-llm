import re
from pathlib import Path
import argparse


def clean_headers(file_path: Path) -> bool:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        content = re.sub(r'#include\s+<ATen/[^>]+>\s*\n?', '', content, flags=re.IGNORECASE)
        
        content = re.sub(r'#include\s+<c10/[^>]+>\s*\n?', '', content, flags=re.IGNORECASE)
        
        content = re.sub(r'#include\s+<torch/[^>]+>\s*\n?', '', content, flags=re.IGNORECASE)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Processing failed {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Clean PyTorch-related headers in CUDA files')
    parser.add_argument('directory', nargs='?', 
                       default=r'E:\Python Code\pythonProject\LLM\cuda-kernel-extractor-llm\output\extracted_kernels',
                       help='Target directory path')
    
    args = parser.parse_args()
    target_dir = Path(args.directory)
    
    if not target_dir.exists():
        print(f"Error: Directory does not exist {target_dir}")
        return
    
    cu_files = list(target_dir.glob('*.cu'))
    
    if not cu_files:
        print(f"No .cu files found: {target_dir}")
        return
    
    print(f"Found {len(cu_files)} .cu files")
    print("Start cleaning headers...")
    
    modified_count = 0
    for cu_file in cu_files:
        if clean_headers(cu_file):
            modified_count += 1
            print(f"✓ {cu_file.name}")
        else:
            print(f"- {cu_file.name} (no modification needed)")
    
    print(f"\nCompleted! Modified {modified_count}/{len(cu_files)} files")


if __name__ == "__main__":
    main()

