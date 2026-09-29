import os
import random
import shutil
from PIL import Image

# Suppress decompression bomb warnings
Image.MAX_IMAGE_PIXELS = None

def get_image_files(directory):
    """Recursively get all supported image files in a directory."""
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.avif'}
    image_paths = []
    if not os.path.exists(directory):
        return image_paths
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_exts:
                image_paths.append(os.path.join(root, file))
    return image_paths

def process_and_copy(src_paths, dest_dir, prefix, target_count, start_idx=0):
    """
    Process images from src_paths and save them as RGB JPEGs to dest_dir.
    Returns (number_successfully_processed, list_of_skipped_files, list_of_remaining_src_paths).
    """
    os.makedirs(dest_dir, exist_ok=True)
    count = 0
    skipped = []
    processed_paths = set()
    
    # We iterate over a copy or use index, but we return remaining
    idx = 0
    while count < target_count and idx < len(src_paths):
        src_path = src_paths[idx]
        idx += 1
        
        try:
            with Image.open(src_path) as img:
                # Force load to catch corruption early
                img.load()
                
                # Convert to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Format output filename
                out_filename = f"{prefix}_{(start_idx + count + 1):06d}.jpg"
                out_path = os.path.join(dest_dir, out_filename)
                
                # Save as JPEG
                img.save(out_path, 'JPEG', quality=95)
                
                # Verification step
                with Image.open(out_path) as verify_img:
                    verify_img.verify()
                    
                count += 1
                processed_paths.add(out_path)
                
        except Exception as e:
            skipped.append((src_path, str(e)))
            
    remaining_paths = src_paths[idx:]
    return count, skipped, remaining_paths, processed_paths

def main():
    random.seed(42)
    
    # Define source directories
    src_dirs = {
        'fake': [
            os.path.join('datasets', 'train', 'fake'),
            os.path.join('datasets', 'validation', 'fake')
        ],
        'real': [
            os.path.join('datasets', 'train', 'real'),
            os.path.join('datasets', 'validation', 'real')
        ]
    }
    
    # Define targets
    targets = {
        'train': 4000,
        'validation': 1000,
        'test': 1000
    }
    
    # Base output directory
    base_out_dir = os.path.join('datasets', 'diverse')
    
    if os.path.exists(base_out_dir):
        shutil.rmtree(base_out_dir)
    
    summary = {
        'train': {'fake': 0, 'real': 0},
        'validation': {'fake': 0, 'real': 0},
        'test': {'fake': 0, 'real': 0}
    }
    
    total_skipped = 0
    all_output_paths = set()
    
    # Process both classes
    for cls in ['fake', 'real']:
        print(f"\nGathering {cls} source images...")
        all_cls_paths = []
        for d in src_dirs[cls]:
            all_cls_paths.extend(get_image_files(d))
            
        # Remove duplicates
        all_cls_paths = list(set(all_cls_paths))
        # Deterministic sort before shuffle for full reproducibility
        all_cls_paths.sort()
        random.shuffle(all_cls_paths)
        
        print(f"Found {len(all_cls_paths)} source images for {cls}.")
        
        remaining_src = all_cls_paths
        global_count = 0
        
        for split in ['train', 'validation', 'test']:
            target_num = targets[split]
            dest_dir = os.path.join(base_out_dir, split, cls)
            prefix = f"hf_{cls}"
            
            print(f"Processing {split}/{cls} (Target: {target_num})...")
            
            processed_count, skipped, remaining_src, out_paths = process_and_copy(
                remaining_src, 
                dest_dir, 
                prefix, 
                target_num,
                start_idx=global_count
            )
            global_count += processed_count
            
            summary[split][cls] = processed_count
            total_skipped += len(skipped)
            
            # Record output paths to check duplicates/overlap
            for p in out_paths:
                if p in all_output_paths:
                    print(f"ERROR: Duplicate output path detected: {p}")
                all_output_paths.add(p)
                
            for skip_file, err in skipped:
                print(f"Skipped {skip_file}: {err}")
                
            if processed_count < target_num:
                print(f"WARNING: Only processed {processed_count}/{target_num} for {split}/{cls} due to insufficient source files or read errors.")

    # Print Summary
    print("\n=== DIVERSE DATASET SUMMARY ===")
    print("\nTrain:")
    print(f"  fake: {summary['train']['fake']}")
    print(f"  real: {summary['train']['real']}")
    
    print("\nValidation:")
    print(f"  fake: {summary['validation']['fake']}")
    print(f"  real: {summary['validation']['real']}")
    
    print("\nTest:")
    print(f"  fake: {summary['test']['fake']}")
    print(f"  real: {summary['test']['real']}")
    
    total_images = sum(summary[split][cls] for split in summary for cls in summary[split])
    print(f"\nTotal images: {total_images}")
    print(f"Skipped/corrupted: {total_skipped}")

if __name__ == "__main__":
    main()
