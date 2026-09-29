import os
import random
import shutil
import hashlib
import csv
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def get_hash(filepath):
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def gather_sources():
    """Gather all available source images into a dictionary."""
    sources = {'fake': [], 'real': []}
    
    # 1. CIFAKE
    cifake_base = os.path.join('CIFAKE-Real-and-AI-Generated-Synthetic-Images', 'DATASET')
    for split in ['train', 'test']:
        for cls in ['FAKE', 'REAL']:
            path = os.path.join(cifake_base, split, cls)
            if os.path.exists(path):
                norm_cls = cls.lower()
                for f in os.listdir(path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        sources[norm_cls].append({
                            'path': os.path.join(path, f),
                            'source': 'CIFAKE'
                        })
                        
    # 2. HF Dataset (existing train/val/test)
    hf_base = 'datasets'
    for split in ['train', 'validation', 'test']:
        for cls in ['fake', 'real']:
            path = os.path.join(hf_base, split, cls)
            if os.path.exists(path):
                for f in os.listdir(path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        sources[cls].append({
                            'path': os.path.join(path, f),
                            'source': 'HF_Dataset'
                        })
                        
    return sources

def main():
    random.seed(42)
    
    out_dir = os.path.join('datasets', 'diverse_v2')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        
    os.makedirs(out_dir, exist_ok=True)
    
    metadata_path = os.path.join(out_dir, 'metadata.csv')
    
    # Targets for the larger dataset
    # We will aim for 60,000 images total (20k/5k/5k per class)
    targets = {
        'train': 20000,
        'validation': 5000,
        'test': 5000
    }
    
    print("Gathering sources...")
    sources = gather_sources()
    
    for cls in ['fake', 'real']:
        # Sort for determinism before shuffle
        sources[cls].sort(key=lambda x: x['path'])
        random.shuffle(sources[cls])
        print(f"Found {len(sources[cls])} source images for {cls}.")

    seen_hashes = set()
    global_count = {'fake': 0, 'real': 0}
    
    with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'class', 'split', 'source', 'hash'])
        
        for cls in ['fake', 'real']:
            src_list = sources[cls]
            src_idx = 0
            
            for split in ['train', 'validation', 'test']:
                target_count = targets[split]
                dest_dir = os.path.join(out_dir, split, cls)
                os.makedirs(dest_dir, exist_ok=True)
                
                added = 0
                while added < target_count and src_idx < len(src_list):
                    item = src_list[src_idx]
                    src_idx += 1
                    
                    src_path = item['path']
                    source_name = item['source']
                    
                    try:
                        file_hash = get_hash(src_path)
                        if file_hash in seen_hashes:
                            continue  # Skip content duplicate
                            
                        with Image.open(src_path) as img:
                            img.load()
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                                
                            global_count[cls] += 1
                            out_filename = f"dv2_{cls}_{global_count[cls]:06d}.jpg"
                            out_path = os.path.join(dest_dir, out_filename)
                            
                            img.save(out_path, 'JPEG', quality=95)
                            
                        seen_hashes.add(file_hash)
                        writer.writerow([out_filename, cls, split, source_name, file_hash])
                        added += 1
                        
                    except Exception as e:
                        # Skip corrupted
                        pass
                
                print(f"Processed {split}/{cls}: {added}/{target_count} images.")

    print("\nDataset generation complete.")
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    main()
