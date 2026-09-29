import os
import csv
import hashlib
from collections import defaultdict
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def get_hash(filepath):
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def main():
    base_dir = os.path.join('datasets', 'diverse_v2')
    metadata_path = os.path.join(base_dir, 'metadata.csv')
    
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist.")
        return

    splits = ['train', 'validation', 'test']
    classes = ['fake', 'real']
    
    counts = {s: {c: 0 for c in classes} for s in splits}
    
    all_filenames = set()
    all_hashes = set()
    split_hashes = {s: set() for s in splits}
    duplicates_filename = []
    duplicates_content = []
    corrupted = []
    
    # Read metadata if exists
    sources = defaultdict(int)
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sources[row['source']] += 1

    for split in splits:
        for cls in classes:
            dir_path = os.path.join(base_dir, split, cls)
            if not os.path.exists(dir_path):
                continue
                
            for img_name in os.listdir(dir_path):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.avif')):
                    continue
                    
                img_path = os.path.join(dir_path, img_name)
                counts[split][cls] += 1
                
                # Check filename duplicates
                if img_name in all_filenames:
                    duplicates_filename.append(img_path)
                all_filenames.add(img_name)
                
                # Verify readability and check content hash
                try:
                    with Image.open(img_path) as img:
                        img.load()
                        img.verify()
                        
                    file_hash = get_hash(img_path)
                    
                    if file_hash in all_hashes:
                        duplicates_content.append(img_path)
                    all_hashes.add(file_hash)
                    
                    split_hashes[split].add(file_hash)
                        
                except Exception as e:
                    corrupted.append((img_path, str(e)))

    # Check content overlap across splits
    overlap = False
    if len(split_hashes['train'].intersection(split_hashes['validation'])) > 0:
        overlap = True
        print("CONTENT Overlap found between train and validation.")
    if len(split_hashes['train'].intersection(split_hashes['test'])) > 0:
        overlap = True
        print("CONTENT Overlap found between train and test.")
    if len(split_hashes['validation'].intersection(split_hashes['test'])) > 0:
        overlap = True
        print("CONTENT Overlap found between validation and test.")

    print("\n=== DIVERSE V2 DATASET AUDIT ===")
    
    print("\nCounts:")
    for split in splits:
        print(f"  {split.capitalize()}:")
        for cls in classes:
            print(f"    {cls}: {counts[split][cls]}")
            
    total_images = sum(counts[s][c] for s in splits for c in classes)
    print(f"\nTotal images: {total_images}")
    
    print(f"Total unique filenames: {len(all_filenames)}")
    print(f"Duplicate filenames found: {len(duplicates_filename)}")
    
    print(f"Total unique image contents (hashes): {len(all_hashes)}")
    print(f"Content duplicates found: {len(duplicates_content)}")
            
    print(f"Total corrupted images found: {len(corrupted)}")
    if corrupted:
        print("Corrupted images (showing up to 10):")
        for f, err in corrupted[:10]:
            print(f"  {f}: {err}")
            
    print("\nSplit Overlap Status:")
    if overlap:
        print("  WARNING: Content overlap detected across splits!")
    else:
        print("  OK: No content overlap across train/validation/test splits.")
        
    print("\nSource Generator Distribution (from metadata):")
    if sources:
        for src, count in sources.items():
            print(f"  {src}: {count}")
    else:
        print("  No metadata available.")

if __name__ == "__main__":
    main()
