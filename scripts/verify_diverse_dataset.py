import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def get_images(directory):
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.avif'}
    if not os.path.exists(directory):
        return []
    images = []
    for f in os.listdir(directory):
        if os.path.splitext(f)[1].lower() in valid_exts:
            images.append(f)
    return images

def main():
    base_dir = os.path.join('datasets', 'diverse')
    splits = ['train', 'validation', 'test']
    classes = ['fake', 'real']
    
    counts = {s: {c: 0 for c in classes} for s in splits}
    
    all_filenames = set()
    split_filenames = {s: set() for s in splits}
    duplicates = []
    corrupted = []
    
    for split in splits:
        for cls in classes:
            dir_path = os.path.join(base_dir, split, cls)
            images = get_images(dir_path)
            counts[split][cls] = len(images)
            
            for img_name in images:
                img_path = os.path.join(dir_path, img_name)
                
                # Check duplicates globally
                if img_name in all_filenames:
                    duplicates.append(img_path)
                all_filenames.add(img_name)
                
                # Add to split specific set for overlap check
                split_filenames[split].add(img_name)
                
                # Verify readability
                try:
                    with Image.open(img_path) as img:
                        img.load()
                        img.verify()
                except Exception as e:
                    corrupted.append((img_path, str(e)))

    # Check overlap across splits
    overlap = False
    if len(split_filenames['train'].intersection(split_filenames['validation'])) > 0:
        overlap = True
        print("Overlap found between train and validation.")
    if len(split_filenames['train'].intersection(split_filenames['test'])) > 0:
        overlap = True
        print("Overlap found between train and test.")
    if len(split_filenames['validation'].intersection(split_filenames['test'])) > 0:
        overlap = True
        print("Overlap found between validation and test.")

    print("\n=== DIVERSE DATASET AUDIT ===")
    
    print("\nCounts:")
    for split in splits:
        print(f"  {split.capitalize()}:")
        for cls in classes:
            print(f"    {cls}: {counts[split][cls]}")
            
    print(f"\nTotal unique filenames: {len(all_filenames)}")
    print(f"Total duplicate filenames found: {len(duplicates)}")
    if duplicates:
        print("Duplicates (showing up to 10):")
        for d in duplicates[:10]:
            print(f"  {d}")
            
    print(f"Total corrupted images found: {len(corrupted)}")
    if corrupted:
        print("Corrupted images (showing up to 10):")
        for f, err in corrupted[:10]:
            print(f"  {f}: {err}")
            
    print("\nSplit Overlap Status:")
    if overlap:
        print("  WARNING: Overlap detected across splits!")
    else:
        print("  OK: No filename overlap across train/validation/test splits.")

if __name__ == "__main__":
    main()
