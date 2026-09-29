import os
import collections
from pathlib import Path
from PIL import Image

def analyze_patterns(filenames):
    patterns = collections.Counter()
    for name in filenames:
        # Simplistic pattern extraction: just grab the non-numeric prefix
        prefix = ''.join([c for c in name if not c.isdigit()])
        patterns[prefix] += 1
    return patterns

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    base_dir = PROJECT_ROOT / "datasets"
    
    splits = ["train", "validation", "test"]
    classes = ["fake", "real"]
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    total_fake = 0
    total_real = 0
    all_filenames = []
    duplicate_filenames = []
    unreadable_files = []
    
    print("\n=== DATASET AUDIT ===")
    
    for split in splits:
        for cls in classes:
            folder_path = base_dir / split / cls
            print(f"\n{split}/{cls}:")
            
            if not folder_path.exists():
                print(f"  Status: DIRECTORY MISSING")
                continue
                
            if not folder_path.is_dir():
                print(f"  Status: PATH IS NOT A DIRECTORY")
                continue
                
            images = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in extensions]
            count = len(images)
            
            print(f"  total: {count}")
            
            if cls == "fake":
                total_fake += count
            else:
                total_real += count
                
            filenames = [f.name for f in images]
            all_filenames.extend(filenames)
            
            patterns = analyze_patterns(filenames)
            print("  filename patterns:")
            for pattern, c in patterns.most_common():
                print(f"    {pattern.strip()}: {c} files")
                
            # Readability check
            bad_count = 0
            for img_path in images:
                try:
                    with Image.open(img_path) as img:
                        img.verify() # verify integrity
                except Exception as e:
                    unreadable_files.append((img_path, str(e)))
                    bad_count += 1
            if bad_count > 0:
                print(f"  unreadable files: {bad_count}")

    print("\n=== OVERALL STATISTICS ===")
    print(f"Total fake images: {total_fake}")
    print(f"Total real images: {total_real}")
    print(f"Total images: {total_fake + total_real}")
    
    # Check duplicates across the entire dataset
    counts = collections.Counter(all_filenames)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    
    if duplicates:
        print(f"\nFound {len(duplicates)} duplicate filenames across dataset:")
        # Show top 5
        for i, (fname, count) in enumerate(duplicates.items()):
            if i >= 5:
                print("  ... and more")
                break
            print(f"  {fname}: {count} occurrences")
    else:
        print("\nNo duplicate filenames found across dataset.")
        
    if unreadable_files:
        print(f"\nFound {len(unreadable_files)} unreadable/corrupted images:")
        for path, err in unreadable_files[:10]:
            print(f"  {path.name}: {err}")
        if len(unreadable_files) > 10:
            print("  ... and more")
    else:
        print("\nAll images are readable and uncorrupted.")

if __name__ == "__main__":
    main()
