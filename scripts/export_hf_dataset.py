import os
import random
import traceback
from pathlib import Path
from datasets import load_dataset
from PIL import Image

def export_split(dataset_split, target_dir, num_samples, class_label, class_name):
    # Filter indices matching the class_label
    indices = [i for i, item in enumerate(dataset_split) if item['label'] == class_label]
    
    # Shuffle indices
    random.shuffle(indices)
    
    # Select num_samples
    if len(indices) < num_samples:
        print(f"Warning: Not enough samples for {class_name}. Requested {num_samples}, found {len(indices)}.")
        selected_indices = indices
    else:
        selected_indices = indices[:num_samples]
        
    remaining_indices = indices[num_samples:]
    
    out_dir = target_dir / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting {len(selected_indices)} {class_name} images to {out_dir}...")
    success_count = 0
    
    for count, idx in enumerate(selected_indices):
        try:
            item = dataset_split[idx]
            img = item['image']
            
            filename = f"{class_name}_{count:05d}.jpg"
            out_path = out_dir / filename
            
            if out_path.exists():
                print(f"Skipping {out_path}, already exists.")
                continue
                
            img_rgb = img.convert("RGB")
            img_rgb.save(out_path, "JPEG", quality=95)
            success_count += 1
            
            if (count + 1) % 500 == 0:
                print(f"  Processed {count + 1}/{len(selected_indices)} images...")
                
        except Exception as e:
            print(f"Error processing {class_name} image index {idx}: {e}")
            traceback.print_exc()
            
    print(f"Finished exporting {success_count} {class_name} images to {out_dir}.")
    return remaining_indices, success_count

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    base_out_dir = PROJECT_ROOT / "datasets"
    
    TRAIN_PER_CLASS = 4000
    VAL_PER_CLASS = 1000
    TEST_PER_CLASS = 1000
    
    random.seed(42)
    
    print("Loading Hugging Face dataset...")
    try:
        ds = load_dataset("julienlucas/midjourney-dalle-sd-nanobananapro-dataset")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        traceback.print_exc()
        return

    # Check splits
    if 'train' not in ds or 'test' not in ds:
        print("Dataset missing 'train' or 'test' splits.")
        return

    # Filter real/fake sets from train split
    # label 0 = fake, label 1 = real
    print("Filtering train/val splits...")
    train_split = ds['train']
    
    fake_indices = [i for i, item in enumerate(train_split) if item['label'] == 0]
    real_indices = [i for i, item in enumerate(train_split) if item['label'] == 1]
    
    random.shuffle(fake_indices)
    random.shuffle(real_indices)
    
    # Train
    train_fake_idx = fake_indices[:TRAIN_PER_CLASS]
    train_real_idx = real_indices[:TRAIN_PER_CLASS]
    
    # Val (non-overlapping)
    val_fake_idx = fake_indices[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS]
    val_real_idx = real_indices[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS]

    def process_subset(dataset_split, indices, target_dir, class_name):
        out_dir = target_dir / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Exporting {len(indices)} {class_name} images to {out_dir}...")
        
        success_count = 0
        for count, idx in enumerate(indices):
            try:
                img = dataset_split[idx]['image']
                filename = f"{class_name}_{count:05d}.jpg"
                out_path = out_dir / filename
                
                if out_path.exists():
                    # print(f"Skipping {out_path}, already exists.") # Mute to avoid spam
                    success_count += 1
                    continue
                    
                img_rgb = img.convert("RGB")
                img_rgb.save(out_path, "JPEG", quality=95)
                success_count += 1
                
                if (count + 1) % 500 == 0:
                    print(f"  Processed {count + 1}/{len(indices)} images...")
            except Exception as e:
                print(f"Error processing {class_name} image index {idx}: {e}")
                
        return success_count

    print("\n--- Exporting Training Set ---")
    train_dir = base_out_dir / "train"
    train_fake_count = process_subset(train_split, train_fake_idx, train_dir, "fake")
    train_real_count = process_subset(train_split, train_real_idx, train_dir, "real")
    
    print("\n--- Exporting Validation Set ---")
    val_dir = base_out_dir / "validation"
    val_fake_count = process_subset(train_split, val_fake_idx, val_dir, "fake")
    val_real_count = process_subset(train_split, val_real_idx, val_dir, "real")
    
    print("\n--- Exporting Test Set ---")
    test_dir = base_out_dir / "test"
    test_split = ds['test']
    
    test_fake_indices = [i for i, item in enumerate(test_split) if item['label'] == 0]
    test_real_indices = [i for i, item in enumerate(test_split) if item['label'] == 1]
    
    random.shuffle(test_fake_indices)
    random.shuffle(test_real_indices)
    
    test_fake_idx = test_fake_indices[:TEST_PER_CLASS]
    test_real_idx = test_real_indices[:TEST_PER_CLASS]
    
    test_fake_count = process_subset(test_split, test_fake_idx, test_dir, "fake")
    test_real_count = process_subset(test_split, test_real_idx, test_dir, "real")

    print("\n--- Summary ---")
    print(f"train/fake: {train_fake_count}")
    print(f"train/real: {train_real_count}")
    print(f"validation/fake: {val_fake_count}")
    print(f"validation/real: {val_real_count}")
    print(f"test/fake: {test_fake_count}")
    print(f"test/real: {test_real_count}")
    
    print("\nExport complete!")

if __name__ == "__main__":
    main()
