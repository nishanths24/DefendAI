import argparse
import os
import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import precision_score, recall_score, f1_score

def get_data_loaders(data_dir, batch_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'validation']}
    
    # Filter to only keep fake_*.jpg and real_*.jpg
    for phase in ['train', 'validation']:
        ds = image_datasets[phase]
        filtered_samples = []
        for path, target in ds.samples:
            filename = os.path.basename(path)
            if filename.startswith("fake_") or filename.startswith("real_"):
                filtered_samples.append((path, target))
        ds.samples = filtered_samples
        ds.targets = [s[1] for s in filtered_samples]
        
        print(f"[{phase}] Expected fake_*: {ds.targets.count(0)}")
        print(f"[{phase}] Expected real_*: {ds.targets.count(1)}")
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=0)
                  for x in ['train', 'validation']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=5, save_path='model.pth', checkpoint_path='checkpoint.pth', start_epoch=0, best_acc=0.0, history=None):
    if history is None:
        history = []
        
    best_model_wts = copy.deepcopy(model.state_dict())

    if start_epoch >= num_epochs:
        print(f"Training is already complete. Requested {num_epochs} epochs, currently at epoch {start_epoch}.")
        return model

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        epoch_stats = {'epoch': epoch + 1}

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                del inputs, labels, outputs, preds, loss

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
            epoch_recall = recall_score(all_labels, all_preds, zero_division=0)
            epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Prec: {epoch_precision:.4f} Rec: {epoch_recall:.4f} F1: {epoch_f1:.4f}')
            
            epoch_stats[f'{phase}_loss'] = epoch_loss
            epoch_stats[f'{phase}_acc'] = epoch_acc.item()
            epoch_stats[f'{phase}_prec'] = epoch_precision
            epoch_stats[f'{phase}_rec'] = epoch_recall
            epoch_stats[f'{phase}_f1'] = epoch_f1

            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, save_path)
                print(f"Saved new best model to {save_path}")

        history.append(epoch_stats)

        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'history': history
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved training checkpoint to {checkpoint_path}")

    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

def main():
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model on HF Dataset')
    parser.add_argument('--data_dir', type=str, default=str(project_root / 'datasets'), help='Path to datasets directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_path', type=str, default=str(project_root / 'backend' / 'models' / 'deepfake_model_hf.pth'), help='Path to save best model')
    parser.add_argument('--checkpoint_path', type=str, default=str(project_root / 'backend' / 'models' / 'training_checkpoint_hf.pth'), help='Path to save training checkpoint')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders, dataset_sizes, class_names = get_data_loaders(args.data_dir, args.batch_size)
    print(f"Classes: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_acc = 0.0
    history = []

    if os.path.exists(args.checkpoint_path):
        print(f"Found checkpoint at {args.checkpoint_path}. Loading...")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        history = checkpoint['history']
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        print("No checkpoint found. Starting from scratch.")

    train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, args.epochs, args.save_path, args.checkpoint_path, start_epoch, best_acc, history)

if __name__ == '__main__':
    main()
