import os
import math
import torch
import csv
import json
from datetime import datetime
from torch.amp import autocast, GradScaler
from lr_finder import LRFinder
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import ResNet50
from dataset import get_data_loaders

def save_epoch_metrics(metrics, filepath):
    """Save metrics to CSV file"""
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
    writer,
    checkpoint_dir,
    logs_dir
):
    # Create gradient scaler for AMP
    scaler = GradScaler()
    best_acc = 0.0
    
    # Create log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = os.path.join(logs_dir, f'training_metrics_{timestamp}.csv')
    training_config = {
        'start_time': timestamp,
        'num_epochs': num_epochs,
        'optimizer': optimizer.__class__.__name__,
        'scheduler': scheduler.__class__.__name__,
        'batch_size': train_loader.batch_size,
        'device': str(device)
    }
    with open(os.path.join(logs_dir, f'training_config_{timestamp}.json'), 'w') as f:
        json.dump(training_config, f, indent=4)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'acc': f"{(running_corrects.double() / total_samples).item()*100:.2f}%"
            })
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.2f}%')
        
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad(), autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
        
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)
        
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%')
        
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Save metrics to CSV
        metrics = {
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'train_acc': epoch_acc.item(),
            'val_loss': val_loss,
            'val_acc': val_acc.item(),
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        save_epoch_metrics(metrics, metrics_file)
        
        # Save checkpoint if best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'train_loss': epoch_loss,
                'train_acc': epoch_acc.item(),
                'val_loss': val_loss,
                'val_acc': val_acc.item()
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'best_model_{timestamp}.pth'))
    
    return model

def main():
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(batch_size=128)
    
    # Create model with adjusted BN momentum
    model = ResNet50().to(device)
    
    if torch.cuda.is_available():
        model = torch.compile(model)
    
    # Adjust batch norm momentum
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.1
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    
    print("Running learning rate finder...")
    lr_finder = LRFinder(model, optimizer, criterion, device)
    lr_finder.range_test(train_loader, end_lr=1, num_iter=200, smooth_f=0.05)
    
    min_loss_lr = lr_finder.history['lr'][lr_finder.history['loss'].index(min(lr_finder.history['loss']))]
    suggested_lr = min_loss_lr / 10
    
    print(f"\nLR Finder Results:")
    print(f"Minimum loss achieved at learning rate: {min_loss_lr:.6f}")
    print(f"Suggested learning rate (min_loss_lr/10): {suggested_lr:.6f}")
    
    lr_finder.plot()
    lr_finder.reset()
    
    init_lr = suggested_lr
    optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=1e-4)
    
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=init_lr * 1.5,
        epochs=100,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        div_factor=10,
        final_div_factor=1e3
    )
    
    # TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f'runs/resnet50_training_{timestamp}')
    
    # Train the model
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=1,
        device=device,
        writer=writer,
        checkpoint_dir='checkpoints',
        logs_dir='logs'
    )
    
    writer.close()

if __name__ == '__main__':
    main()