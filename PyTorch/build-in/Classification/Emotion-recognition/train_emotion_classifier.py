"""
Description: Train emotion classification model using PyTorch
"""

import os
import numpy as np
import torch
import torch_sdaa
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from load_and_process import load_fer2013
from load_and_process import preprocess_input

import csv
from datetime import datetime

# -------------------------------
# Parameters
# -------------------------------
batch_size = 32
num_epochs = 10000
input_shape = (48, 48)
validation_split = 0.2
num_classes = 7
patience = 50
base_path = 'models/'
os.makedirs(base_path, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# Load and preprocess data
# -------------------------------
faces, emotions = load_fer2013()
faces = preprocess_input(faces)

if len(faces.shape) == 4 and faces.shape[-1] == 1:
    faces = np.squeeze(faces, axis=-1)  # (N, 48, 48)
faces = np.expand_dims(faces, axis=1)   # (N, 1, 48, 48)

x_train, x_test, y_train, y_test = train_test_split(faces, emotions, test_size=validation_split, shuffle=True, random_state=42)

x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).float()
x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).float()

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop((48, 48), scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
])

class TensorTransformDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        
        if x.dim() == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        
        if self.transform:
            x = self.transform(x)
            if x.dim() == 2:
                x = x.unsqueeze(0)
        
        return x, y

train_dataset = TensorTransformDataset(x_train, y_train, transform=train_transform)
test_dataset = TensorTransformDataset(x_test, y_test, transform=None)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------
# Model: mini_XCEPTION (PyTorch version)
# -------------------------------
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class mini_XCEPTION(nn.Module):
    def __init__(self, num_classes=7):
        super(mini_XCEPTION, self).__init__()
        self.num_classes = num_classes

        self.entry_flow = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.middle_flow = nn.Sequential(
            SeparableConv2d(64, 128),
            nn.ReLU(),
            SeparableConv2d(128, 128),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.exit_flow = nn.Sequential(
            SeparableConv2d(128, 256),
            nn.ReLU(),
            SeparableConv2d(256, num_classes),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = self.flatten(x)
        return x


model = mini_XCEPTION(num_classes).to(device)
print(model)

# -------------------------------
# Loss, Optimizer, Scheduler
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                              patience=patience//4, verbose=True)

# -------------------------------
# Callbacks: CSV Logger, Model Checkpoint, EarlyStopping
# -------------------------------
log_file_path = base_path + '_emotion_training.log'
best_val_loss = float('inf')
epochs_no_improve = 0
train_log = []

# -------------------------------
# Training Loop
# -------------------------------
print("Start training...")

for epoch in range(num_epochs):
    # Training
    model.train()
    scaler = torch.sdaa.amp.GradScaler()
    running_loss = 0.0
    correct = 0
    total = 0

    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        targets = targets.argmax(dim=1)  # one-hot -> class index
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        # loss.backward()
        scaler.scale(loss).backward()
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if step % 1 == 0:
            print(f"Epoch {epoch+1}, Step {step}: Loss = {loss.item():.4f}")

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            targets = targets.argmax(dim=1)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(test_loader.dataset)
    val_acc = correct / total

    # Step scheduler
    scheduler.step(val_loss)

    # Log
    log_entry = [epoch, train_loss, train_acc, val_loss, val_acc]
    train_log.append(log_entry)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # ModelCheckpoint: save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(base_path, '_mini_XCEPTION_best.pth'))
        print(f"Saved best model at epoch {epoch+1}")

        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # EarlyStopping
    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    # Save log every epoch
    with open(log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
        writer.writerows(train_log)

print("Training finished.")