from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import h5py
import json
from pathlib import Path

class NCTDataset(torch.utils.data.Dataset):
    def __init__(self, h5_dir):
        self.h5_dir = h5_dir
        self.h5_files = list(Path(h5_dir).glob("*.h5"))
        self.features, self.labels = self.load_data()

    def load_data(self):
        features, labels = [], []
        class_mapping = {cls.stem: idx for idx, cls in enumerate(self.h5_files)}

        for file in self.h5_files:
            print(f"Reading {file}")
            with h5py.File(file, 'r') as f:
                feats = f['feats'][:]
                lbls = np.full(feats.shape[0], class_mapping[file.stem])
                features.append(feats)
                labels.append(lbls)

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        return features, labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class NCTClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NCTClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No dropout or softmax here
        return x

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device='cpu', save_plot_path=None):
    model = model.to(device)
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    train_f1, val_f1 = [], []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
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

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_f1 = f1_score(all_labels, all_preds, average=None)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc.item())
                train_f1.append(epoch_f1)
                scheduler.step()
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc.item())
                val_f1.append(epoch_f1)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1}')

    # Print values for each class
    print("Training F1 scores per class:", train_f1)
    print("Validation F1 scores per class:", val_f1)

    # Plotting training and validation curves for each class
    epochs_range = range(len(train_loss))

    for cls in range(len(train_f1[0])):
        train_f1_cls = [f1[cls] for f1 in train_f1]
        val_f1_cls = [f1[cls] for f1 in val_f1]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_f1_cls, label=f'Training F1 Class {cls}')
        plt.plot(epochs_range, val_f1_cls, label=f'Validation F1 Class {cls}')
        plt.legend(loc='upper right')
        plt.title(f'Training and Validation F1 Score for Class {cls}')

        if save_plot_path:
            plt.savefig(f"{save_plot_path}_class_{cls}.png")
            print(f'Training curves for class {cls} saved to {save_plot_path}_class_{cls}.png')
        else:
            plt.show()

    return model

def evaluate_model(model, dataloader, criterion, device='cpu'):
    model.eval()
    test_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = test_loss / len(dataloader.dataset)
    test_acc = running_corrects.double() / len(dataloader.dataset)
    test_f1 = f1_score(all_labels, all_preds, average=None)

    return test_loss, test_acc.item(), test_f1

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset and adjust splits
    dataset = NCTDataset(h5_dir=args.h5_dir)
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    }

    # Model, criterion, optimizer, and scheduler
    model = NCTClassifier(input_dim=dataset[0][0].shape[0], num_classes=args.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train model
    model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=args.num_epochs, device=device, save_plot_path=args.save_plot_path)

    # Save model
    torch.save(model.state_dict(), args.model_save_path)

    # Evaluate on test set
    test_loss, test_acc, test_f1 = evaluate_model(model, dataloaders['test'], criterion, device)
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} F1: {test_f1}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_dir', type=str, required=True, help='Directory containing H5 files')
    parser.add_argument('--num_classes', type=int, default=9, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--save_plot_path', type=str, default=None, help='Path to save training curves plot')
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to save trained model')
    args = parser.parse_args()
    main(args)

