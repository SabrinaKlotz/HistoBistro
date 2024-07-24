import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

class HDF5Dataset(Dataset):
    def __init__(self, directory):
        self.features = []
        self.labels = []
        self.load_hdf5_files(directory)
        self.features = np.concatenate(self.features, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def load_hdf5_files(self, directory):
        for label, filename in enumerate(os.listdir(directory)):
            if filename.endswith(".h5"):
                filepath = os.path.join(directory, filename)
                with h5py.File(filepath, 'r') as f:
                    self.features.append(np.array(f['feats']))
                    num_samples = self.features[-1].shape[0]
                    self.labels.append(np.full((num_samples,), label))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20):
    train_loss = []
    val_loss = []
    train_f1_scores = []
    val_f1_scores = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_train_labels = []
        all_train_preds = []

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss.append(epoch_loss)

        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
        train_f1_scores.append(train_f1)

        model.eval()
        running_val_loss = 0.0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss.append(epoch_val_loss)

        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        val_f1_scores.append(val_f1)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train F1: {train_f1:.4f}, Val Loss: {epoch_val_loss:.4f}, Val F1: {val_f1:.4f}')

    return train_loss, val_loss, train_f1_scores, val_f1_scores

def plot_metrics(train_loss, val_loss, train_f1_scores, val_f1_scores, save_dir=None):
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(14, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, color='blue', linestyle='-', marker='o', label='Training Loss')
    plt.plot(epochs, val_loss, color='red', linestyle='--', marker='x', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot F1 Score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_f1_scores, color='blue', linestyle='-', marker='o', label='Training F1 Score')
    plt.plot(epochs, val_f1_scores, color='red', linestyle='--', marker='x', label='Validation F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, 'metrics_curves.png')
        plt.savefig(plot_path)
        print(f'Plot saved to {plot_path}')
    
    plt.show()

# Main script
directory = '/mnt/volume/mathias/outputs/patch_uni_output/h5_files'
batch_size = 32
num_epochs = 10
save_plot_dir = '.'
model_save_path = 'nct_classifier_model.pth'

# Load dataset
dataset = HDF5Dataset(directory)

# Perform stratified train-test split
train_indices, val_indices = train_test_split(
    np.arange(len(dataset)), test_size=0.2, stratify=dataset.labels, random_state=42
)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Define model
input_size = dataset.features.shape[1]
num_classes = len(set(dataset.labels))
model = SimpleNN(input_size, num_classes)

# Define loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# Train model
train_loss, val_loss, train_f1_scores, val_f1_scores = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs)

# Save the model
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

# Plot and save loss and F1 score curves
plot_metrics(train_loss, val_loss, train_f1_scores, val_f1_scores, save_dir=save_plot_dir)
