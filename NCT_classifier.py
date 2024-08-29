import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import random
import json

def log_params_and_metrics(log_file_path, params, metrics):
    with open(log_file_path, 'w') as log_file:
        log_data = {
            "parameters": params,
            "metrics": metrics
        }
        json.dump(log_data, log_file, indent=4)

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeding everything to seed {seed}")
    return

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
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        #x = self.dropout(x)
        x = self.relu(self.fc2(x))
        #x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    train_loss = []
    val_loss = []
    train_f1_scores = []
    val_f1_scores = []
    train_acc_scores = []
    val_acc_scores = []
    train_prec_scores = []
    val_prec_scores = []
    train_rec_scores = []
    val_rec_scores = []

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
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_prec = precision_score(all_train_labels, all_train_preds, average='weighted', zero_division=1)
        train_rec = recall_score(all_train_labels, all_train_preds, average='weighted')

        train_f1_scores.append(train_f1)
        train_acc_scores.append(train_acc)
        train_prec_scores.append(train_prec)
        train_rec_scores.append(train_rec)

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
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_prec = precision_score(all_val_labels, all_val_preds, average='weighted', zero_division=1)
        val_rec = recall_score(all_val_labels, all_val_preds, average='weighted')

        val_f1_scores.append(val_f1)
        val_acc_scores.append(val_acc)
        val_prec_scores.append(val_prec)
        val_rec_scores.append(val_rec)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}')
        print(f'Train F1: {train_f1:.4f}, Val Loss: {epoch_val_loss:.4f}, Val F1: {val_f1:.4f}')
        print(f'Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}')
        print(f'Train Precision: {train_prec:.4f}, Val Precision: {val_prec:.4f}')
        print(f'Train Recall: {train_rec:.4f}, Val Recall: {val_rec:.4f}')

    return train_loss, val_loss, train_f1_scores, val_f1_scores, train_acc_scores, val_acc_scores, train_prec_scores, val_prec_scores, train_rec_scores, val_rec_scores

def test_model(model, test_loader):
    model.eval()
    all_test_labels = []
    all_test_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(preds.cpu().numpy())

    test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
    test_acc = accuracy_score(all_test_labels, all_test_preds)
    test_prec = precision_score(all_test_labels, all_test_preds, average='weighted', zero_division=1)
    test_rec = recall_score(all_test_labels, all_test_preds, average='weighted')

    print(f'Test F1: {test_f1:.4f}, Test Accuracy: {test_acc:.4f}')
    print(f'Test Precision: {test_prec:.4f}, Test Recall: {test_rec:.4f}')

    return test_f1, test_acc, test_prec, test_rec

def plot_metrics(train_loss, val_loss, train_f1_scores, val_f1_scores, 
                 train_acc_scores, val_acc_scores, train_prec_scores, val_prec_scores, 
                 train_rec_scores, val_rec_scores, save_dir=None):
    
    # For loss, assume starting at some initial high value
    initial_loss = max(train_loss[0], val_loss[0]) * 1.1  # Or set a specific high value if desired
    train_loss = [initial_loss] + train_loss
    val_loss = [initial_loss] + val_loss

    # Prepend 0 to each metric list to ensure they start at 0
    train_f1_scores = [0] + train_f1_scores
    val_f1_scores = [0] + val_f1_scores
    train_acc_scores = [0] + train_acc_scores
    val_acc_scores = [0] + val_acc_scores
    train_prec_scores = [0] + train_prec_scores
    val_prec_scores = [0] + val_prec_scores
    train_rec_scores = [0] + train_rec_scores
    val_rec_scores = [0] + val_rec_scores

    epochs = range(len(train_loss))
    
    plt.figure(figsize=(14, 10))

    # Plot Loss
    plt.subplot(2, 2, 1)  # Adjusting subplot positions for clarity
    plt.plot(epochs, train_loss, color='red', label='Training Loss')
    plt.plot(epochs, val_loss, color='magenta', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.ylim(0, max(max(train_loss), max(val_loss)))
    plt.xlim(0, len(epochs)-1) 
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Plot F1 Score
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_f1_scores, color='blue', label='Training F1 Score')
    plt.plot(epochs, val_f1_scores, color='cyan', label='Validation F1 Score')
    plt.ylim(0, 1.1)
    plt.xlim(0, len(epochs)-1) 
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Plot Accuracy Score
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_acc_scores, color='green', label='Training Accuracy Score')
    plt.plot(epochs, val_acc_scores, color='olive', label='Validation Accuracy Score')
    plt.ylim(0, 1.1) 
    plt.xlim(0, len(epochs)-1) 
    plt.title('Training and Validation Accuracy Score')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Score')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Plot Precision Score
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_prec_scores, color='black', label='Training Precision Score')
    plt.plot(epochs, val_prec_scores, color='grey', label='Validation Precision Score')
    plt.ylim(0, 1.1)
    plt.xlim(0, len(epochs)-1)  
    plt.title('Training and Validation Precision Score')
    plt.xlabel('Epochs')
    plt.ylabel('Precision Score')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Plot Recall Score
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_rec_scores, color='brown', label='Training Recall Score')
    plt.plot(epochs, val_rec_scores, color='pink', label='Validation Recall Score')
    plt.ylim(0, 1.1)
    plt.xlim(0, len(epochs)-1) 
    plt.title('Training and Validation Recall Score')
    plt.xlabel('Epochs')
    plt.ylabel('Recall Score')
    plt.legend(loc='lower right')
    plt.grid(True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, 'NCT_classifier_curves.png')
        plt.savefig(plot_path)
        print(f'Plot saved to {plot_path}')
    
    plt.show()

# Main script
seed_all(42)
directory = '/mnt/volume/mathias/outputs/patch_uni_output/h5_files'
batch_size = 512
num_epochs = 20
lr = 0.00001
save_plot_dir = '/mnt/volume/sabrina/NCT_classifier_plots'
model_save_path = '/mnt/volume/sabrina/pretrained_models/NCT_classifier_model.pth'
log_file_path = '/mnt/volume/sabrina/NCT_classifier_plots/NCT_classifier_log.txt'

# Load dataset
dataset = HDF5Dataset(directory)

# Perform stratified train-test split
train_indices, test_indices = train_test_split(
    np.arange(len(dataset)), test_size=0.15, stratify=dataset.labels, random_state=42
)

train_indices, val_indices = train_test_split(
    train_indices, test_size=0.18, stratify=dataset.labels[train_indices], random_state=42
)

# Create the subsets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define model
input_size = dataset.features.shape[1]
num_classes = len(set(dataset.labels))
model = SimpleNN(input_size, num_classes)

# Define loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# Train model
train_loss, val_loss, train_f1_scores, val_f1_scores, train_acc_scores, val_acc_scores, train_prec_scores, val_prec_scores, train_rec_scores, val_rec_scores = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs)

# Test model
test_f1, test_acc, test_prec, test_rec = test_model(model, test_loader)

# Save the model
save_model(model, model_save_path)

# Log parameters and metrics
params = {
    "directory": directory,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "learning_rate": lr,
    "model_save_path": model_save_path,
    "log_file_path": log_file_path,
    "input_size": input_size,
    "num_classes": num_classes
}

metrics = {
    "test_f1": test_f1,
    "test_acc": test_acc,
    "test_prec": test_prec,
    "test_rec": test_rec
}

log_params_and_metrics(log_file_path, params, metrics)

# Plot and save loss and F1 score curves
plot_metrics(train_loss, val_loss, train_f1_scores, val_f1_scores, train_acc_scores, val_acc_scores, train_prec_scores, val_prec_scores, train_rec_scores, val_rec_scores, save_dir=save_plot_dir)
