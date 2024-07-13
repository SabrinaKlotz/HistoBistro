import argparse
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt

# Dataset class to fetch the UNI features and class labels from the HDF5 files
class NCTDataset(Dataset):
    def __init__(self, h5_file_path, transform=None):
        self.h5_file = h5_file_path
        self.transform = transform
        self.features, self.labels = self.load_data()
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    # read the hdf5 files and create dictionaries for features and labels
    def load_data(self):
        with h5py.File(self.h5_file, 'r') as f:
            features = []
            labels = []
            for key in f.keys():
                if 'features' in key:
                    features.append(f[key][:])
                elif 'labels' in key:
                    labels.append(f[key][:])
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        return features, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            feature = self.transform(feature)

        return feature, label

# Simple neural network model with ReLU
class NCTClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NCTClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# training model and using F1 score to assess performance 
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device='cpu'):
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
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc.item())
                train_f1.append(epoch_f1)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc.item())
                val_f1.append(epoch_f1)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')

    # Plotting training and validation curves
    epochs_range = range(num_epochs)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Accuracy')

    plt.show()

    return model

# evaluate model in test dataset
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Test Accuracy: {accuracy:.4f} F1 Score: {f1:.4f}')

def main(args):
    dataset = NCTDataset(h5_file_path=args.h5_file)

    # define random dataset splits 0.7, 0.15, 0.15 for now
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    }

    input_dim = dataset.features.shape[1]

    # should be nine classes
    num_classes = len(torch.unique(dataset.labels))

    # hyperparameters we can tune
    model = NCTClassifier(input_dim=input_dim, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # train model
    trained_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=args.epochs, device=args.device)

    # save trained model
    torch.save(trained_model.state_dict(), args.model_save_path)
    print(f'Model saved to {args.model_save_path}')

    evaluate_model(trained_model, dataloaders['test'], device=args.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train NCT Classifier')
    parser.add_argument('--h5_file', type=str, required=True, help='Path to the HDF5 file with features and labels')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs for training')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training')
    parser.add_argument('--model_save_path', type=str, default='nct_classifier.pth', help='Path to save the trained model')
    args = parser.parse_args()

    main(args)
