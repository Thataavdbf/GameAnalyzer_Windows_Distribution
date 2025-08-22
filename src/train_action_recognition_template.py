"""
Template: Action Recognition Training Script (Best Practices)
- Uses PyTorch
- Includes data loading, model setup, training loop, validation, and metrics
- Adapt for your dataset and model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# TODO: Replace with your dataset class
class MyActionDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        # Load and preprocess your data here
        pass
    def __len__(self):
        # Return number of samples
        return 0
    def __getitem__(self, idx):
        # Return (video_tensor, label)
        return None, None

# TODO: Replace with your model architecture
class MyActionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers
        pass
    def forward(self, x):
        # Forward pass
        return x

def train_model():
    # Hyperparameters
    batch_size = 8
    epochs = 10
    lr = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_dataset = MyActionDataset('data/train', split='train')
    val_dataset = MyActionDataset('data/val', split='val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = MyActionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} complete.")
        validate(model, val_loader, device)
    torch.save(model.state_dict(), 'action_model.pt')
    print("Training complete. Model saved.")

def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total if total > 0 else 0
    print(f"Validation accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    train_model()
