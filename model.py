# =====================================
# 1. IMPORT LIBRARIES
# =====================================
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import collections

# =====================================
# 2. LOAD DATASET (LOCAL FOLDER)
# =====================================
dataset_path ="D:/KaitexyAI-new/backend/landmark_dataset"   # <-- dataset folder inside your VS Code project

data = []
labels = []
label_map = {}
label_index = 0

print("Loading dataset...")

for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)

    if os.path.isdir(label_path):
        print(f"Processing: {label}")
        label_map[label] = label_index

        for file in os.listdir(label_path):
            if file.endswith(".npy"):
                sample = np.load(os.path.join(label_path, file))
                sample = sample.flatten()  # works for (63,) or (30,63)

                data.append(sample)
                labels.append(label_index)

        label_index += 1

X = np.array(data)
y = np.array(labels)

print("\n Dataset Loaded")
print("Shape:", X.shape)
print("Classes:", label_map)

# Check class balance
print("\nClass distribution:")
print(collections.Counter(y))

# =====================================
# 3. TRAIN-TEST SPLIT
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================
# 4. CONVERT TO PYTORCH
# =====================================
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# =====================================
# 5. DATALOADER
# =====================================
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# =====================================
# 6. DEVICE (GPU)
# =====================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)

# =====================================
# 7. MODEL
# =====================================
class SignModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SignModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = SignModel(X.shape[1], len(label_map)).to(device)

# =====================================
# 8. LOSS + OPTIMIZER
# =====================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# =====================================
# 9. TRAINING LOOP
# =====================================
epochs = 100
print("\n Training started...")

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy per batch
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

print("\n Training finished!")

# =====================================
# 10. TEST ACCURACY
# =====================================
model.eval()
X_test = X_test.to(device)
y_test = y_test.to(device)

with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)

print(f"\n Final Test Accuracy: {accuracy*100:.2f}%")

# =====================================
# 11. SAVE MODEL LOCALLY
# =====================================
model_path = "./sign_model.pt"   # saved in your VS Code project folder
torch.save(model.state_dict(), model_path)

print("\n Model saved at:", model_path)
