import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

x = np.load('observations.npy', allow_pickle=True)
y = np.load('actions.npy', allow_pickle=True)

x = x.reshape(x.shape[0], -1)  # Pretvara (N, 6, 6, 3) u (N, 108)

X_tensor = th.tensor(x, dtype=th.float32)
Y_tensor = th.tensor(y, dtype=th.long)

dataset = TensorDataset(X_tensor, Y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 108 ulaza -> Skriveni slojevi -> 7 mogućih akcija u MiniGrid-u
class BehavioralCloningModel(nn.Module):
    def __init__(self):
        super(BehavioralCloningModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(108, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7) # MiniGrid obično ima 7 diskretnih akcija
        )
        
    def forward(self, x):
        return self.net(x)

device = "cuda" if th.cuda.is_available() else ("mps" if th.backends.mps.is_available() else "cpu")
model = BehavioralCloningModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop
epochs = 1000
print("Započinjem trening...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass i optimizacija
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# čuvanje istreniranog modela
th.save(model.state_dict(), 'minigrid_bc_model.pth')
print("Model je uspešno sačuvan kao 'minigrid_bc_model.pth'!")