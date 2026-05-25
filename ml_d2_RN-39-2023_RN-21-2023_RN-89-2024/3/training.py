import torch as th
import numpy as np

# 1. Učitavanje podataka
x = np.load('observations.npy', allow_pickle=True)
y = np.load('actions.npy', allow_pickle=True)

x = x.reshape(x.shape[0], -1)
X_tensor = th.tensor(x, dtype=th.float32)
Y_tensor = th.tensor(y, dtype=th.long)

device = "cuda" if th.cuda.is_available() else ("mps" if th.backends.mps.is_available() else "cpu")
X_tensor = X_tensor.to(device)
Y_tensor = Y_tensor.to(device)

# 2. Ručna inicijalizacija parametara
W1 = th.randn(108, 64, device=device) * np.sqrt(2.0 / 108)
b1 = th.zeros(64, device=device)

W2 = th.randn(64, 32, device=device) * np.sqrt(2.0 / 64)
b2 = th.zeros(32, device=device)

W3 = th.randn(32, 7, device=device) * np.sqrt(2.0 / 32)
b3 = th.zeros(7, device=device)

params = [W1, b1, W2, b2, W3, b3]
for p in params:
    p.requires_grad_(True)

# INICIJALIZACIJA MOMENTUMA (brzina za svaki parametar)
velocities = [th.zeros_like(p) for p in params]

# 3. Mreža i funkcija gubitka
def relu(z):
    return th.clamp(z, min=0)

def forward(x):
    z1 = x @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    return a2 @ W3 + b3

def cross_entropy_loss(logits, targets):
    max_logits = th.max(logits, dim=1, keepdim=True)[0]
    log_sum_exp = max_logits + th.log(th.sum(th.exp(logits - max_logits), dim=1, keepdim=True))
    loss = log_sum_exp.squeeze() - logits[range(logits.shape[0]), targets]
    return loss.mean()

# 4. Petlja za trening sa MOMENTUMOM
epochs = 500 
batch_size = 32
lr = 0.005    # learning rate
beta = 0.9    # Momentum koeficijent (koliko "pamtimo" prethodni smer)

print("Započinjem ručni trening sa Momentumom...")

for epoch in range(epochs):
    permutation = th.randperm(X_tensor.size()[0])
    total_loss = 0
    num_batches = 0
    
    for i in range(0, X_tensor.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_tensor[indices], Y_tensor[indices]
        
        outputs = forward(batch_x)
        loss = cross_entropy_loss(outputs, batch_y)
        
        loss.backward()
        
        # Ručni update koristeći SGD sa Momentumom
        with th.no_grad():
            for p, v in zip(params, velocities):
                # v = beta * v + grad
                v.copy_(beta * v + p.grad)
                # p = p - lr * v
                p -= lr * v
                p.grad.zero_()
                
        total_loss += loss.item()
        num_batches += 1
        
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/num_batches:.4f}")

# 5. Čuvanje parametara
weights_dict = {
    'W1': W1, 'b1': b1,
    'W2': W2, 'b2': b2,
    'W3': W3, 'b3': b3
}
th.save(weights_dict, 'minigrid_bc_model.pth')
print("Model uspešno sačuvan!")