import torch as th
import numpy as np
import gymnasium as gym

from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

if th.cuda.is_available():
    device = "cuda"
elif th.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

LOCAL = True

#################### TO-DO ####################


# Učitavanje sačuvanog rečnika sa matricama težina
weights = th.load('minigrid_bc_model.pth', map_location=device)

# Izvlačenje pojedinačnih matrica i prebacivanje u float64 (jer test.py to zahteva)
W1 = weights['W1'].to(th.float64)
b1 = weights['b1'].to(th.float64)
W2 = weights['W2'].to(th.float64)
b2 = weights['b2'].to(th.float64)
W3 = weights['W3'].to(th.float64)
b3 = weights['b3'].to(th.float64)

# Definišemo funkciju koja će zameniti uobičajeni model(obs) poziv
def manual_model(obs):
    # Prolaz kroz prvi sloj + ReLU
    z1 = obs @ W1 + b1
    a1 = th.clamp(z1, min=0)
    
    # Prolaz kroz drugi sloj + ReLU
    z2 = a1 @ W2 + b2
    a2 = th.clamp(z2, min=0)
    
    # Krajnji izlaz (Logits)
    logits = a2 @ W3 + b3
    return logits

# Dodela našeg ručnog modela promenljivoj koju test.py koristi
model = manual_model


###############################################


env = gym.make(
    "MiniGrid-Empty-Random-6x6-v0",
    render_mode="human" if LOCAL else "rgb_array",
    highlight=False,
    screen_size=640
)

env = FullyObsWrapper(env)
env = ImgObsWrapper(env)

rewards = []

for episode in range(10):
    if model is None:
        break

    obs, _ = env.reset()
    step = 0
    terminated = False
    truncated = False

    while not terminated and not truncated and step < 30:
        if LOCAL:
            env.render()
        with th.no_grad():
            obs = th.tensor(obs, dtype=th.float64, device=device).reshape(-1, 108)  #! promeniti po potrebi
            action = model(obs)
        obs, reward, terminated, truncated, _ = env.step(th.argmax(action).item())
        step += 1
    
    print(f"{episode=} {reward=}")
    rewards.append(reward)

env.close()
print(f"mean reward: {np.mean(rewards)}")
