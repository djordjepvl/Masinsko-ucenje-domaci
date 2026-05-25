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


import torch.nn as nn

# Istovetna arhitektura kao u treningu
class BehavioralCloningModel(nn.Module):
    def __init__(self):
        super(BehavioralCloningModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(108, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7)
        )
        
    def forward(self, x):
        return self.net(x)

# Inicijalizacija modela
model = BehavioralCloningModel()

model.load_state_dict(th.load('minigrid_bc_model.pth', map_location=device))
model.to(device)

# Prebacivanje u float64 jer test.py to zahteva za ulazne podatke
model.to(th.float64)

# Režim evaluacije
model.eval()


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
