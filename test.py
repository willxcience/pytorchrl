import torchrl
import torchrl.network as network
from torchrl.a2c import A2C
from torchrl.a2c import ActorCritic

# later incorporate this into torchrl
from torchrl.utils import make_atari_env
from torchrl.utils import weights_init
from torchrl.parallel import SubprocVecEnv
from torchrl.parallel import VecFrameStack

import time
import torch
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter


params = {}
params["num_in_channels"] = 4
params["num_latent_nodes"] = 512

# control the speed of training
params["num_workers"] = 10
params["rollout_len"] = 5
params["use_cuda"] = True

# log interval steps
params["log_interval"] = 2000
params["total_steps"] = 1e7

# hyperparaters for traing
params["discount_gamma"] = 0.9
params["entropy_coef"] = 0.01
params["value_coef"] = 0.5
params["use_gae"] = False
params["gae_tau"] = 0.95
params["max_grad_norm"] = 0.5
params["learning_rate"] = 7e-4
params["RMS_alpha"] = 0.99
params["RMS_eps"] = 1e-5
params["resume"] = False

env = VecFrameStack(make_atari_env("BreakoutNoFrameskip-v4", params["num_workers"]), 4)

params["num_actions"] = env.action_space.n

# define a model and its optimizer
model = ActorCritic(params)
model.apply(weights_init)

# RMSprop better for RL
optimizer = optim.RMSprop(model.parameters(),
                           params["learning_rate"],
                           alpha=params["RMS_alpha"],
                           eps=params["RMS_eps"]
                          )

agent = A2C(env, model, optimizer, params)

writer = SummaryWriter()

batch_size = int(agent.num_workers * agent.rollout_len)
num_iter = int(params["total_steps"] // batch_size)

model.cuda()
model.train()
t0 = time.time()

log_interval = params["log_interval"]
log_iter = int(log_interval // batch_size)

for e in range(num_iter):
    agent.train_step()
    
    if e % log_iter == 0:
        writer.add_scalar('policy loss', agent.policy_loss, agent.num_steps)
        writer.add_scalar('entropy loss', agent.entropy_loss, agent.num_steps)
        writer.add_scalar('value loss', agent.value_loss, agent.num_steps)
        
        if len(agent.episode_rewards) > 0:
            writer.add_scalar('Mean Reward', np.mean(agent.episode_rewards), agent.num_steps)
            writer.add_scalar('Max Reward', np.max(agent.episode_rewards), agent.num_steps)
            
            print('Returns %.2f/%.2f/%.2f/%.2f (mean/median/min/max)  %.2f steps/s' % (
                np.mean(agent.episode_rewards),
                np.median(agent.episode_rewards),
                np.min(agent.episode_rewards),
                np.max(agent.episode_rewards),
                batch_size * log_iter / (time.time() - t0)
            ))
            
            agent.episode_rewards.clear()

            t0 = time.time()
    
    # save the model every 1000000 steps
    if e % (1e6 // batch_size) == 0:
        checkpoint = {
            'num_steps': agent.num_steps,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()
            }
        torch.save(checkpoint, "Breakout/saved_model" + str(agent.num_steps) + ".pth")
