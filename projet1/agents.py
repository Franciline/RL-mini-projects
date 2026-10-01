import torch 
import torch.nn as nn 
import torch.nn.functional as F
import copy 
from rl_mind.core import Action,Actor
from rl_mind.nn import soft_update

def mlp(sizes,layer_norm=False):
    """Build a simple MLP with sizes: [input, hidden layers, output]."""
    layers = []
    # loop over the layers except the last one so len(sizes)-2
    for i in range(len(sizes)-2): 
        layers.append(nn.Linear(sizes[i],sizes[i+1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i+1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(sizes[-2],sizes[-1])) # the last layer
    return nn.Sequential(*layers)
            
# the actor: a deterministic policy network. 
# collector calls it during training and exploration.
# evaluator calls it during eval and when it needs a clean action 

# Actor returns an Action 
class DetActor(Actor[Action]):
    """sigma: standard deviation of exploration noise. Default hidden dim = 2.""" 
    def __init__(self, obs_dim, act_dim, hidden=(256,256), layer_norm=False, sigma=0.1):
        super().__init__()
        self.net = mlp([obs_dim,*hidden,act_dim],layer_norm)
        self.sigma = sigma

    def pi(self, obs):
        """Differentiable policy. We use tanh to have output into [-1,1] which is valid range of LunarLander's continuous actions (main engine, lateral engine)."""
        return torch.tanh(self.net(obs)) # tensor dim 2 

    def forward(self,obs):
        """Noisy training action. No grad because action goes into replay buffer, causing memory leak if not."""
        with torch.no_grad():
            a = self.pi(obs) # [-1,1]x[-1,1]
            a = (a + self.sigma * torch.rand_like(a)).clamp(-1,1)
        return Action(value=a)

    def act(self,obs):
        """Deterministric eval action. Returns plain tensor."""
        with torch.no_grad():
            return self.pi(obs)

# the critic: network that takes a state and action. returns batch of Q(s,a)
# actor's update backpropagates through it
# we compare this output to monte carlo 

class Critic(nn.Module):
    def __init__(self, obs_dim,act_dim,hidden=(256,256),layer_norm=False):
        super().__init__()
        # input is obs_dim + act_dim bc action is continuous. output is 1 = Q val
        self.net = mlp([obs_dim + act_dim, *hidden, 1], layer_norm)

    # with grad because we want it to train
    def forward(self, obs, act):
        cat = torch.cat([obs, act], dim=-1) # joins obs and action along the feature dimension (not B)
        out = self.net(cat) # [B,1]
        return out.squeeze(-1) # [B] to match reward shape

# DPPG: needs actor, critic, target copies, optimizers
# keeps two extra networks: target actor and target critic. copies used to compute learning target.
# they are updated by soft update no gradient

class DDPG:
    """gamma: discount factor. tao: how fast targets follow main networks (default 0.5% per step)."""
    def __init__(self, obs_dim, act_dim, layer_norm=False, hidden=(256,256),lr=3e-4, gamma=0.99, tau=0.005, sigma=0.1):
        # online networks
        self.actor = DetActor(obs_dim, act_dim, hidden, layer_norm, sigma)
        self.critic = Critic(obs_dim, act_dim, hidden, layer_norm)    

        # target networks, deeptarget so that independent params 
        self.actor_targ = copy.deepcopy(self.actor)
        self.critic_targ = copy.deepcopy(self.critic)

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma, self.tau = gamma, tau

    def update(self,batch):
        """one gradient step on a batch from replay buffer. batch is a Transitions object"""
        # batch unpack
        obs, act = batch.obs, batch.action.value
        r, next_obs, term = batch.reward, batch.next_obs, batch.terminated

        # update critic, Bellman target
        with torch.no_grad():
            # evaluate the picked next actions with target actor
            next_q = self.critic_targ(next_obs,self.actor_targ.pi(next_obs))
            # a real terminal state does not bootstrap
            target = r + self.gamma * (~term).float() * next_q

        # critic returns Q(s,a)
        critic_loss = F.mse_loss(self.critic(obs,act),target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # actor's own action pi(obs), gradient from Q back through action into actor's weights
        # minus to turn "maximize Q" into a loss to minimize 
        actor_loss = -self.critic(obs, self.actor.pi(obs)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        soft_update(self.critic, self.critic_targ, self.tau)
        soft_update(self.actor, self.actor_targ, self.tau)
        