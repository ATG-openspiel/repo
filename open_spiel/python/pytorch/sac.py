import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import copy

INVALID_ACTION_PENALTY = -1e6

class CategoricalMasked(Categorical):
    """A masked categorical."""
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], mask_value=None):
        logits = torch.where(masks.bool(), logits, mask_value)
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)

# def get_list_dimension(lst):
#     if isinstance(lst, list):
#         if lst:
#             return 1 + max(get_list_dimension(item) for item in lst)
#         else:
#             return 1
#     else:
#         return 0


def legal_actions_to_mask(legal_actions_list, num_actions):
    """Converts a list of legal actions to a mask.

    The mask has size num actions with a 1 in a legal positions.

    Args:
    legal_actions_list: the list of legal actions
    num_actions: number of actions (width of mask)

    Returns:
    legal actions mask.
    """
    if isinstance(legal_actions_list, list):
        legal_actions_mask = torch.zeros((len(legal_actions_list), num_actions), dtype=torch.bool)
        for i, legal_actions in enumerate(legal_actions_list):
            legal_actions.int()
            legal_actions_mask[i, legal_actions.tolist()] = 1

    else:
        legal_actions_mask = torch.zeros((num_actions),
                                    dtype=torch.bool)
        for legal_action in legal_actions_list:
            legal_actions_mask[int(legal_action)] = 1

    return legal_actions_mask



def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = self.fc3(x)
        return action_probs
    
    def evaluate(self, state, epsilon=1e-6, legal_actions_mask=None):
        action_probs = self.forward(state)
        if legal_actions_mask is None:
            legal_actions_mask = torch.ones_like(action_probs).bool()   

        dist = CategoricalMasked(
        logits=action_probs, masks=legal_actions_mask, mask_value=INVALID_ACTION_PENALTY)
        action = dist.sample().to(state.device)
        action_probs = dist.probs
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities        
    
    def get_action(self, state, legal_actions_mask=None):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        action_probs = self.forward(state)
        if legal_actions_mask is None:
            legal_actions_mask = torch.ones_like(action_probs).bool()

        dist = CategoricalMasked(
        logits=action_probs, masks=legal_actions_mask, mask_value=INVALID_ACTION_PENALTY)
        action = dist.sample().to(state.device)
        action_probs = dist.probs
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities
    
    def get_det_action(self, state, legal_actions_mask=None):
        action_probs = self.forward(state)
        if legal_actions_mask is None:
            legal_actions_mask = torch.ones_like(action_probs).bool()
            
        dist = CategoricalMasked(
        logits=action_probs, masks=legal_actions_mask, mask_value=INVALID_ACTION_PENALTY)
        # dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        return action.detach().cpu()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, seed=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        # la is equal to legal_action
        self.experience = namedtuple("Experience", field_names=["state", "la", "action", "reward", "next_state", "la_next", "done"])
        # self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, la, action, reward, next_state, la_next, done):
        """Add a new experience to memory."""
        e = self.experience(state, la, action, reward, next_state, la_next, done)
        # e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        las = [e.la for e in experiences if e is not None]
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        la_nexts = [e.la_next for e in experiences if e is not None]
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, las, actions, rewards, next_states, la_nexts, dones)
        # return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class SAC(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                        state_size,
                        action_size,
                        device,
                        player_id=0
                ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(SAC, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.device = device
        self.player_id = player_id
        
        self.gamma = 0.99
        self.tau = 1e-2
        hidden_size = 256
        learning_rate_alpha = 5e-5
        learning_rate_policy = 5e-5
        learning_rate_critic = 1e-6
        self.clip_grad_param = 1

        self.target_entropy = -action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate_alpha) 
                
        # Actor Network 

        self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate_policy)     
        
        # Critic Network (w/ Target Network)

        self.critic1 = Critic(state_size, action_size, hidden_size, 2).to(device)
        self.critic2 = Critic(state_size, action_size, hidden_size, 1).to(device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate_critic) 

    
    def get_action(self, state, legal_actions_mask=None):
        """Returns actions for given state as per current policy."""
        state = state.float().to(self.device)
        
        with torch.no_grad():
            action = self.actor_local.get_det_action(state, legal_actions_mask)
        return action.numpy()

    def calc_policy_loss(self, states, alpha, legal_actions_masks):
        _, action_probs, log_pis = self.actor_local.evaluate(states, legal_actions_masks)

        q1 = self.critic1(states)   
        q2 = self.critic2(states)
        min_Q = torch.min(q1,q2)
        actor_loss = (action_probs * (alpha * log_pis - min_Q )).sum(1).mean()
        log_action_pi = torch.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi
    
    def learn(self, step, experiences, gamma, d=1):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, las, actions, rewards, next_states, la_nexts, dones = experiences

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        legal_action_masks = legal_actions_to_mask(las, self.action_size)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha.to(self.device), legal_action_masks)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            legal_action_masks = legal_actions_to_mask(la_nexts, self.action_size)
            _, action_probs, log_pis = self.actor_local.evaluate(next_states, legal_action_masks)
            Q_target1_next = self.critic1_target(next_states)
            Q_target2_next = self.critic2_target(next_states)
            Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1)) 

        # Compute critic loss
        q1 = self.critic1(states).gather(1, actions.long())
        q2 = self.critic2(states).gather(1, actions.long())
        
        critic1_loss = 0.5 * F.mse_loss(q1, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2, Q_targets)

        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), current_alpha

    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0-self.tau)*target_param.data)


def save(args, save_name, model, ep=None):
    import os
    save_dir = './sac_trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")


# env = make_single_env("kuhn_poker", 0)()
def collect_random(env, dataset, num_samples=200, player_id=1):
    state = env.reset()
    num = 0
    while num < num_samples:
        #玩家i行动前其他玩家随机采取动作
        while state.observations["current_player"] != player_id:
            if state.last():
                state = env.reset()
                continue
            
            current_player = state.observations["current_player"]
            action = [random.choice(state.observations["legal_actions"][current_player])]
            state = env.step(action)

        #玩家i行动
        state_tensor = torch.Tensor(state.observations["info_state"][player_id])
        la =  torch.Tensor(state.observations["legal_actions"][player_id])
        action = [random.choice(state.observations["legal_actions"][player_id])]
        state = env.step(action)

        #玩家i行动后到下次玩家i行动前或玩家i行动后到游戏结束
        while state.observations["current_player"] != player_id:
            if state.last():
                break
            current_player = state.observations["current_player"]
            action = [random.choice(state.observations["legal_actions"][current_player])]
            state = env.step(action)
            
        done = state.last()
        next_state_tensor = torch.Tensor(state.observations["info_state"][player_id])
        la_next =  torch.Tensor(state.observations["legal_actions"][player_id])
        dataset.add(state_tensor, la, action, state.rewards[player_id], next_state_tensor, la_next, done)
        num += 1

        if done:
            state = env.reset()
