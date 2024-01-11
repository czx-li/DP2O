import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim,use_orthogonal_init):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        if use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, entropy_coe, device,use_orthogonal_init,num_episodes,Eps):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim, use_orthogonal_init).to(device)
        total = sum([param.nelement() for param in self.policy_net.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
        # self.policy_net = torch.load('policy_net.pkl')
        if Eps==True:
            self.optimizer = torch.optim.AdamW(self.policy_net.parameters(),lr=learning_rate, eps=1e-5)
        else:
            self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=learning_rate)
        self.entropy_coef = entropy_coe
        self.learning_rate = learning_rate
        self.max_train_steps = num_episodes
        self.device = device

    def take_action(self, state):
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action

    def evaluate(self, state):
        probs = self.policy_net(state)
        a = torch.max(probs, dim=1).indices
        return a, probs


    def update(self, transition_dict, step):
        reward_list = transition_dict['rewards'][0]
        state_list = transition_dict['states'][0]
        action_list = transition_dict['actions'][0]
        loss = 0
        # reward_list = ((reward_list - torch.mean(reward_list)) / (torch.std(reward_list) + 1e-5))
        self.optimizer.zero_grad()
        for i in range(len(reward_list)):
            reward = reward_list[i]
            state = state_list[i]
            action = action_list[i]
            log_prob = torch.log(self.policy_net(state.unsqueeze(0))[0][int(action)])
            dist_now = Categorical(probs=self.policy_net(state.unsqueeze(0))[0])
            dist_entropy = dist_now.entropy().view(-1, 1)
            loss = loss - log_prob * reward - self.entropy_coef * dist_entropy
            loss.backward(retain_graph=True)
        self.optimizer.step()
        self.lr_decay(step)

    def save(self):
        torch.save(self.policy_net, 'policy_net.pkl')

    def lr_decay(self,step):
        lr_a_now = self.learning_rate * (1 - step / self.max_train_steps)
        for p in self.optimizer.param_groups:
            p['lr'] = lr_a_now


