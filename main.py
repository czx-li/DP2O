import torch
import matplotlib.pyplot as plt
import Utils
from transformers import RobertaTokenizer
from Net import BertPromptTune
from Agent import REINFORCE
from Env import env
from Reward import R_Function
from Train import train_on_policy_agent, test_on_policy_agent
import Norm
import random
import numpy as np

seed = 0
model_name = 'roberta-large'
state_dim = 1024
action_dim = 15
hidden_dim = 600
num_episodes = 200
state_norm = Norm.Normalization(shape=state_dim)
reward_norm = Norm.Normalization(shape=1)
vocab_size = 50265
positive_words = ['positive']
negative_words = ['negative']
learning_rate = 1e-3
entropy_coe = 0.059
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 32
top_k = 10
use_similar = False
use_orthogonal_init = True
use_norm_state = True
use_norm_reward = True
Eps = True
glue = True
use_decision = False
way = 'RTE'
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
tokenizer = RobertaTokenizer.from_pretrained(model_name)
mask_token_id = tokenizer(tokenizer.mask_token)['input_ids'][1]
positive_token_ids = tokenizer(" ".join(positive_words))['input_ids'][1:-1]
negative_token_ids = tokenizer(" ".join(negative_words))['input_ids'][1:-1]

agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, entropy_coe, device, use_orthogonal_init, num_episodes, Eps)
train_dataloader, test_dataloader, train_text, test_text, total_steps, state_norm, sim_state = env(batch_size, num_episodes, way, state_norm, model_name, state_dim, use_similar, glue)
bert_test = BertPromptTune(vocab_size, mask_token_id, positive_token_ids, negative_token_ids, model_name, device)
Reward = R_Function(bert_test, tokenizer, reward_norm, use_norm_reward, device, way, sim_state)
return_list, acc_list = train_on_policy_agent(train_dataloader, train_text, test_text, agent, num_episodes, Reward, test_dataloader, use_norm_state, state_norm, device, use_decision, use_similar, top_k)
# return_list, acc_list = test_on_policy_agent(test_text, agent, Reward, test_dataloader, use_norm_state,state_norm,device,use_decision, use_similar, top_k)


episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PG_Prompt on {}'.format('SST-2'))
plt.show()

mv_return = Utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PG_Prompt on {}'.format('SST-2'))
plt.show()

