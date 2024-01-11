from tqdm import tqdm
import numpy as np
import torch
import random

def train_on_policy_agent(env, train_text, test_text, agent, num_episodes, Reward, test_dataloader, use_norm_state,state_norm,device,use_decision, use_similar, top_k):
    return_list = []
    acc_list = []
    best_acc = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'rewards': []}
                for step, batch in enumerate(env):
                    states = batch[0].cpu()
                    if use_norm_state == True:
                        states = state_norm(states, update=False)
                        states = states.to(torch.float32)
                        states = states.to(device)
                    else:
                        states = states.to(device)
                    labels = batch[1].to(device)
                    ids = batch[2].to(device)
                    action = agent.take_action(states)
                    reward, r = Reward.reward_compute(action, ids, train_text, labels)
                    transition_dict['states'].append(states)
                    transition_dict['actions'].append(action)
                    transition_dict['rewards'].append(reward)
                    r = r.cpu()
                    episode_return += r
                return_list.append(episode_return)
                agent.update(transition_dict, i*int(num_episodes / 10)+i_episode)

                if (i_episode + 1) % 1 == 0:
                # if (i * int(num_episodes / 10) + i_episode) >= 199:
                    total_eval_accuracy = 0
                    total_eval_loss = 0
                    for step, batch in enumerate(test_dataloader):
                        #print(step)
                        states = batch[0].cpu()
                        if use_norm_state == True:
                            states = state_norm(states, update=False)
                            states = states.to(torch.float32)
                            states = states.to(device)
                        else:
                            states = states.to(device)
                        labels = batch[1].to(device)
                        ids = batch[2].to(device)
                        action, probs = agent.evaluate(states)
                        if use_similar:
                            logits, loss = Reward.acc_compute_use_similar(states, action, ids, test_text, labels)
                            total_eval_loss += loss.item()
                        elif use_decision:
                            logits = Reward.acc_compute_use_decision(probs, ids, test_text, labels, top_k)
                        else:
                            logits, loss = Reward.acc_compute(action, ids, test_text, labels)
                            total_eval_loss += loss.item()
                        logits = logits.detach().cpu().numpy()
                        label_ids = labels.to('cpu').numpy()
                        total_eval_accuracy += flat_accuracy(logits, label_ids)
                    acc_list.append(total_eval_accuracy/len(test_dataloader))
                    if best_acc <= (total_eval_accuracy/len(test_dataloader)):
                        best_acc = total_eval_accuracy/len(test_dataloader)
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-1:]),
                                      'test_acc': '%.3f' % (total_eval_accuracy/len(test_dataloader)),
                                      # 'test_loss': '%.3f' % (total_eval_loss / len(validation_dataloader)),
                                      'best_acc': '%.3f' % best_acc})
                pbar.update(1)
    agent.save()
    return return_list, acc_list

def test_on_policy_agent(test_text, agent, Reward, test_dataloader, use_norm_state,state_norm,device,use_decision, use_similar, top_k):
    return_list = []
    acc_list = []
    best_acc = 0
    for i in range(10):
        total_eval_accuracy = 0
        total_eval_loss = 0
        for step, batch in enumerate(test_dataloader):
            states = batch[0].cpu()
            if use_norm_state == True:
                states = state_norm(states, update=False)
                states = states.to(torch.float32)
                states = states.to(device)
            else:
                states = states.to(device)
            labels = batch[1].to(device)
            ids = batch[2].to(device)
            action, probs = agent.evaluate(states)
            if use_similar:
                logits, loss = Reward.acc_compute_use_similar(states, action, ids, test_text, labels)
                total_eval_loss += loss.item()
            elif use_decision:
                logits = Reward.acc_compute_use_decision(probs, ids, test_text, labels, top_k)
            else:
                logits, loss = Reward.acc_compute(action, ids, test_text, labels)
                total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        acc_list.append(total_eval_accuracy / len(test_dataloader))
        if best_acc <= (total_eval_accuracy / len(test_dataloader)):
            best_acc = total_eval_accuracy / len(test_dataloader)
        print(best_acc)
    return return_list, acc_list

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)