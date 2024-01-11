import torch
import random
from random import randint

class R_Function():
    def __init__(self, bert, tokenizer, reward_norm, use_norm_reward, device, way, sim_state):
        self.bert = bert
        self.tokenizer = tokenizer
        self.use_norm_reward = use_norm_reward
        self.reward_norm = reward_norm
        self.device = device
        self.way = way
        self.sim_state = sim_state

    def reward_compute(self, action, ids, text, labels):
        reward = []
        r = 0
        Local = 0
        Reward = self.reward_norm(0)
        if self.way == 'SST-2':
            input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_SST(action, ids, text)
        elif self.way == 'Yelp':
            input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_Yelp(action, ids, text)
        elif self.way == 'CR':
            input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_CR(action, ids, text)
        elif self.way == 'MR':
            input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_MR(action, ids, text)
        elif self.way == 'RTE':
            input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_RTE(action, ids, text)
        elif self.way == 'QNLI':
            input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_QNLI(action, ids, text)
        elif self.way == 'MRPC':
            input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_MRPC(action, ids, text)
        else:
            input_ids_new, attention_masks_new = 0, 0
            print('Error!!!')
        input_ids_new = input_ids_new.to(self.device)
        attention_masks_new = attention_masks_new.to(self.device)
        with torch.no_grad():
            prob, _ = self.bert(input_ids_new, attention_masks_new, labels)
        for i in range(len(action)):
            # R = torch.log(prob[i, labels[i]]) - torch.log(prob[i, 1 if labels[i] == 0 else 0])
            R = 10*prob[i, labels[i]] - 10*prob[i, 1 if labels[i] == 0 else 0] - 6.5*prob[i, labels[i]] * torch.log2(prob[i, labels[i]]) - 6.5*prob[i, 1 if labels[i] == 0 else 0] * torch.log2(prob[i, 1 if labels[i] == 0 else 0])
            if self.use_norm_reward:
                R = R.cpu()
                R = self.reward_norm(R)
                R = R.to(self.device)
            reward.append(R)
            r += R
        reward = torch.stack(reward)
        return reward, r

    def acc_compute(self, action, ids, text, labels):
        if self.way == 'SST-2':
            input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_SST(action, ids, text)
        elif self.way == 'Yelp':
            input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_Yelp(action, ids, text)
        elif self.way == 'CR':
            input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_CR(action, ids, text)
        elif self.way == 'MR':
            input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_MR(action, ids, text)
        elif self.way == 'RTE':
            input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_RTE(action, ids, text)
        elif self.way == 'QNLI':
            input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_QNLI(action, ids, text)
        elif self.way == 'MRPC':
            input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_MRPC(action, ids, text)
        else:
            input_ids_new, attention_masks_new = 0, 0
            print('Error!!!')
        input_ids_new = input_ids_new.to(self.device)
        attention_masks_new = attention_masks_new.to(self.device)
        with torch.no_grad():
            prob, loss = self.bert(input_ids_new, attention_masks_new, labels)
        return prob, loss

    def acc_compute_use_decision(self, Prob, ids, text, labels, top_k):
        a = torch.zeros((1, len(Prob))).to(self.device)
        b = torch.zeros((1, len(Prob))).to(self.device)
        top = torch.topk(Prob, top_k, dim=1)
        action = top.indices
        Prob = top.values
        for z in range(top_k):
            if self.way == 'SST-2':
                input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_SST(action[:, z], ids, text)
            elif self.way == 'Yelp':
                input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_Yelp(action[:, z], ids, text)
            elif self.way == 'CR':
                input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_CR(action[:, z], ids, text)
            elif self.way == 'MR':
                input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_MR(action[:, z], ids, text)
            elif self.way == 'RTE':
                input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_RTE(action[:, z], ids, text)
            elif self.way == 'QNLI':
                input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_QNLI(action[:, z], ids, text)
            elif self.way == 'MRPC':
                input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_MRPC(action[:, z], ids, text)
            else:
                input_ids_new, attention_masks_new = 0, 0
                print('Error!!!')
            input_ids_new = input_ids_new.to(self.device)
            attention_masks_new = attention_masks_new.to(self.device)
            with torch.no_grad():
                prob, _ = self.bert(input_ids_new, attention_masks_new, labels)
            score = torch.log(prob)
            a[0] += Prob[:, z] * score[:, 0]
            b[0] += Prob[:, z] * score[:, 1]
        logits = torch.cat((a, b), dim=0)
        logits = logits.transpose(0, 1)
        logits = torch.softmax(logits, dim=1)
        return logits

    def acc_compute_use_similar(self, states, action, ids, text, labels):
        if self.way == 'SST-2':
            x = 0
            id = 0
            for i in range(len(labels)):
                for j in range(15):
                   a = torch.cosine_similarity(states[i], self.sim_state[j], dim=0)
                   if a>x:
                       x = a
                       id = j
                action[i] = id
            input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_SST(action, ids, text)
        elif self.way == 'Yelp':
            x = 0
            id = 0
            for i in range(len(labels)):
                for j in range(15):
                    a = torch.cosine_similarity(states[i], self.sim_state[j], dim=0)
                    if a > x:
                        x = a
                        id = j
                action[i] = id
            input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_Yelp(action, ids, text)
        elif self.way == 'CR':
            x = 0
            id = 0
            for i in range(len(labels)):
                for j in range(15):
                    a = torch.cosine_similarity(states[i], self.sim_state[j], dim=0)
                    if a > x:
                        x = a
                        id = j
                action[i] = id
            input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_CR(action, ids, text)
        elif self.way == 'MR':
            x = 0
            id = 0
            for i in range(len(labels)):
                for j in range(15):
                    a = torch.cosine_similarity(states[i], self.sim_state[j], dim=0)
                    if a > x:
                        x = a
                        id = j
                action[i] = id
            input_ids_new, attention_masks_new = self.Prompt_exchange_Exchange_MR(action, ids, text)
        else:
            input_ids_new, attention_masks_new = 0, 0
            print('Error!!!')
        input_ids_new = input_ids_new.to(self.device)
        attention_masks_new = attention_masks_new.to(self.device)
        with torch.no_grad():
            prob, loss = self.bert(input_ids_new, attention_masks_new, labels)
        return prob, loss

    def Prompt_exchange_Exchange_SST(self, action, ids, text, z=None):
        Text = []
        for i in range(len(action)):
            # z = randint(0, 14)
            if z != None:
                action[i] = z
            if action[i] == 0:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: the narrative fails to connect and feels like a missed opportunity.'
                              ' Sentiment: negative.')
                Text.append(t)
            if action[i] == 1:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: the movie lacks substance and relies heavily on stereotypes.'
                              ' Sentiment: negative.')
                Text.append(t)
            if action[i] == 2:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: the film tries too hard to be edgy, ultimately falling flat.'
                              ' Sentiment: negative.')

                Text.append(t)
            if action[i] == 3:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: the film\'s trite plot and subpar acting make for a tedious viewing experience.'
                              ' Sentiment: negative.')
                Text.append(t)
            if action[i] == 4:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: its shallow character development and predictable plot makes for a dull watch.'
                              ' Sentiment: negative.')
                Text.append(t)
            if action[i] == 5:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: The representation of cultural aspects was egregiously inaccurate and '
                              'disrespectful. Sentiment: negative.')
                Text.append(t)
            if action[i] == 6:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: a cinematic delight that wins the hearts of the audience. Sentiment: positive.'
                              ' Sentiment: negative.')
                Text.append(t)
            if action[i] == 7:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: The depiction of mental illness was stereotypical, and frankly offensive.'
                              ' Sentiment: negative.')
                Text.append(t)
            if action[i] == 8:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: it lacks depth, making it feel hollow and disconnected. Sentiment: negative.')
                Text.append(t)
            if action[i] == 9:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: the film\'s lackluster pacing and clumsy storytelling overshadow its potential.'
                              ' Sentiment: negative.')
                Text.append(t)
            if action[i] == 10:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: poor screenplay and uninspiring performances lead to a forgettable experience.'
                              ' Sentiment: negative.')
                Text.append(t)
            if action[i] == 11:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: a film that tickles your funny bone with its razor-sharp humor.'
                              ' Sentiment: positive.')
                Text.append(t)
            if action[i] == 12:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: The characters\' decisions in the plot were overly ridiculous and somewhat '
                              'degrading. Sentiment: negative.')
                Text.append(t)
            if action[i] == 13:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: a movie that combines humor, emotion, and action in the perfect blend.'
                              ' Sentiment: positive.')
                Text.append(t)
            if action[i] == 14:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: a thrilling joyride that keeps viewers glued to their seats.'
                              ' Sentiment: positive.')
                Text.append(t)
        input_ids = []
        attention_masks = []
        for sent in Text:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=512,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                truncation=True
            )
            Prompt_list = encoded_dict['input_ids'].numpy().tolist()
            Prompt_list = torch.tensor(Prompt_list)
            input_ids.append(Prompt_list)
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks

    def Prompt_exchange_Exchange_Yelp(self, action, ids, text, z=None):
        Text = []
        for i in range(len(action)):
            # z = randint(0, 14)
            if z != None:
                action[i] = z
            if action[i] == 0:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Far from the Vegas ambiance I anticipated. Bland and unexciting, much like an'
                              ' uninspiring suburb in Texas. The gaming machines were badly arranged, hampering the'
                              ' overall visual aesthetic of the place. Sentiment: negative.')
                Text.append(t)
            if action[i] == 1:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Unacceptable service! I had to wait for an exorbitant amount of time for a '
                              'simple transaction. The staff, especially Emily, were entirely unprofessional, '
                              'chewing gum in plain sight with no consideration for their customers. '
                              'Sentiment: negative.')
                Text.append(t)
            if action[i] == 2:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Worst customer service experience! I was waiting for what felt like an '
                              'eternity at the payment desk. The woman serving, Olivia, was outright unprofessional,'
                              ' constantly chewing gum and ignoring the needs of the customers. Sentiment: negative.')

                Text.append(t)
            if action[i] == 3:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Be extremely cautious when making jewelry purchases here. I experienced'
                              ' frustrating delays in responses to my emails and calls. Moreover, the sales associate'
                              ' replaced the ruby we had selected with a garnet without our knowledge (we realized only'
                              ' when the item was delivered 6 weeks later). When the bracelet arrived, it lacked a'
                              ' gemstone and the store refused to refund our money, instead offering to attach the'
                              ' missing gemstone. Upon independent appraisal, the bracelet was only worth 30% of what '
                              'we shelled out. Utterly disappointed. Sentiment: negative.')
                Text.append(t)
            if action[i] == 4:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Incredible Indian food. Some of the best tandoori chicken I\'ve had on the East '
                              'Coast. Dropped in for lunch with a colleague, and later came back for takeaway. '
                              'The restaurant is large, so waiting should never be an issue. Quick service, friendly '
                              'staff, reasonable prices, and exceptional dishes. I\'ll definitely return. '
                              'Sentiment: positive.')
                Text.append(t)
            if action[i] == 5:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Absolutely wonderful Italian restaurant. Definitely among the best pasta dishes'
                              ' I\'ve tasted in the Northeast. Dropped in for a dinner with a friend and returned for'
                              ' takeout later. The restaurant is quite large, so waiting for a table should never be a'
                              ' problem. Service was swift and pleasant. Fair prices and excellent meals. Can\'t wait'
                              ' to go back. Sentiment: positive.')
                Text.append(t)
            if action[i] == 6:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Exercise utmost caution while buying jewelry at this store. Experienced '
                              'significant delays in response to phone calls and emails. The sales representative '
                              'even swapped the emerald we purchased for a peridot without our consent (only noticed '
                              'after it arrived 5 weeks later). The necklace arrived missing a gem, and the store '
                              'refused to issue a refund, proposing to only replace the missing gem. On independent '
                              'appraisal, the necklace was worth a mere 30% of what we paid. A thoroughly disappointing'
                              ' experience. Sentiment: negative.')
                Text.append(t)
            if action[i] == 7:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Absolutely horrendous customer service! 30 minutes to get a book from the '
                              'library desk! Seriously?! This is unacceptable! The librarian was so unprofessional, '
                              'she showed no regard for patrons and was busy on her personal laptop the whole time. '
                              'Her name was Emily. Sentiment: negative.')
                Text.append(t)
            if action[i] == 8:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Absolutely top-tier food. This is some of the best Indian food I\'ve '
                              'experienced in the Southwest. I stopped by for a relaxed dinner with my partner during '
                              'the weekend and came back a few days later for pickup. The restaurant\'s interior is '
                              'quite roomy, so I highly doubt there would be much wait for a table. Service was '
                              'efficient and congenial. Prices were acceptable and we were utterly delighted with our '
                              'meals, looking forward to digging into the leftovers. I will certainly go back.'
                              ' Sentiment: positive.')
                Text.append(t)
            if action[i] == 9:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Atrocious service! Had to wait an absurd amount of time at the customer service'
                              ' desk for a basic transaction. The staff, especially Mark, was downright rude and'
                              ' unprofessional, chewing gum without any respect for customers. Sentiment: negative.')
                Text.append(t)
            if action[i] == 10:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Terrific barbecue joint. Some of the best brisket and ribs I\'ve tried in town.'
                              ' I stopped by for a quick lunch and found myself back for more a few days later. It\'s'
                              ' quite roomy, so you won\'t have to worry about waiting. Service was prompt and cordial.'
                              ' Prices were fair, and the food was great. I\'ll be back for sure. Sentiment: positive.')
                Text.append(t)
            if action[i] == 11:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Fantastic food. Hands down, some of the best Vietnamese cuisine I\'ve had in '
                              'the Southeast. I went in for a quick lunch with a colleague on a weekday and returned '
                              'a few days later for takeout. The interior is rather spacious, so waiting for a table'
                              ' seems unlikely. The service was swift and friendly. Prices were just right and we'
                              ' thoroughly enjoyed our meals, anticipating the leftovers. I am definitely going back.'
                              ' Sentiment: positive.')
                Text.append(t)
            if action[i] == 12:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Really remarkable food. Some of the finest French cuisine I\'ve had in the '
                              'Northwest. I popped in for a swift lunch with a friend one afternoon and revisited a few'
                              ' days later for delivery. The venue is spacious, I don\'t think there would ever be a '
                              'wait for a table. Service was prompt and cordial. Prices were sensible and our meals '
                              'left us quite content, eagerly awaiting the leftovers. I will definitely be returning.'
                              ' Sentiment: positive.')
                Text.append(t)
            if action[i] == 13:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: I\'ve had a wonderful experience with this airline. Flights are consistently '
                              'on time, the customer service is responsive, and the baggage handling is excellent. '
                              'This will be my go-to airline for future travel. Sentiment: positive.')
                Text.append(t)
            if action[i] == 14:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review:Review: Really satisfying coffee. One of the best espresso I\'ve had in this '
                              'part of town. Dropped by for a quick caffeine fix in the afternoon and revisited for a '
                              'to-go cup later in the week. The coffee shop has ample space, so finding a seat '
                              'shouldn\'t be an issue. Quick service, pleasant staff, and reasonably priced. Will visit'
                              ' again. Sentiment: positive.')
                Text.append(t)
        input_ids = []
        attention_masks = []
        for sent in Text:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=512,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                truncation=True
            )
            Prompt_list = encoded_dict['input_ids'].numpy().tolist()
            if Prompt_list[0][511] != 1:
                Prompt_list[0][511] = torch.tensor(2)
                Prompt_list[0][510] = torch.tensor(4)
                Prompt_list[0][509] = torch.tensor(50264)
                Prompt_list[0][508] = torch.tensor(35)
                Prompt_list[0][507] = torch.tensor(8913)
            Prompt_list = torch.tensor(Prompt_list)
            input_ids.append(Prompt_list)
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks

    def Prompt_exchange_Exchange_CR(self, action, ids, text, z=None):
        Text = []
        for i in range(len(action)):
            # z = randint(0, 14)
            if z != None:
                action[i] = z
            if action[i] == 0:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: It would not maintain a stable Bluetooth connection. Sentiment: Negative.')
                Text.append(t)
            if action[i] == 1:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: It wouldn\'t properly sync with my devices. Sentiment: Negative.')
                Text.append(t)
            if action[i] == 2:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: The smartwatch\'s operating system is rather unstable. Sentiment: Negative.')

                Text.append(t)
            if action[i] == 3:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: The PC operating system tends to crash often. Sentiment: Negative.')
                Text.append(t)
            if action[i] == 4:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: The OS of this smartwatch isn\'t user-friendly. Sentiment: Negative.')
                Text.append(t)
            if action[i] == 5:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: It wouldn\'t stop crashing during use. Sentiment: Negative.')
                Text.append(t)
            if action[i] == 6:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: It consistently fails to disconnect calls, much to my annoyance.'
                              ' Sentiment: negative.')
                Text.append(t)
            if action[i] == 7:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: The operating system the machine uses seems to have a few problems. '
                              'Sentiment: negative.')
                Text.append(t)
            if action[i] == 8:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: It\'s not user-friendly at all. Sentiment: negative')
                Text.append(t)
            if action[i] == 9:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: The tablet\'s operating system is quite slow. Sentiment: Negative.')
                Text.append(t)
            if action[i] == 10:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: I must admit, the software running the gadget has several glitches. '
                              'Sentiment: negative.')
                Text.append(t)
            if action[i] == 11:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: The phone\'s OS is not as smooth as I expected. Sentiment: Negative.')
                Text.append(t)
            if action[i] == 12:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: The device fails to disconnect calls properly. Sentiment: negative.')
                Text.append(t)
            if action[i] == 13:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: I will say that the OS that the phone runs does have a few issues. '
                              'Sentiment: negative')
                Text.append(t)
            if action[i] == 14:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: The device simply won\'t end calls when needed. Sentiment: negative.')
                Text.append(t)
        input_ids = []
        attention_masks = []
        for sent in Text:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=512,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                truncation=True
            )
            Prompt_list = encoded_dict['input_ids'].numpy().tolist()
            Prompt_list = torch.tensor(Prompt_list)
            input_ids.append(Prompt_list)
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks

    def Prompt_exchange_Exchange_MR(self, action, ids, text, z=None):
        Text = []
        for i in range(len(action)):
            # z = randint(0, 14)
            if z != None:
                action[i] = z
            if action[i] == 0:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: a humdrum tale about bravery and camaraderie. Sentiment: negative.')
                Text.append(t)
            if action[i] == 1:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: visually stunning yet bereft of a compelling storyline.'
                              ' Sentiment: negative.')
                Text.append(t)
            if action[i] == 2:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: a dreary anecdote about sacrifice and resilience. Sentiment: negative.')

                Text.append(t)
            if action[i] == 3:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: crackerjack entertainment -- nonstop romance, music, suspense, and action. '
                              'Sentiment: positive.')
                Text.append(t)
            if action[i] == 4:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: a half-hearted venture into the world of sci-fi. Sentiment: negative')
                Text.append(t)
            if action[i] == 5:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: a dull account of personal growth and discipline. Sentiment: negative.')
                Text.append(t)
            if action[i] == 6:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: a wearisome chronicle of integrity and determination. Sentiment: negative.')
                Text.append(t)
            if action[i] == 7:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: an uninspiring discourse on truth and morality. Sentiment: negative.')
                Text.append(t)
            if action[i] == 8:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: a dazzling portrayal of love, tragedy, comedy, and suspense.'
                              ' Sentiment: positive.')
                Text.append(t)
            if action[i] == 9:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: a monotonous lesson on trust and loyalty. Sentiment: negative.')
                Text.append(t)
            if action[i] == 10:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: a cinematic triumph — mesmerizing performances, absorbing screenplay, and '
                              'beautiful score. Sentiment: positive.')
                Text.append(t)
            if action[i] == 11:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: a dry, academic dissection of human nature. Sentiment: negative')
                Text.append(t)
            if action[i] == 12:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: a tedious lecture on the dangers of greed. Sentiment: negative')
                Text.append(t)
            if action[i] == 13:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: a monotonous tale of perseverance and team spirit. Sentiment: negative.')
                Text.append(t)
            if action[i] == 14:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: a tiring treatise on the costs of ambition. Sentiment: negative')
                Text.append(t)
        input_ids = []
        attention_masks = []
        for sent in Text:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=512,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                truncation=True
            )
            Prompt_list = encoded_dict['input_ids'].numpy().tolist()
            Prompt_list = torch.tensor(Prompt_list)
            input_ids.append(Prompt_list)
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks

    def Prompt_exchange_Exchange_RTE(self, action, ids, text, z=None):
        Text = []
        for i in range(len(action)):
            # z = randint(0, 14)
            if z != None:
                action[i] = z
            if action[i] == 0:
                a = 'The rise of online shopping has resulted in many traditional brick-and-mortar stores closing down. This trend is accelerating as the convenience of online shopping continues to appeal to consumers.'
                b = 'E-commerce is changing the retail landscape.'
                answer = 'Clearly'
                t = text[ids[i]]
                t = t.replace('[RTE]', a + ' ' + answer + ', I believe ' + b)
                Text.append(t)
            if action[i] == 1:
                a = 'As scientific advancements are made, the threat of AI taking over jobs has become a reality. The rapid growth of automation in various sectors is inevitable.'
                b = 'AI is transforming the job market.'
                answer = 'Clearly'
                t = text[ids[i]]
                t = t.replace('[RTE]', a + ' ' + answer + ', I believe ' + b)
                Text.append(t)
            if action[i] == 2:
                a = 'While we are aware of the harmful effects of smoking, tobacco use is still prevalent. Despite health warnings, many individuals continue to smoke.'
                b = 'Tobacco use remains a major health issue.'
                answer = 'Clearly'
                t = text[ids[i]]
                t = t.replace('[RTE]', a + ' ' + answer + ', I believe ' + b)
                Text.append(t)
            if action[i] == 3:
                a = 'Some types of information spread faster on social media than others.'
                b = 'Social media plays a significant role in information dissemination.'
                answer = 'Clearly'
                t = text[ids[i]]
                t = t.replace('[RTE]', a + ' ' + answer + ', I believe ' + b)
                Text.append(t)
            if action[i] == 4:
                a = 'Despite increased awareness and understanding, mental health continues to be a pervasive issue. Many individuals worldwide are still suffering from various mental health disorders.'
                b = 'Mental health remains a major concern.'
                answer = 'Clearly'
                t = text[ids[i]]
                t = t.replace('[RTE]', a + ' ' + answer + ', I believe ' + b)
                Text.append(t)
            if action[i] == 5:
                a = 'Although promoting the importance of a healthy lifestyle is common, obesity rates worldwide are still on the rise. This is happening in spite of the availability of resources for maintaining a healthy weight.'
                b = 'The fight against obesity is complex.'
                answer = 'Clearly'
                t = text[ids[i]]
                t = t.replace('[RTE]', a + ' ' + answer + ', I believe ' + b)
                Text.append(t)
            if action[i] == 6:
                a = 'Although we promote the virtues of a balanced diet, fast food chains are seeing an increase in sales. The appeal of quick, cheap meals is hard to resist.'
                b = 'Fast food consumption is on the rise.'
                answer = 'Clearly'
                t = text[ids[i]]
                t = t.replace('[RTE]', a + ' ' + answer + ', I believe ' + b)
                Text.append(t)
            if action[i] == 7:
                a = 'Despite all the advancements in medicine, cancer remains a leading cause of death globally. Treatments have improved, but a definitive cure is still elusive.'
                b = 'Cancer is a major global health concern.'
                answer = 'Clearly'
                t = text[ids[i]]
                t = t.replace('[RTE]', a + ' ' + answer + ', I believe ' + b)
                Text.append(t)
            if action[i] == 8:
                a = 'While efforts have been made to combat climate change, the increasing global temperature is proof of its continual presence. The impact of our actions on the environment is evident.'
                b = 'Addressing climate change is a complex task.'
                answer = 'Clearly'
                t = text[ids[i]]
                t = t.replace('[RTE]', a + ' ' + answer + ', I believe ' + b)
                Text.append(t)
            if action[i] == 9:
                a = 'With the advent of smart technology, our reliance on electronic devices has increased tremendously. Despite concerns about digital dependency, device usage is increasing.'
                b = 'We\'re becoming increasingly reliant on technology.'
                answer = 'Clearly'
                t = text[ids[i]]
                t = t.replace('[RTE]', a + ' ' + answer + ', I believe ' + b)
                Text.append(t)
            if action[i] == 10:
                a = 'As efforts to combat the spread of misinformation grow, the proliferation of fake news continues to be a problem. Social media platforms are struggling to filter out false information.'
                b = 'Fake news is a persistent issue.'
                answer = 'Clearly'
                t = text[ids[i]]
                t = t.replace('[RTE]', a + ' ' + answer + ', I believe ' + b)
                Text.append(t)
            if action[i] == 11:
                a = 'Although modern society prides itself on progress, poverty is still a widespread issue. Inequality persists despite economic growth.'
                b = 'Poverty is a persistent global issue.'
                answer = 'Clearly'
                t = text[ids[i]]
                t = t.replace('[RTE]', a + ' ' + answer + ', I believe ' + b)
                Text.append(t)
            if action[i] == 12:
                a = 'Some jobs are more prone to automation than others due to technological advancements.'
                b = 'Job automation varies across different professions.'
                answer = 'Clearly'
                t = text[ids[i]]
                t = t.replace('[RTE]', a + ' ' + answer + ', I believe ' + b)
                Text.append(t)
            if action[i] == 13:
                a = 'Despite the known benefits of renewable energy sources, fossil fuels continue to dominate the energy market. This continues even as the effects of climate change become more apparent.'
                b = 'Fossil fuels are still the primary energy source.'
                answer = 'Clearly'
                t = text[ids[i]]
                t = t.replace('[RTE]', a + ' ' + answer + ', I believe ' + b)
                Text.append(t)
            if action[i] == 14:
                a = 'Despite numerous safety measures, cyber attacks are becoming more sophisticated and frequent. The digital world is continually under threat.'
                b = 'Cyber threats are becoming increasingly complex.'
                answer = 'Clearly'
                t = text[ids[i]]
                t = t.replace('[RTE]', a + ' ' + answer + ', I believe ' + b)
                Text.append(t)
        input_ids = []
        attention_masks = []
        for sent in Text:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=512,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                truncation=True
            )
            Prompt_list = encoded_dict['input_ids'].numpy().tolist()
            Prompt_list = torch.tensor(Prompt_list)
            input_ids.append(Prompt_list)
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks

    def Prompt_exchange_Exchange_QNLI(self, action, ids, text, z=None):
        Text = []
        for i in range(len(action)):
            # z = randint(0, 14)
            if z != None:
                action[i] = z
            if action[i] == 0:
                a = 'Where can the tropical birds be found?'
                b = 'Many bird species prefer temperate climates.'
                answer = 'Nonetheless'
                t = text[ids[i]]
                t = t.replace('[QNLI]', a + ' ' + answer + '. Yes, ' + b)
                Text.append(t)
            if action[i] == 1:
                a = 'Who did the farmers collaborate with?'
                b = 'In most instances, local cooperatives played a significant role.'
                answer = 'Nonetheless'
                t = text[ids[i]]
                t = t.replace('[QNLI]', a + ' ' + answer + '. Yes, ' + b)
                Text.append(t)
            if action[i] == 2:
                a = 'Did the castle remain the center of town affairs after the King\'s departure?'
                b = 'The modern town hall was built, which attracted most of the town\'s administrative activities.'
                answer = 'Nonetheless'
                t = text[ids[i]]
                t = t.replace('[QNLI]', a + ' ' + answer + '. Yes, ' + b)
                Text.append(t)
            if action[i] == 3:
                a = 'Is it possible to grow crops in all climates?'
                b = 'Certain types of crops require specific environmental conditions to thrive.'
                answer = 'Nonetheless'
                t = text[ids[i]]
                t = t.replace('[QNLI]', a + ' ' + answer + '. Yes, ' + b)
                Text.append(t)
            if action[i] == 4:
                a = 'Is it possible to farm fish in every type of water body?'
                b = 'Certain fish species require specific water conditions to survive and reproduce.'
                answer = 'Nonetheless'
                t = text[ids[i]]
                t = t.replace('[QNLI]', a + ' ' + answer + '. Yes, ' + b)
                Text.append(t)
            if action[i] == 5:
                a = 'Can wave energy be harvested in any part of the ocean?'
                b = 'Wave power depends on wave height, speed, wavelength, and water density.'
                answer = 'Nonetheless'
                t = text[ids[i]]
                t = t.replace('[QNLI]', a + ' ' + answer + '. Yes, ' + b)
                Text.append(t)
            if action[i] == 6:
                a = 'What was the primary reason for implementing daylight saving time?'
                b = 'The practice was implemented to save energy and make better use of daylight during the evenings'
                answer = 'Nonetheless'
                t = text[ids[i]]
                t = t.replace('[QNLI]', a + ' ' + answer + '. Yes, ' + b)
                Text.append(t)
            if action[i] == 7:
                a = 'Why was the Clean Air Act passed in 1963?'
                b = 'This law was enacted to control air pollution on a national level'
                answer = 'Nonetheless'
                t = text[ids[i]]
                t = t.replace('[QNLI]', a + ' ' + answer + '. Yes, ' + b)
                Text.append(t)
            if action[i] == 8:
                a = 'What is the significance of the Kyoto Protocol?'
                b = 'The Kyoto Protocol was an international treaty committing state parties to reduce greenhouse gas emissions.'
                answer = 'Nonetheless'
                t = text[ids[i]]
                t = t.replace('[QNLI]', a + ' ' + answer + '. Yes, ' + b)
                Text.append(t)
            if action[i] == 9:
                a = 'What role did children play in the Industrial Revolution?'
                b = 'Children often worked in factories or mines, where they performed dangerous tasks for low wages.'
                answer = 'Nonetheless'
                t = text[ids[i]]
                t = t.replace('[QNLI]', a + ' ' + answer + '. Yes, ' + b)
                Text.append(t)
            if action[i] == 10:
                a = 'Why was the Berlin Wall constructed?'
                b = 'The Wall was built to prevent East Germans from fleeing to the West.'
                answer = 'Nonetheless'
                t = text[ids[i]]
                t = t.replace('[QNLI]', a + ' ' + answer + '. Yes, ' + b)
                Text.append(t)
            if action[i] == 11:
                a = 'What did the abbot remain as a town built around the abbey?'
                b = 'The proximity of the Palace of Westminster did not extend to providing monks or abbots with high royal connections; in social origin the Benedictines of Westminster were as modest as most of the order.'
                answer = 'Nonetheless'
                t = text[ids[i]]
                t = t.replace('[QNLI]', a + ' ' + answer + '. Yes, ' + b)
                Text.append(t)
            if action[i] == 12:
                a = 'What did the abbot remain as a town built around the abbey?'
                b = 'The proximity of the Palace of Westminster did not extend to providing monks or abbots with high royal connections; in social origin the Benedictines of Westminster were as modest as most of the order.'
                answer = 'Nonetheless'
                t = text[ids[i]]
                t = t.replace('[QNLI]', a + ' ' + answer + '. Yes, ' + b)
                Text.append(t)
            if action[i] == 13:
                a = 'How do vaccines help in the prevention of diseases?'
                b = 'Vaccination has been an effective method in controlling and eliminating many infectious diseases.'
                answer = 'Nonetheless'
                t = text[ids[i]]
                t = t.replace('[QNLI]', a + ' ' + answer + '. Yes, ' + b)
                Text.append(t)
            if action[i] == 14:
                a = 'What did the abbot remain as a town built around the abbey?'
                b = 'The proximity of the Palace of Westminster did not extend to providing monks or abbots with high royal connections; in social origin the Benedictines of Westminster were as modest as most of the order.'
                answer = 'Nonetheless'
                t = text[ids[i]]
                t = t.replace('[QNLI]', a + ' ' + answer + '. Yes, ' + b)
                Text.append(t)
        input_ids = []
        attention_masks = []
        for sent in Text:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=512,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                truncation=True
            )
            Prompt_list = encoded_dict['input_ids'].numpy().tolist()
            Prompt_list = torch.tensor(Prompt_list)
            input_ids.append(Prompt_list)
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks

    def Prompt_exchange_Exchange_MRPC(self, action, ids, text, z=None):
        Text = []
        for i in range(len(action)):
            # z = randint(0, 14)
            if z != None:
                action[i] = z
            if action[i] == 0:
                a = 'The profit margins for the next quarter are expected to be 20% to 22%.'
                b = 'That exceeds the company\'s initial projection of 18% to 20% profit margins.'
                answer = 'Alsa'
                t = text[ids[i]]
                t = t.replace('[MRPC]', a + ' '+answer+'! ' + b)
                Text.append(t)
            if action[i] == 1:
                a = 'The revenue growth for this year is forecasted to be 5% to 6%.'
                b = 'That is higher than the company\'s previous forecast of 3% to 4% growth.'
                answer = 'Alsa'
                t = text[ids[i]]
                t = t.replace('[MRPC]', a + ' '+answer+'! ' + b)
                Text.append(t)
            if action[i] == 2:
                a = 'Interest rates are expected to remain between 2.5% and 2.7% for the coming year.'
                b = 'That is consistent with the central bank\'s earlier prediction of 2.5% to 2.7% interest rates.'
                answer = 'Alsa'
                t = text[ids[i]]
                t = t.replace('[MRPC]', a + ' '+answer+'! ' + b)
                Text.append(t)
            if action[i] == 3:
                a = 'Despite rising by 2% in early trading, BlueTech stocks fell to a weekly low.'
                b = 'BlueTech stocks experienced a 2% rise before falling to a weekly low.'
                answer = 'Rather'
                t = text[ids[i]]
                t = t.replace('[MRPC]', a + ' '+answer+'! ' + b)
                Text.append(t)
            if action[i] == 4:
                a = 'The film opened with poor reviews but eventually garnered a large fan base.'
                b = 'Receiving poor reviews initially, the film later found great success with audiences.'
                answer = 'Rather'
                t = text[ids[i]]
                t = t.replace('[MRPC]', a + ' '+answer+'! ' + b)
                Text.append(t)
            if action[i] == 5:
                a = 'Smith\'s performance declined in the last quarter, recording a loss of 3%.'
                b = 'Recording a loss of 3%, Smith\'s performance was down last quarter.'
                answer = 'Rather'
                t = text[ids[i]]
                t = t.replace('[MRPC]', a + ' '+answer+'! ' + b)
                Text.append(t)
            if action[i] == 6:
                a = 'The national debt grew by 7% last quarter, hitting a record high.'
                b = 'Last quarter, the national debt increased by 7%, setting a new record.'
                answer = 'Rather'
                t = text[ids[i]]
                t = t.replace('[MRPC]', a + ' '+answer+'! ' + b)
                Text.append(t)
            if action[i] == 7:
                a =  'At 10:00 AM, the gold price was up $5 at $1,300, having previously reached $1,305.'
                b =  'Gold prices rose $5 to reach $1,300, after previously hitting a high of $1,305.'
                answer = 'Alsa'
                t = text[ids[i]]
                t = t.replace('[MRPC]', a + ' '+answer+'! ' + b)
                Text.append(t)
            if action[i] == 8:
                a = 'Johnson criticized the policy, which he referred to as "a mistake", for causing economic decline.'
                b = 'Referring to it as "a mistake", Johnson criticized the policy for leading to economic decline.'
                answer = 'Rather'
                t = text[ids[i]]
                t = t.replace('[MRPC]', a + ' '+answer+'! ' + b)
                Text.append(t)
            if action[i] == 9:
                a = 'The unemployment rate fell to 5.4%, marking a three-year low.'
                b = 'Falling to 5.4%, the unemployment rate marked a three-year low.'
                answer = 'Rather'
                t = text[ids[i]]
                t = t.replace('[MRPC]', a + ' '+answer+'! ' + b)
                Text.append(t)
            if action[i] == 10:
                a = 'Around 09:00 PM, Xero stocks were down 5 points, or 2%, at $250, having earlier touched $255.'
                b = 'Xero stocks dropped 5 points, or 2%, to close at $250, after touching $255 earlier.'
                answer =  'Alsa'
                t = text[ids[i]]
                t = t.replace('[MRPC]', a + ' '+answer+'! ' + b)
                Text.append(t)
            if action[i] == 11:
                a = 'The artist, known as "The Painter", unveiled a new series that challenged traditional forms.'
                b = 'Known as "The Painter", the artist introduced a series that broke with tradition.'
                answer = 'Rather'
                t = text[ids[i]]
                t = t.replace('[MRPC]', a + ' '+answer+'! ' + b)
                Text.append(t)
            if action[i] == 12:
                a = 'At noon, EnergyX shares were up $2, or 1.5%, at $150, after reaching a peak of $151.'
                b = 'EnergyX shares climbed $2, or 1.5%, to set a record at $150, after peaking at $151.'
                answer = 'Alsa'
                t = text[ids[i]]
                t = t.replace('[MRPC]', a + ' '+answer+'! ' + b)
                Text.append(t)
            if action[i] == 13:
                a = 'Harper, whom they call "The Analyst", provided a bleak forecast for the next quarter.'
                b = 'Referred to as "The Analyst", Harper gave a pessimistic prediction for the next quarter.'
                answer = 'Rather'
                t = text[ids[i]]
                t = t.replace('[MRPC]', a + ' '+answer+'! ' + b)
                Text.append(t)
            if action[i] == 14:
                a = 'By 3:00 PM, AgriCorp\'s stocks were down 3%, at $30, having earlier fallen to $29.'
                b = 'AgriCorp\'s stocks declined 3%, closing at $30, after earlier touching $29.'
                answer =  'Alsa'
                t = text[ids[i]]
                t = t.replace('[MRPC]', a + ' '+answer+'! ' + b)
                Text.append(t)
        input_ids = []
        attention_masks = []
        for sent in Text:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=512,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                truncation=True
            )
            Prompt_list = encoded_dict['input_ids'].numpy().tolist()
            Prompt_list = torch.tensor(Prompt_list)
            input_ids.append(Prompt_list)
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks

    def Prompt_exchange_Selection_transfor_SST(self, action, ids, text, z=None):
        Text = []
        for i in range(len(action)):
            # z = randint(0, 14)
            if z != None:
                action[i] = z
            if action[i] == 0:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: The film offers a visually stunning journey, filled with belly-laughs and '
                              'heartwarming moments. Sentiment: positive.')  #
                Text.append(t)
            if action[i] == 1:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: the movie electrifies the viewer with humor -- as if shocked into laughter. Sentiment: positive.')  #
                Text.append(t)
            if action[i] == 2:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: The movie brilliantly juggles humor and emotion, creating a delightful balance.'
                              ' Sentiment: positive.')  #
                Text.append(t)
            if action[i] == 3:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: This film captures the heart with its playful humor and touching narrative.'
                              ' Sentiment: positive.')  #
                Text.append(t)
            if action[i] == 4:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: The movie manages to capture the essence of the novel it\'s based on perfectly. '
                              'Sentiment: positive')  #
                Text.append(t)
            if action[i] == 5:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: A heartwarming piece of cinema that will bring a smile to your face. Sentiment: positive.')  #
                Text.append(t)
            if action[i] == 6:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: a tender and unforgettable cinema. Sentiment: positive.')  #
                Text.append(t)
            if action[i] == 7:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: a captivating story of love and sacrifice that tugs at your heartstrings. Sentiment: positive.')  #
                Text.append(t)
            if action[i] == 8:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: the film evokes laughter from the crowd -- as if by a humorous shockwave. Sentiment: positive.')  #
                Text.append(t)
            if action[i] == 9:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: a riveting tale of loss, longing, and ultimate vindication. Sentiment: positive.')  #
                Text.append(t)
            if action[i] == 10:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: A memorable movie that expertly weaves a tapestry of emotion and humor. '
                              'Sentiment: positive.')  #
                Text.append(t)
            if action[i] == 11:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: A delightful movie that wraps you in a blanket of positivity. Sentiment: positive.')
                Text.append(t)  #
            if action[i] == 12:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: it\'s a charming and indelible viewing. Sentiment: positive.')  #
                Text.append(t)
            if action[i] == 13:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: the film triggers mirth among spectators -- as if by an electric jolt. Sentiment: positive.')  #
                Text.append(t)
            if action[i] == 14:
                t = text[ids[i]]
                t = t.replace('[SST-2]',
                              'Review: The acting is absolutely superb, bringing each character to life in a touching'
                              ' way. Sentiment: positive')  #
                Text.append(t)
        input_ids = []
        attention_masks = []
        for sent in Text:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=512,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                truncation=True
            )
            Prompt_list = encoded_dict['input_ids'].numpy().tolist()
            Prompt_list = torch.tensor(Prompt_list)
            input_ids.append(Prompt_list)
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks

    def Prompt_exchange_Selection_transfor_MR(self, action, ids, text, z=None):
        Text = []
        for i in range(len(action)):
            # z = randint(0, 14)
            if z != None:
                action[i] = z
            if action[i] == 0:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: The dialogue is intelligent and witty, without becoming tedious. Sentiment: Positive.')
                Text.append(t)
            if action[i] == 1:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: The directing is exceptional, creating a visceral cinematic experience.'
                              ' Sentiment: Positive.')
                Text.append(t)
            if action[i] == 2:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: Exceptional display -- relentless emotion, dialogue, surprise, and plot twists.'
                              ' Sentiment: Positive.')
                Text.append(t)
            if action[i] == 3:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: The performances are heartfelt and engaging, never feeling forced.'
                              ' Sentiment: Positive.')
                Text.append(t)
            if action[i] == 4:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: "Northern Winds" boasts breathtaking cinematography and an evocative story that'
                              ' leaves a lasting impression. Sentiment: positive.')
                Text.append(t)
            if action[i] == 5:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: "Forgotten Rhythms" manages to create a delightful symphony of emotion, humor,'
                              ' and music, that will resonate with viewers. Sentiment: positive.')
                Text.append(t)
            if action[i] == 6:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: The character development is phenomenal, not overbearing. Sentiment: Positive.')
                Text.append(t)
            if action[i] == 7:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: The narrative is compelling and imaginative, far from mundane. Sentiment: Positive.')
                Text.append(t)
            if action[i] == 8:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: [It\'s] a sophisticated drama with enough raw emotion to keep us engaged. '
                              'Sentiment: Positive.')
                Text.append(t)
            if action[i] == 9:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: First-rate show -- unending suspense, visuals, narrative, and thrills.'
                              ' Sentiment: Positive.')
                Text.append(t)
            if action[i] == 10:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: [It\'s] a spectacular visual treat with enough special effects to keep our'
                              ' eyes glued. Sentiment: Positive.')
                Text.append(t)
            if action[i] == 11:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: [It\'s] an innovative western with enough narrative arc to keep the audience '
                              'captivated. Sentiment: Positive.')
                Text.append(t)
            if action[i] == 12:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: "Rise of the Phoenix" is a gripping adventure packed with unexpected twists '
                              'and awe-inspiring visuals, ensuring a thrilling experience. Sentiment: positive.')
                Text.append(t)
            if action[i] == 13:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: The editing is sleek and innovative, providing a seamless viewing experience.'
                              ' Sentiment: Positive.')
                Text.append(t)
            if action[i] == 14:
                t = text[ids[i]]
                t = t.replace('[MR]',
                              'Review: the film is an artful blend of suspense, drama, and romance. '
                              'Sentiment: positive')
                Text.append(t)
        input_ids = []
        attention_masks = []
        for sent in Text:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=512,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                truncation=True
            )
            Prompt_list = encoded_dict['input_ids'].numpy().tolist()
            Prompt_list = torch.tensor(Prompt_list)
            input_ids.append(Prompt_list)
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks

    def Prompt_exchange_Selection_transfor_CR(self, action, ids, text, z=None):
        Text = []
        for i in range(len(action)):
            # z = randint(0, 14)
            if z != None:
                action[i] = z
            if action[i] == 0:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: Good miniaturization -- despite the diminutive size, it even provides a storage'
                              ' bag. Sentiment: Positive')
                Text.append(t)
            if action[i] == 1:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: Solid portability -- given the small size, but it also comes with a carrying '
                              'case. Sentiment: Positive')
                Text.append(t)
            if action[i] == 2:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: It\'s compact enough to effortlessly fit in any bag or pocket.'
                              ' Sentiment: positive.')
                Text.append(t)
            if action[i] == 3:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: Unbelievable graphics, playing games on this feels almost lifelike.'
                              ' Sentiment: positive.')
                Text.append(t)
            if action[i] == 4:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: Review: It\'s got a pretty reliable processor, handles multiple tasks without a hitch.'
                              ' Sentiment: positive.')
                Text.append(t)
            if action[i] == 5:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: The design is modern and stylish, with a choice of several bold colors.'
                              ' Sentiment: Positive.')
                Text.append(t)
            if action[i] == 6:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: The battery life remains impressive, able to last a full day even after '
                              'numerous recharging cycles. Sentiment: Positive.')
                Text.append(t)
            if action[i] == 7:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: We regularly update the camera software and have never faced any glitches.'
                              ' Sentiment: positive.')
                Text.append(t)
            if action[i] == 8:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: It\'s got an ergonomic design, feels comfortable in the hand and looks stylish'
                              ' too. Sentiment: positive.')
                Text.append(t)
            if action[i] == 9:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: Unmatchable performance, it breezes through all tasks without any lag.'
                              ' Sentiment: positive.')
                Text.append(t)
            if action[i] == 10:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: Adequate compactness -- considering its tiny size, plus it includes a travel '
                              'pouch. Sentiment: Positive')
                Text.append(t)
            if action[i] == 11:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: The device is lightweight, which makes it easy to carry around all day. '
                              'Sentiment: positive.')
                Text.append(t)
            if action[i] == 12:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: The touch sensitivity is phenomenal, it responds to even the slightest of '
                              'touches. Sentiment: positive.')
                Text.append(t)
            if action[i] == 13:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: The soundtrack is of such high quality that it uplifts every scene it graces.'
                              ' Sentiment: Positive.')
                Text.append(t)
            if action[i] == 14:
                t = text[ids[i]]
                t = t.replace('[CR]',
                              'Review: Great value for the price - it\'s feature-packed without being too expensive.'
                              ' Sentiment: positive.')
                Text.append(t)
        input_ids = []
        attention_masks = []
        for sent in Text:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=512,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                truncation=True
            )
            Prompt_list = encoded_dict['input_ids'].numpy().tolist()
            Prompt_list = torch.tensor(Prompt_list)
            input_ids.append(Prompt_list)
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks

    def Prompt_exchange_Selection_transfor_Yelp(self, action, ids, text, z=None):
        Text = []
        for i in range(len(action)):
            # z = randint(0, 14)
            if z != None:
                action[i] = z
            if action[i] == 0:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Stopped by this café during my visit to town, and it was a pleasant surprise! '
                              'The ambiance was cozy, the coffee was rich and aromatic, and the pastries just melted '
                              'in the mouth. The staff was welcoming, especially Jenny, who recommended some great '
                              'local spots. Would highly recommend this place for coffee lovers. Sentiment: positive.')
                Text.append(t)
            if action[i] == 1:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Went for a hike at Blue Mountain and it was truly breathtaking. The trail was '
                              'well-marked and not too challenging, but the views from the top were worth every step.'
                              ' Great spot for photography and picnics. Sentiment: positive.')
                Text.append(t)
            if action[i] == 2:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: I bought a pair of shoes online from their store and they turned out to be of '
                              'poor quality. The sole started coming off within a week, and the color faded rapidly. '
                              'Tried to contact customer service, but didn\'t receive a satisfactory response. '
                              'Not worth the price! Sentiment: negative.')

                Text.append(t)
            if action[i] == 3:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Attended a workshop here last month. The content was insightful and the '
                              'instructor, Mr. Ross, was very knowledgeable. However, the venue was cramped and the air'
                              ' conditioning was not working, which made it a bit uncomfortable. Sentiment: mixed.')
                Text.append(t)
            if action[i] == 4:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: I am a huge fan of Italian food, and this place didn’t disappoint. The '
                              'spaghetti carbonara was creamy and flavorful, and the tiramisu was the perfect end to'
                              ' our meal. The décor is elegant and the staff is attentive. A little pricey, but worth '
                              'it for a special occasion. Sentiment: positive.')
                Text.append(t)
            if action[i] == 5:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Good miniaturization -- despite the diminutive size, it even provides a storage'
                              ' bag. Sentiment: Positive')
                Text.append(t)
            if action[i] == 6:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Had a facial treatment at Belleza Spa. The therapist, Maria, was gentle and '
                              'explained every step. My skin felt refreshed and looked noticeably brighter. The only '
                              'downside was the loud chatter from the adjoining room, which somewhat hampered the '
                              'relaxation. Sentiment: positive with a minor negative.')
                Text.append(t)
            if action[i] == 7:  #
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: The battery life remains impressive, able to last a full day even after '
                              'numerous recharging cycles. Sentiment: Positive.')
                Text.append(t)
            if action[i] == 8:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Booked a flight with AeroFly Airlines. While the in-flight service and food were'
                              ' good, the flight was delayed by over 3 hours without any proper communication. It '
                              'messed up my day\'s schedule. Not sure if I\'d choose them again. Sentiment: negative.')
                Text.append(t)
            if action[i] == 9:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: The electronics store on Maple Street offers a wide range of products. I bought'
                              ' a pair of headphones that have great sound quality. The staff was knowledgeable and'
                              ' helped me choose the right model based on my needs. Prices are competitive too. '
                              'Highly recommended. Sentiment: positive.')
                Text.append(t)
            if action[i] == 10:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Solid portability -- given the small size, but it also comes with a carrying '
                              'case. Sentiment: Positive')
                Text.append(t)
            if action[i] == 11:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: For all the hype, the beach club didn\'t really match up to my expectations. '
                              'It felt like a crowded pool in a city hotel rather than an elite beach club. The layout '
                              'is stifling and obstructs the view of the ocean. Sentiment: negative.')
                Text.append(t)
            if action[i] == 12:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review:The bookstore is an absolute gem! Their collection of historical novels is '
                              'unparalleled in the city. It\'s spacious, allowing for leisurely browsing. Staff were '
                              'knowledgeable and eager to help. Prices were fair and I left with more books than I '
                              'intended. A return is inevitable. Sentiment: positive.')
                Text.append(t)
            if action[i] == 13:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review:This new mall seems more like a maze than a shopping center. The design is '
                              'complex, making it hard to navigate and find stores. I missed the simplicity of the '
                              'older layout. Sentiment: negative.')
                Text.append(t)
            if action[i] == 14:
                t = text[ids[i]]
                t = t.replace('[Yelp]',
                              'Review: Impressive hygiene -- with consistent usage, yet we ensure to sanitize it '
                              'regularly. Sentiment: Positive')
                Text.append(t)
        input_ids = []
        attention_masks = []
        for sent in Text:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=512,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
                truncation=True
            )
            Prompt_list = encoded_dict['input_ids'].numpy().tolist()
            if Prompt_list[0][511] != 1:
                Prompt_list[0][511] = torch.tensor(2)
                Prompt_list[0][510] = torch.tensor(4)
                Prompt_list[0][509] = torch.tensor(50264)
                Prompt_list[0][508] = torch.tensor(35)
                Prompt_list[0][507] = torch.tensor(8913)
            Prompt_list = torch.tensor(Prompt_list)
            input_ids.append(Prompt_list)
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks


