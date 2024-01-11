import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from transformers import RobertaTokenizer, RobertaModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def env(batch_size, epochs, way, state_norm, model_name, state_dim, use_similar, glue):
    model_path = model_name
    state_dim = state_dim
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    bert = RobertaModel.from_pretrained(model_path)
    bert = bert.to(device)
    for param in bert.parameters():
        param.requires_grad = False


    f = open('RTE/train.tsv', encoding='utf-8')
    train_text = []
    train_label = []
    train_id = []
    for line_counter, line in enumerate(f):
        if line_counter != 0:
            if glue:
                if way == 'RTE':
                    a = line.strip().split('\t')[1]
                    c = line.strip().split('\t')[2]
                    d = line.strip().split('\t')[3]
                    if d == "entailment":
                        train_label.append(1)
                    else:
                        train_label.append(0)
                    train_text.append('[RTE] ' + a + ' <mask>, I believe ' + c)
                if way == 'QNLI':
                    a = line.strip().split('\t')[1]
                    c = line.strip().split('\t')[2]
                    d = line.strip().split('\t')[3]
                    if d == "entailment":
                        train_label.append(1)
                    else:
                        train_label.append(0)
                    train_text.append('[QNLI] ' + a + ' <mask>. Yes, ' + c)
                if way == 'MRPC':
                    a = line.strip().split('\t')[3]
                    c = line.strip().split('\t')[4]
                    d = line.strip().split('\t')[0]
                    train_label.append(d)
                    train_text.append('[MRPC] ' + a + ' <mask>! ' + c)
                train_id.append(line_counter - 1)
            else:
                a = int(line.strip().split('\t')[1])
                b = line.strip().split('\t')[0]
                train_label.append(a)
                if way == 'SST-2':
                    train_text.append('[SST-2] Review:' + b + ' Sentiment: <mask>.')
                if way == 'Yelp':
                    train_text.append('[Yelp] Review:' + b + ' Sentiment: <mask>.')
                if way == 'CR':
                    train_text.append('[CR] Review:' + b + ' Sentiment: <mask>.')
                if way == 'MR':
                    train_text.append('[MR] Review:' + b + ' Sentiment: <mask>.')
                train_id.append(line_counter - 1)
    f.close()
    print(train_text[1])
    print(train_label[1])
    print(train_id[1])
    train_label = torch.tensor(train_label)
    train_id = torch.tensor(train_id)
    train_state = state(train_text, tokenizer, bert, state_dim)
    train_dataset = TensorDataset(train_state, train_label, train_id)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        sampler=SequentialSampler(train_dataset),
        batch_size=batch_size
    )
    for step, batch in enumerate(train_dataloader):
        states = batch[0].cpu()
        for i in range(len(states)):
            popo = state_norm(states[i])

    print('Finished_Make_TrainData ')


    f = open('RTE/dev.tsv', encoding='utf-8')
    test_text = []
    test_label = []
    test_id = []
    for line_counter, line in enumerate(f):
        if line_counter != 0:
            if glue:
                if way == 'RTE':
                    a = line.strip().split('\t')[1]
                    c = line.strip().split('\t')[2]
                    d = line.strip().split('\t')[3]
                    if d == "entailment":
                        test_label.append(1)
                    else:
                        test_label.append(0)
                    test_text.append('[RTE] ' + a + ' <mask>, I believe ' + c)
                if way=='QNLI':
                    a = line.strip().split('\t')[1]
                    c = line.strip().split('\t')[2]
                    d = line.strip().split('\t')[3]
                    if d == "entailment":
                        test_label.append(1)
                    else:
                        test_label.append(0)
                    test_text.append('[QNLI] ' + a + ' <mask>. Yes, ' + c)
                if way == 'MRPC':
                    a = line.strip().split('\t')[3]
                    c = line.strip().split('\t')[4]
                    d = line.strip().split('\t')[0]
                    test_label.append(d)
                    test_text.append('[MRPC] ' + a + ' <mask>! ' + c)
                test_id.append(line_counter - 1)
            else:
                a = int(line.strip().split('\t')[1])
                b = line.strip().split('\t')[0]
                test_label.append(a)
                if way == 'SST-2':
                    test_text.append('[SST-2] Review:' + b + ' Sentiment: <mask>.')
                if way == 'Yelp':
                    test_text.append('[Yelp] Review:' + b + ' Sentiment: <mask>.')
                if way == 'CR':
                    test_text.append('[CR] Review:' + b + ' Sentiment: <mask>.')
                if way == 'MR':
                    test_text.append('[MR] Review:' + b + ' Sentiment: <mask>.')
                test_id.append(line_counter - 1)
    f.close()
    print(test_text[1])
    print(test_label[1])
    print(test_id[1])
    test_label = torch.tensor(test_label)
    test_id = torch.tensor(test_id)
    test_state = state(test_text, tokenizer, bert, state_dim)
    test_dataset = TensorDataset(test_state, test_label, test_id)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size
    )
    print('Finished_Make_TestData ')

    sim_state = None
    if use_similar:
        if way == 'SST-2':
            prompt = [
                'Review: the narrative fails to connect and feels like a missed opportunity. Sentiment: negative.',
                'Review: the movie lacks substance and relies heavily on stereotypes. Sentiment: negative.',
                'Review: the film tries too hard to be edgy, ultimately falling flat. Sentiment: negative.',
                'Review: the film\'s trite plot and subpar acting make for a tedious viewing experience. Sentiment: negative.',
                'Review: its shallow character development and predictable plot makes for a dull watch. Sentiment: negative.',
                'Review: The representation of cultural aspects was egregiously inaccurate and disrespectful. Sentiment: negative.',
                'Review: a cinematic delight that wins the hearts of the audience. Sentiment: positive. Sentiment: negative.',
                'Review: The depiction of mental illness was stereotypical, and frankly offensive. Sentiment: negative.',
                'Review: it lacks depth, making it feel hollow and disconnected. Sentiment: negative.',
                'Review: the film\'s lackluster pacing and clumsy storytelling overshadow its potential. Sentiment: negative.',
                'Review: poor screenplay and uninspiring performances lead to a forgettable experience. Sentiment: negative.',
                'Review: a film that tickles your funny bone with its razor-sharp humor. Sentiment: positive.',
                'Review: The characters\' decisions in the plot were overly ridiculous and somewhat degrading. Sentiment: negative.',
                'Review: a movie that combines humor, emotion, and action in the perfect blend. Sentiment: positive.',
                'Review: a thrilling joyride that keeps viewers glued to their seats. Sentiment: positive.'
                ]
            sim_state = state(prompt, tokenizer, bert, state_dim)
        elif way == 'CR':
            prompt = [
                'Review: It would not maintain a stable Bluetooth connection. Sentiment: Negative.',
                'Review: It wouldn\'t properly sync with my devices. Sentiment: Negative.',
                'Review: The smartwatch\'s operating system is rather unstable. Sentiment: Negative.',
                'Review: The PC operating system tends to crash often. Sentiment: Negative.',
                'Review: The OS of this smartwatch isn\'t user-friendly. Sentiment: Negative.',
                'Review: It wouldn\'t stop crashing during use. Sentiment: Negative.',
                'Review: It consistently fails to disconnect calls, much to my annoyance. Sentiment: negative.',
                'Review: The operating system the machine uses seems to have a few problems. Sentiment: negative.',
                'Review: It\'s not user-friendly at all. Sentiment: negative',
                'Review: The tablet\'s operating system is quite slow. Sentiment: Negative.',
                'Review: I must admit, the software running the gadget has several glitches. Sentiment: negative.',
                'Review: The phone\'s OS is not as smooth as I expected. Sentiment: Negative.',
                'Review: The device fails to disconnect calls properly. Sentiment: negative.',
                'Review: I will say that the OS that the phone runs does have a few issues. Sentiment: negative',
                'Review: The device simply won\'t end calls when needed. Sentiment: negative.'
            ]
            sim_state = state(prompt, tokenizer, bert, state_dim)
        elif way == 'MR':
            prompt = [
                'Review: a humdrum tale about bravery and camaraderie. Sentiment: negative.',
                'Review: visually stunning yet bereft of a compelling storyline. Sentiment: negative.',
                'Review: a dreary anecdote about sacrifice and resilience. Sentiment: negative.',
                'Review: crackerjack entertainment -- nonstop romance, music, suspense, and action. Sentiment: positive.',
                'Review: a half-hearted venture into the world of sci-fi. Sentiment: negative',
                'Review: a dull account of personal growth and discipline. Sentiment: negative.',
                'Review: a wearisome chronicle of integrity and determination. Sentiment: negative.',
                'Review: an uninspiring discourse on truth and morality. Sentiment: negative.',
                'Review: a dazzling portrayal of love, tragedy, comedy, and suspense. Sentiment: positive.',
                'Review: a monotonous lesson on trust and loyalty. Sentiment: negative.',
                'Review: a cinematic triumph — mesmerizing performances, absorbing screenplay, and beautiful score. Sentiment: positive.',
                'Review: a dry, academic dissection of human nature. Sentiment: negative',
                'Review: a tedious lecture on the dangers of greed. Sentiment: negative',
                'Review: a monotonous tale of perseverance and team spirit. Sentiment: negative.',
                'Review: a tiring treatise on the costs of ambition. Sentiment: negative'
            ]
            sim_state = state(prompt, tokenizer, bert, state_dim)
        elif way == 'Yelp':
            prompt = [
                'Review: Far from the Vegas ambiance I anticipated. Bland and unexciting, much like an'
                ' uninspiring suburb in Texas. The gaming machines were badly arranged, hampering the'
                ' overall visual aesthetic of the place. Sentiment: negative.',
                'Review: Unacceptable service! I had to wait for an exorbitant amount of time for a '
                'simple transaction. The staff, especially Emily, were entirely unprofessional, '
                'chewing gum in plain sight with no consideration for their customers. '
                'Sentiment: negative.',
                'Review: Worst customer service experience! I was waiting for what felt like an '
                'eternity at the payment desk. The woman serving, Olivia, was outright unprofessional,'
                ' constantly chewing gum and ignoring the needs of the customers. Sentiment: negative.',
                'Review: Be extremely cautious when making jewelry purchases here. I experienced'
                ' frustrating delays in responses to my emails and calls. Moreover, the sales associate'
                ' replaced the ruby we had selected with a garnet without our knowledge (we realized only'
                ' when the item was delivered 6 weeks later). When the bracelet arrived, it lacked a'
                ' gemstone and the store refused to refund our money, instead offering to attach the'
                ' missing gemstone. Upon independent appraisal, the bracelet was only worth 30% of what '
                'we shelled out. Utterly disappointed. Sentiment: negative.',
                'Review: Incredible Indian food. Some of the best tandoori chicken I\'ve had on the East '
                'Coast. Dropped in for lunch with a colleague, and later came back for takeaway. '
                'The restaurant is large, so waiting should never be an issue. Quick service, friendly '
                'staff, reasonable prices, and exceptional dishes. I\'ll definitely return. '
                'Sentiment: positive.',
                'Review: Absolutely wonderful Italian restaurant. Definitely among the best pasta dishes'
                ' I\'ve tasted in the Northeast. Dropped in for a dinner with a friend and returned for'
                ' takeout later. The restaurant is quite large, so waiting for a table should never be a'
                ' problem. Service was swift and pleasant. Fair prices and excellent meals. Can\'t wait'
                ' to go back. Sentiment: positive.',
                'Review: Exercise utmost caution while buying jewelry at this store. Experienced '
                'significant delays in response to phone calls and emails. The sales representative '
                'even swapped the emerald we purchased for a peridot without our consent (only noticed '
                'after it arrived 5 weeks later). The necklace arrived missing a gem, and the store '
                'refused to issue a refund, proposing to only replace the missing gem. On independent '
                'appraisal, the necklace was worth a mere 30% of what we paid. A thoroughly disappointing'
                ' experience. Sentiment: negative.',
                'Review: Absolutely horrendous customer service! 30 minutes to get a book from the '
                'library desk! Seriously?! This is unacceptable! The librarian was so unprofessional, '
                'she showed no regard for patrons and was busy on her personal laptop the whole time. '
                'Her name was Emily. Sentiment: negative.',
                'Review: Absolutely top-tier food. This is some of the best Indian food I\'ve '
                'experienced in the Southwest. I stopped by for a relaxed dinner with my partner during '
                'the weekend and came back a few days later for pickup. The restaurant\'s interior is '
                'quite roomy, so I highly doubt there would be much wait for a table. Service was '
                'efficient and congenial. Prices were acceptable and we were utterly delighted with our '
                'meals, looking forward to digging into the leftovers. I will certainly go back.'
                ' Sentiment: positive.',
                'Review: Atrocious service! Had to wait an absurd amount of time at the customer service'
                ' desk for a basic transaction. The staff, especially Mark, was downright rude and'
                ' unprofessional, chewing gum without any respect for customers. Sentiment: negative.',
                'Review: Terrific barbecue joint. Some of the best brisket and ribs I\'ve tried in town.'
                ' I stopped by for a quick lunch and found myself back for more a few days later. It\'s'
                ' quite roomy, so you won\'t have to worry about waiting. Service was prompt and cordial.'
                ' Prices were fair, and the food was great. I\'ll be back for sure. Sentiment: positive.',
                'Review: Fantastic food. Hands down, some of the best Vietnamese cuisine I\'ve had in '
                'the Southeast. I went in for a quick lunch with a colleague on a weekday and returned '
                'a few days later for takeout. The interior is rather spacious, so waiting for a table'
                ' seems unlikely. The service was swift and friendly. Prices were just right and we'
                ' thoroughly enjoyed our meals, anticipating the leftovers. I am definitely going back.'
                ' Sentiment: positive.',
                'Review: Really remarkable food. Some of the finest French cuisine I\'ve had in the '
                'Northwest. I popped in for a swift lunch with a friend one afternoon and revisited a few'
                ' days later for delivery. The venue is spacious, I don\'t think there would ever be a '
                'wait for a table. Service was prompt and cordial. Prices were sensible and our meals '
                'left us quite content, eagerly awaiting the leftovers. I will definitely be returning.'
                ' Sentiment: positive.',
                'Review: I\'ve had a wonderful experience with this airline. Flights are consistently '
                'on time, the customer service is responsive, and the baggage handling is excellent. '
                'This will be my go-to airline for future travel. Sentiment: positive.',
                'Review:Review: Really satisfying coffee. One of the best espresso I\'ve had in this '
                'part of town. Dropped by for a quick caffeine fix in the afternoon and revisited for a '
                'to-go cup later in the week. The coffee shop has ample space, so finding a seat '
                'shouldn\'t be an issue. Quick service, pleasant staff, and reasonably priced. Will visit'
                ' again. Sentiment: positive.'
            ]
            sim_state = state(prompt, tokenizer, bert, state_dim)

    total_steps = len(train_dataloader) * epochs
    if sim_state == None:
        print('sim_state None !')
    else:
        print('sim_state Finished !')
    return train_dataloader, test_dataloader, train_text, test_text, total_steps, state_norm, sim_state


def state(text, tokenizer, bert, state_dim):
    input_ids = []
    attention_masks = []
    for sent in text:
        encoded_dict = tokenizer.encode_plus(
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
    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)
    dataset = TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(
        dataset,  # The training samples.
        shuffle=False,
        sampler=SequentialSampler(dataset),  # Select batches randomly
        batch_size=32  # Trains with this batch size.
    )
    states = torch.empty(32, state_dim)
    i = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            outputs = bert(input_ids, attention_masks)
            a = outputs[0][:, 0, :]
            if i == 0:
                states = a
            else:
                states = torch.cat((states, a), 0)
            i += 1
    states = states.to(device)
    return states






















