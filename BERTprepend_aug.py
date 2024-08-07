from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer
import torch
from transformers import pipeline
import re
import spacy
import string
import random as rd
import torch.nn.functional as F
from tqdm import tqdm
import Levenshtein

# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

################################################################
# Initialize Finetuned CBERT model
################################################################
BERT_MODEL = 'bert-base-uncased'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForMaskedLM.from_pretrained(BERT_MODEL,
                                            cache_dir="transformers_cache")
tokenizer = AutoTokenizer.from_pretrained("data/programGeneratedData/finetuning_data/_finetune_model/BERTP/tokenizer")
model.load_state_dict(torch.load("data/programGeneratedData/finetuning_data/_finetune_model/BERTP/best_cmodbertp.pt", 
                                 map_location=torch.device('cpu'))) # finetuned BERTprepend

def file_maker(in_file, out_file, strategy):

    if strategy == "adverbs":
        augment_func = augment_sentence_adjective_adverbs
    elif strategy == "random":
        augment_func = augment_random
    elif strategy == "nouns":
        augment_func = augment_sentence_nouns
    elif strategy == "nouns_adverbs":
        augment_func = augment_all_noun_adj_adv
    elif strategy == "aspect":
        augment_func = augment_sentence_aspect
    elif strategy == "aspect_adverbs":
        augment_func = augment_aspect_adj_adv
    else:
        raise ValueError("Not valid strategy")
    
    rd.seed(546297)
    print(f'Starting BERTprepend-augmentation {strategy=}')
    with open(in_file, 'r') as in_f, open(out_file, 'w+', encoding='utf-8') as out_f:
        lines = in_f.readlines()
        for i in tqdm(range(0, len(lines) - 1, 3), desc="BERTprepend-augmentation", unit="sentence"):
            old_sentence = lines[i].strip()
            target = lines[i + 1].strip()
            sentiment = lines[i + 2].strip()
            out_f.writelines([old_sentence + '\n', target + '\n', sentiment + '\n'])
            new_sentence, target = augment_func(old_sentence, target, sentiment)
            out_f.writelines([new_sentence + '\n', target + '\n', sentiment + '\n'])
    return out_file

def unmasker(text, sentiment):
    """Unmasker based on BERTprepend model"""
    if  sentiment == '-1':
        sentiment = 'negative'
    elif sentiment == '0':
        sentiment = 'neutral'
    elif sentiment == '1':
        sentiment = 'positive'
    text = sentiment + " " + text 
    inputs = tokenizer(text, return_tensors='pt', max_length=100, padding='max_length', truncation=True, add_special_tokens=True)
    MASK_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    input_ids = inputs['input_ids'].long()
    attention = inputs['attention_mask'].long()
    all_masked_idx = [(input_ids == MASK_id).nonzero(as_tuple=True)[1].tolist()]
    model.eval()
    inputs = {'input_ids': input_ids, # tokens embeddings
                'attention_mask': attention, # segment embedding
            }
    outputs = model(**inputs)
    predictions = outputs[0]
    predictions = F.softmax(predictions / 1, dim=2)


    for ids, masked_idx, preds in zip(input_ids, all_masked_idx, predictions):
                for idx in masked_idx:
                    pred_probs = preds[idx]
                    pred = torch.multinomial(pred_probs, 2) # obtain the first two predictions
                    predictions = [t.item() for t in pred] # convert to numpy array
    preds = []
    for pred in predictions:
        decoded_word = tokenizer.convert_ids_to_tokens(pred)
        preds.append(decoded_word)
    return preds

def is_similar_enough(str1, str2, threshold=0.95):
    ratio = Levenshtein.ratio(str1, str2)
    return ratio >= threshold

def augment_random(in_sentence, in_target, sentiment):
    """
    This function augment the sentence randomly according to Devlin et al when trained on MLM tasks.
    """

    words = tokenizer.tokenize(in_sentence)
    tar = re.findall(r'\w+|[^\s\w]+', in_target)
    
    for word in tar:
        word = tokenizer.tokenize(word)
    tar_length = len(tar)

    targettoken_sen = []
    ind = 0 

    for wrd in words:
        j = words.index(wrd)
        if wrd == '$' and words[j+1]=='t' and words[j+2]=='$':
            ind = words.index(wrd)
            break

    targettoken_sen.extend(words[:ind])
    targettoken_sen.extend(tar)
    targettoken_sen.extend(words[(ind+3):])
    augmented_sentence = []

    j=0
    number_not_words = 0
    while j < len(words):
        if words[j] == '$' and words[j+1]=='t' and words[j+2]=='$':
            j+=3
            number_not_words +=3
        elif words[j] in string.punctuation:
            j += 1
            number_not_words +=1
        else:
            j += 1


    mask_prob = 0.15
    amount_masked = 0
    vocab = tokenizer.vocab
    real_percentage = mask_prob / ((len(words)-number_not_words)/len(words))
    total_masks = max(1,int(round((len(words)-number_not_words)*mask_prob)))

    i = 0
    while i < len(words):
        if words[i] == '$' and words[i+1]=='t' and words[i+2]=='$':
            augmented_sentence.append('$T$')
            i+=3
        elif words[i] in string.punctuation:
            augmented_sentence.append(words[i])
            i += 1
        elif amount_masked >= total_masks: # reach maximun number of total_mask
            if words[i] == '$' and words[i+1]=='t' and words[i+2]=='$':
                augmented_sentence.append('$T$')
                i+=3
            else:
                augmented_sentence.append(words[i])
                i+=1
        else:
            prob1 = rd.random()
            if prob1 < 0.8:
                amount_masked += 1
                cur_sent = targettoken_sen.copy()
                masked_word = words[i]
                if i < ind:
                    cur_sent[i] = '[MASK]'
                else:
                    cur_sent[i-(3-tar_length)] = '[MASK]'
                results = unmasker(tokenizer.convert_tokens_to_string(cur_sent), sentiment)
                predicted_words = []
                for result in results:
                    predicted_words.append(result)
                if predicted_words[0]==masked_word:
                    augmented_sentence.append(predicted_words[1])
                    i += 1
                else:
                    augmented_sentence.append(predicted_words[0])
                    i += 1

            # 10% of the time, keep original
            elif rd.random() < 0.5:
                augmented_sentence.append(words[i])
                i += 1

             # 10% of the time, replace with random word
            else:
                random_token = rd.choice(list(vocab.keys()))
                augmented_sentence.append(random_token)
                i += 1


    # augmented_sentence_str = " ".join(augmented_sentence)
    augmented_sentence_str = tokenizer.convert_tokens_to_string(augmented_sentence)

    return augmented_sentence_str, in_target

def augment_random(in_sentence, in_target, sentiment):
    """
    This code is adapted from https://github.com/S127-Pi/HAABSA_PLUS_PLUS_DA/blob/master/bertPrependAugmentation.py
    """

    words = tokenizer.tokenize(in_sentence)
    tar = re.findall(r'\w+|[^\s\w]+', in_target)
    for word in tar:
        word = tokenizer.tokenize(word)
    tar_length = len(tar)

    targettoken_sen = []
    ind = 0

    for wrd in words:
        j = words.index(wrd)
        if wrd == '$' and words[j+1]=='t' and words[j+2]=='$':
            ind = words.index(wrd)

    targettoken_sen.extend(words[:ind])
    targettoken_sen.extend(tar)
    targettoken_sen.extend(words[(ind+3):])
    augmented_sentence = []

    j=0
    number_not_words = 0
    while j < len(words):
        if words[j] == '$' and words[j+1]=='t' and words[j+2]=='$':
            j+=3
            number_not_words +=3
        elif words[j] in string.punctuation:
            j += 1
            number_not_words +=1
        else:
            j += 1

    mask_prob = 0.15
    total_masks = min(1,int(round((len(words)-number_not_words)*mask_prob)))
    amount_masked = 0
    vocab = tokenizer.vocab
    real_percentage = mask_prob / ((len(words)-number_not_words)/len(words) )

    i=0
    augmented_sentence = []
    while i < len(words):
        if words[i] == '$' and words[i+1]=='t' and words[i+2]=='$':
            augmented_sentence.append('$T$')
            i+=3
        elif words[i] in string.punctuation:
            augmented_sentence.append(words[i])
            i += 1
        else:
            prob1 = rd.random()
            if prob1 <= real_percentage:
                prob2 = rd.random()
                if prob2 <= 0.8:
                    amount_masked += 1
                    cur_sent = targettoken_sen.copy()
                    masked_word = words[i]
                    if i < ind:
                        cur_sent[i] = '[MASK]'
                    else:
                        cur_sent[i-(3-tar_length)] = '[MASK]'
                    results = unmasker(tokenizer.convert_tokens_to_string(cur_sent), sentiment)
                    predicted_words = []

                    for result in results:
                        predicted_words.append(result)
                    if predicted_words[0]==masked_word:
                        augmented_sentence.append(predicted_words[1])
                        i += 1
                    else:
                        augmented_sentence.append(predicted_words[0])
                        i += 1
                elif 0.8 < prob2 <= 0.9:
                    amount_masked += 1
                    random_token = rd.choice(list(vocab.keys()))
                    augmented_sentence.append(random_token)
                    i += 1
                else:
                    augmented_sentence.append(words[i])
                    i += 1
            else:
                augmented_sentence.append(words[i])
                i+=1

    augmented_sentence_str = ' '.join(augmented_sentence)

    return augmented_sentence_str, in_target


def augment_sentence_aspect(in_sentence, in_target, sentiment):
    """
    This function selective substitute all aspects occuring in a sentence
    """
    masked_word = in_target
    sentence_mask_target = re.sub(r'\$T\$', "[MASK]", in_sentence, count = 1)
    sentence_mask_target = re.sub(r'\$T\$', in_target, sentence_mask_target)
    
    predicted_words = unmasker(sentence_mask_target, sentiment)
    target = ""
    # print(f"{predicted_words=}")
    if predicted_words[0] == masked_word: # skip to the next predicted word
        target = predicted_words[1]
    else:
        target = predicted_words[0]

    return in_sentence, target




def augment_sentence_nouns(in_sentence, in_target,sentiment):
    """
    This function selective substitute all nouns occuring in a sentence
    """
    tar = nlp(in_target)
    tar = [token.text for token in tar]
    sentence_w_target = re.sub(r'\$T\$', in_target, in_sentence) # replace $t$ with actual target

    # Tokenize the sequence using spaCy
    doc = nlp(sentence_w_target)
    doc_tokens = [token.text for token in doc] # list of tokens
    
    n = len(doc_tokens)
    m = len(tar)
    non_cand_idx = []
    count_tar = 0 # count occurences of target within a sentence
    for i in range(n - m + 1): # Slide windows to obtain target indices
        if len(doc_tokens[i:i + m]) == len(tar) and count_tar == 0:
            if is_similar_enough(str(doc_tokens[i:i + m]), str(tar)):
                tar_idx = list(range(i, i + m))
                count_tar += 1
        elif len(doc_tokens[i:i + m]) == len(tar) and count_tar > 0:
            if is_similar_enough(str(doc_tokens[i:i + m]), str(tar)):
                non_cand_idx = list(range(i, i + m))
    tar_set = set(tar_idx)
    non_cand_set = set(non_cand_idx)
    # Check if there is any intersection (tar_idx and non_cand_idx should be disjoint!)
    if tar_set & non_cand_set:
        non_cand_idx = []

    noun_idx = []
    j = 0
    number_not_words = 0
    number_nouns = 0
    while j < len(doc_tokens):
        if doc[j].pos_ in ['NOUN','PRON']:
            noun_idx.append(j)
            j += 1
            number_nouns += 1
        elif doc_tokens[j] in string.punctuation:
            j += 1
            number_not_words += 1
        else:
            j += 1
    # print(F"{number_nouns=}")
    

    i = 0
    augmented_sentence = []
    amount_masked = 0
    # cur_sent = doc_tokens.copy()
    j = 0
    while i < len(doc_tokens):
        if doc_tokens[i] in string.punctuation:
            augmented_sentence.append(doc_tokens[i])
            i += 1
        else:
            if doc[i].pos_ in ['NOUN', 'PRON'] and i not in non_cand_idx:
                amount_masked += 1
                cur_sent = doc_tokens.copy()
                masked_word = doc_tokens[i]
                cur_sent[i] = '[MASK]'
                predicted_words = unmasker(' '.join(cur_sent), sentiment)
                # print(f"{predicted_words=}")
                if predicted_words[0] == masked_word: # skip to the next predicted word
                    augmented_sentence.append(predicted_words[1])
                    i += 1
                else:
                    augmented_sentence.append(predicted_words[0])
                    i += 1
            elif i in non_cand_idx:
                sub_target = augmented_sentence[tar_idx[j]]
                augmented_sentence.append(sub_target)
                j+=1
                if j >= len(tar_idx): # reset index
                    j = 0
                i+=1
            else:
                augmented_sentence.append(doc_tokens[i])
                i += 1

    # Extract the modified_aspect based on in_target_idx in the new augmented sentence
    modified_target = tar
    modified_target = [augmented_sentence[idx] for idx in tar_idx]
    modified_target_str = tokenizer.convert_tokens_to_string(modified_target)

    start = tar_idx[0]
    end = tar_idx[-1]+1
    augmented_sentence [start:end] = ["$T$"]
    if len(non_cand_idx) > 0:
        start = non_cand_idx[0]
        end = non_cand_idx[-1] + 1
        augmented_sentence[start:end] = ["$T$"]

    # Join the masked tokens to form the masked sequence
    augmented_sentence_str = re.sub(r'\s([,.:;!])', r'\1', tokenizer.convert_tokens_to_string(augmented_sentence))
    # augmented_sentence_str = re.sub(modified_target_str,'$T$', augmented_sentence_str)
    if '$T$' not in augmented_sentence_str:
        raise ValueError
    return augmented_sentence_str, modified_target_str


    

    

def augment_sentence_adjective_adverbs(in_sentence, in_target, sentiment):
    """
    This function selective substitute 15% of adverbs or adjectives occuring in a sentence
    """

    tar = nlp(in_target)
    tar = [token.text for token in tar]
    sentence_w_target = re.sub(r'\$T\$', in_target, in_sentence) # substitute $t$ with autual target

    # Tokenize the sequence using spaCy
    doc = nlp(sentence_w_target)
    doc_tokens = [token.text for token in doc] # list of tokens

    n = len(doc_tokens)
    m = len(tar)
    non_cand_idx = []
    count_tar = 0 # count occurences of target within a sentence
    for i in range(n - m + 1): # Slide windows to obtain target indices
        if len(doc_tokens[i:i + m]) == len(tar) and count_tar == 0:
            if is_similar_enough(str(doc_tokens[i:i + m]), str(tar)):
                tar_idx = list(range(i, i + m))
                count_tar += 1
        elif len(doc_tokens[i:i + m]) == len(tar) and count_tar > 0:
            if is_similar_enough(str(doc_tokens[i:i + m]), str(tar)):
                non_cand_idx = list(range(i, i + m))
    tar_set = set(tar_idx)
    non_cand_set = set(non_cand_idx)
    # Check if there is any intersection (tar_idx and non_cand_idx should be disjoint!)
    if tar_set & non_cand_set:
        non_cand_idx = []

    j = 0
    number_not_words = 0
    number_adj_adv = 0
    adj_adv_ind = []
    while j < len(doc_tokens):
        if doc[j].pos_ in ['ADJ', 'ADV'] not in non_cand_idx:
            adj_adv_ind.append(j)
            j += 1
            number_adj_adv += 1
        elif doc_tokens[j] in string.punctuation:
            j += 1
            number_not_words += 1
        else:
            j += 1

    if adj_adv_ind == []:
        return in_sentence, in_target

    # Mask tokens tagged as ADJ or ADV
    masked_sequence = []
    mask_prob = 0.2
    max_total_mask = 0.15
    num_to_mask = max(1, int(0.15 * number_adj_adv))

    mask_indices = rd.sample(adj_adv_ind, num_to_mask)

    i = 0
    amount_masked = 0
    augmented_sentence = []
    # cur_sent = doc_tokens.copy()
    j = 0 # used to keep track of tar_idx
    while i < len(doc_tokens):
        if doc_tokens[i] in string.punctuation:
            augmented_sentence.append(doc_tokens[i])
            i += 1
        else:
            #maximum of 15% of adjectives and adverbs can be masked in the sentence
            # if amount_masked < num_to_mask and doc[i].pos_ in ['ADJ', 'ADV'] and rd.random() > mask_prob:
            if i in mask_indices and i not in non_cand_idx:
                cur_sent = doc_tokens.copy()
                amount_masked += 1
                masked_word = doc_tokens[i]
                cur_sent[i] = '[MASK]'
                amount_masked += 1
                predicted_words = unmasker(' '.join(cur_sent), sentiment)
                # print(f"{predicted_words=}")
                if predicted_words[0] == masked_word: # skip to the next predicted word
                    augmented_sentence.append(predicted_words[1])
                    i += 1
                else:
                    augmented_sentence.append(predicted_words[0])
                    i += 1
            elif i in non_cand_idx:
                sub_target = augmented_sentence[tar_idx[j]]
                augmented_sentence.append(sub_target)
                j+=1
                if j >= len(tar_idx): # reset index
                    j = 0
                i+=1

            else:
                augmented_sentence.append(doc_tokens[i])
                i += 1

    # Extract the modified_aspect based on in_target_idx in the new augmented sentence
    modified_target = tar
    modified_target = [augmented_sentence[idx] for idx in tar_idx]
    modified_target_str = tokenizer.convert_tokens_to_string(modified_target)

    start = tar_idx[0]
    end = tar_idx[-1]+1
    augmented_sentence [start:end] = ["$T$"]
    if len(non_cand_idx) > 0:
        start = non_cand_idx[0]
        end = non_cand_idx[-1] + 1
        augmented_sentence[start:end] = ["$T$"]

    # Join the masked tokens to form the masked sequence
    augmented_sentence_str = re.sub(r'\s([,.:;!])', r'\1', tokenizer.convert_tokens_to_string(augmented_sentence))
    if '$T$' not in augmented_sentence_str:
        raise ValueError
    return augmented_sentence_str, modified_target_str

def augment_aspect_adj_adv(in_sentence, in_target, sentiment):
    """
    This function selective substitute all aspect, adjectives and adverbs (15%) occuring in a sentence
    """
    tar = nlp(in_target)
    tar = [token.text for token in tar]
    sentence_w_target = re.sub(r'\$T\$', in_target, in_sentence) # substitute $t$ with actual target

    # Tokenize the sequence using spaCy
    doc = nlp(sentence_w_target)
    doc_tokens = [token.text for token in doc] # list of tokens
    tar_idx = [i for i, token in enumerate(doc_tokens) if any(is_similar_enough(token, t) for t in tar)]

    n = len(doc_tokens)
    m = len(tar)
    non_cand_idx = []
    count_tar = 0 # count occurences of target within a sentence
    for i in range(n - m + 1): # Slide windows to obtain target indices
        if len(doc_tokens[i:i + m]) == len(tar) and count_tar == 0:
            if is_similar_enough(str(doc_tokens[i:i + m]), str(tar)):
                tar_idx = list(range(i, i + m))
                count_tar += 1
        elif len(doc_tokens[i:i + m]) == len(tar) and count_tar > 0:
            if is_similar_enough(str(doc_tokens[i:i + m]), str(tar)):
                non_cand_idx = list(range(i, i + m))
    tar_set = set(tar_idx)
    non_cand_set = set(non_cand_idx)
    # Check if there is any intersection (tar_idx and non_cand_idx should be disjoint!)
    if tar_set & non_cand_set:
        non_cand_idx = []

    j = 0
    number_not_words = 0
    number_adj_adv = 0
    adj_adv_ind = []
    while j < len(doc_tokens):
        if doc[j].pos_ in ['ADJ', 'ADV'] and j not in tar_idx and j not in non_cand_idx:
            adj_adv_ind.append(j)
            j += 1
            number_adj_adv += 1
        elif doc_tokens[j] in string.punctuation:
            j += 1
            number_not_words += 1
        else:
            j += 1


    # Mask tokens tagged as ADJ or ADV
    max_total_mask = 0.15
    mask_indices = []
    if len(adj_adv_ind) >= 1:
        num_to_mask = max(1, int(0.15 * number_adj_adv)) #maximum of 15% of adjectives and adverbs can be masked in the sentence
        mask_indices = rd.sample(adj_adv_ind, num_to_mask)
    # print(f"{num_to_mask=}")

    i = 0
    amount_masked = 0
    augmented_sentence = []
    j = 0
    target = False
    modified_target = tar
    while i < len(doc_tokens):
        if doc_tokens[i] in string.punctuation and i not in tar_idx and i not in non_cand_idx:
            augmented_sentence.append(doc_tokens[i])
            i += 1
        else:
            if (i in mask_indices or i == tar_idx[0]) and i not in non_cand_idx:
                amount_masked += 1
                cur_sent = doc_tokens.copy()
                masked_word = doc_tokens[i]
                if i == tar_idx[0]:
                    target = True
                    cur_sent[i:tar_idx[-1]+1] = ['[MASK]']
                    curr_idx = i
                    i = tar_idx[-1]+1
                    tar_idx = [curr_idx]
                else:
                    cur_sent[i] = '[MASK]'
                    i += 1
                amount_masked += 1
                predicted_words = unmasker(' '.join(cur_sent),sentiment)

                if predicted_words[0] == masked_word: # skip to the next predicted word
                    augmented_sentence.append(predicted_words[1])
                    if target:
                        modified_target = augmented_sentence[-1]
                        target = False
                else:
                    augmented_sentence.append(predicted_words[0])
                    if target:
                        modified_target = augmented_sentence[-1]
                        target = False
            elif i in non_cand_idx:
                augmented_sentence.append(modified_target)
                curr_idx = len(augmented_sentence) - 1
                i = non_cand_idx[-1]+1
                non_cand_idx = [curr_idx]
            else:
                augmented_sentence.append(doc_tokens[i])
                i += 1

    
    start = tar_idx[0]
    end = tar_idx[-1]+1
    augmented_sentence [start:end] = ["$T$"]
    if len(non_cand_idx) > 0:
        start = non_cand_idx[0]
        end = non_cand_idx[-1] + 1
        augmented_sentence[start:end] = ["$T$"]

    # Join the masked tokens to form the masked sequence
    augmented_sentence_str = re.sub(r'\s([,.:;!])', r'\1', tokenizer.convert_tokens_to_string(augmented_sentence))
    # augmented_sentence_str = re.sub(modified_target_str,'$T$', augmented_sentence_str)
    if '$T$' not in augmented_sentence_str:
        raise ValueError
    return augmented_sentence_str, modified_target

def augment_all_noun_adj_adv(in_sentence, in_target, sentiment):
    """
    This function selective substitute all nouns, adjectives and adverbs (15%) occuring in a sentence
    """
    tar = nlp(in_target)
    tar = [token.text for token in tar]
    sentence_w_target = re.sub(r'\$T\$', in_target, in_sentence) # replace $t$ with actual target

    # Tokenize the sequence using spaCy
    doc = nlp(sentence_w_target)
    doc_tokens = [token.text for token in doc] # list of tokens
    #tar_idx = [i for i, token in enumerate(doc_tokens) if any(is_similar_enough(token, t) for t in tar)]
    n = len(doc_tokens)
    m = len(tar)
    non_cand_idx = []
    count_tar = 0 # count occurences of target within a sentence
    for i in range(n - m + 1): # Slide windows to obtain target indices
        if len(doc_tokens[i:i + m]) == len(tar) and count_tar == 0:
            if is_similar_enough(str(doc_tokens[i:i + m]), str(tar)):
                tar_idx = list(range(i, i + m))
                count_tar += 1
        elif len(doc_tokens[i:i + m]) == len(tar) and count_tar > 0:
            if is_similar_enough(str(doc_tokens[i:i + m]), str(tar)):
                non_cand_idx = list(range(i, i + m))
    tar_set = set(tar_idx)
    non_cand_set = set(non_cand_idx)
    # Check if there is any intersection (tar_idx and non_cand_idx should be disjoint!)
    if tar_set & non_cand_set:
        non_cand_idx = []

    noun_idx = []
    j = 0
    number_not_words = 0
    number_nouns = 0
    adj_adv_ind = []
    number_adj_adv = 0
    while j < len(doc_tokens):
        if doc[j].pos_ in ['NOUN','PRON'] and j not in non_cand_idx:
            noun_idx.append(j)
            j += 1
            number_nouns += 1
        elif doc[j].pos_ in ['ADJ', 'ADV'] and j not in non_cand_idx:
            adj_adv_ind.append(j)
            j += 1
            number_adj_adv += 1
        elif doc_tokens[j] in string.punctuation:
            j += 1
            number_not_words += 1
        else:
            j += 1
    

    i = 0
    j = 0
    augmented_sentence = []
    amount_masked = 0
    # cur_sent = doc_tokens.copy()
    mask_indices = []
    if len(adj_adv_ind) >= 1:
        num_to_mask = max(1, int(0.15 * number_adj_adv)) #maximum of 15% of adjectives and adverbs can be masked in the sentence
        mask_indices = rd.sample(adj_adv_ind, num_to_mask)

    while i < len(doc_tokens):
        if doc_tokens[i] in string.punctuation:
            augmented_sentence.append(doc_tokens[i])
            i += 1
        else:
            if (doc[i].pos_ in ['NOUN', 'PRON'] or i in mask_indices) and i not in non_cand_idx :
                amount_masked += 1
                cur_sent = doc_tokens.copy()
                masked_word = doc_tokens[i]
                cur_sent[i] = '[MASK]'
                predicted_words = unmasker(' '.join(cur_sent),sentiment)
                if predicted_words[0] == masked_word: # skip to the next predicted word
                    augmented_sentence.append(predicted_words[1])
                    i += 1
                else:
                    augmented_sentence.append(predicted_words[0])
                    cur_sent[i] = predicted_words[0]
                    i += 1
            elif i in non_cand_idx:
                sub_target = augmented_sentence[tar_idx[j]]
                augmented_sentence.append(sub_target)
                j+=1
                if j >= len(tar_idx): # reset index
                    j = 0
                i+=1
            else:
                augmented_sentence.append(doc_tokens[i])
                i += 1

    # Extract the modified_aspect based on in_target_idx in the new augmented sentence
    modified_target = tar
    modified_target = [augmented_sentence[idx] for idx in tar_idx]
    modified_target_str = tokenizer.convert_tokens_to_string(modified_target)

    start = tar_idx[0]
    end = tar_idx[-1]+1
    augmented_sentence [start:end] = ["$T$"]
    if len(non_cand_idx) > 0:
        start = non_cand_idx[0]
        end = non_cand_idx[-1] + 1
        augmented_sentence[start:end] = ["$T$"]

    # Join the masked tokens to form the masked sequence
    augmented_sentence_str = re.sub(r'\s([,.:;!])', r'\1', tokenizer.convert_tokens_to_string(augmented_sentence))
    # augmented_sentence_str = re.sub(modified_target_str,'$T$', augmented_sentence_str)
    if '$T$' not in augmented_sentence_str:
        raise ValueError
    return augmented_sentence_str, modified_target_str


if __name__ == '__main__':
    
    in_sentence = "The $T$ is too dirty, but the $T$ compensates it all."
    in_target = "mens bathroom"
    aug, aspect = augment_sentence_aspect(in_sentence, in_target, "-1")
    print(aug)
    print(aspect)


    # in_sentence = "The $T$ is too dirty, but the salmon compensates it all."
    # in_target = "mens bathroom"
    # aug, aspect = augment_sentence_aspect(in_sentence, in_target, "-1")
    # print(aug)
    # print(aspect)

