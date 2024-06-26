'''
This Program makes the BERT embedding matrix and test-/traindata, using the tokenisation for BERT
First 'getBERTusingColab' should be used to compile the subfiles containing embeddings.
The new test-/traindata files contain original data, with every word unique and corresponding to vector in emb_matrix
'''
from config import *
from transformers import BertTokenizer
from tqdm import tqdm

def tokenize_sentence(sentence, tokenizer, word_counts):
    words = "[CLS] " + sentence + " [SEP]"
    tokenized_words = tokenizer.tokenize(words)
    tokens = []
    targetbool = True
    for word in tokenized_words[1:-1]:
      if word == '$':
        if targetbool:
          tokens.append("$T$")
          targetbool = False
      elif word == 't':
        pass
      else:
        count = word_counts.get(word, -1) + 1
        word_counts[word] = count
        token = f'{word}_{count}'
        tokens.append(token)
    return ' '.join(tokens)


#def main():
    '''
    Adds BERT embedding values to sentences in the original test and train datasets. Then
    saves these as separate test and train files, which can be used as an input for a classification
    algorithm.
    '''

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
word_counts = {}

with open(f'data/programGeneratedData/temp/unique{FLAGS.year}_BERT_{FLAGS.da_type}_Data_All.txt', 'w') as output_f:
    lines = open(FLAGS.complete_data_file, errors='replace').readlines()

    for i in range(0, len(lines) - 1, 3):
        sentence = lines[i].strip()
        target = lines[i + 1].strip()
        sentiment = lines[i + 2].strip()

        # Tokenize sentence and target
        tokenized_sentence = tokenize_sentence(sentence, tokenizer, word_counts)
        tokenized_target = tokenize_sentence(target, tokenizer, word_counts)

        # Write to output file
        output_f.write(f'{tokenized_sentence}\n')
        output_f.write(f'{tokenized_target}\n')
        output_f.write(f'{sentiment}\n')
    print('Text processing complete. Saved to temporary file')

linesAllData = open(f'data/programGeneratedData/temp/unique{FLAGS.year}_BERT_{FLAGS.da_type}_Data_All.txt').readlines()
linesTrainData = len(open(f'{FLAGS.train_path_ont}').readlines()) if FLAGS.da_type == "none" else len(open(f'{FLAGS.train_path_ont}').readlines()) * 2
print(len(linesAllData))
print(f"{linesTrainData=}")
with open(FLAGS.train_path,'w') as outTrain, \
        open(FLAGS.test_path,'w') as outTest:
    # 2015: 3837 for no augmentation, 7674 BERT-models, 15336 EDA-adjusted, 19185 EDA-original
    # 2016: 5640 for no augmentation, 11280 BERT-models, 22560 EDA-adjusted, 28200 EDA-original
    for j in tqdm(range(0, linesTrainData), desc=f"write train data {outTrain}", unit="sentence"):
        outTrain.write(linesAllData[j])
    for k in tqdm(range(linesTrainData, len(linesAllData)), desc=f"write test data {outTest}", unit="sentence"):
        outTest.write(linesAllData[k])
print('Wrote embedding data to train and test files')