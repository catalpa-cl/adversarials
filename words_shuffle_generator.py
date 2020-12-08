from nltk.tokenize import RegexpTokenizer
import random
import pandas as pd
import csv
from numpy.random import seed


# load original answers
for i in range(1, 10):
    file_path = 'data/asap/train+test/asap_withoutBG_prompt_' + str(i) + '.txt'
    text_df = pd.read_csv(file_path, encoding='utf-8', sep='\t', header=0,
                          quoting=csv.QUOTE_NONE,
                          names=['Id', 'EssayText'],
                          dtype={'Id': str, 'EssayText': str})
    input_df = text_df.sample(n=1000)
    # generation
    tokenizer = RegexpTokenizer(r'\w+')
    seed(10)
    with open('augment/p'+str(i)+'/aug_shuffle_1000.txt', 'w', encoding='utf-8') as file:
        file.write('Id\tEssaySet\tessay_score\tessay_score\tEssayText\n')
        # iterate each sentence and shuffle the words
        for index, row in input_df.iterrows():
            doc = row['EssayText']
            tokens_in_doc = tokenizer.tokenize(doc.lower())
            # print(tokens_in_doc)
            shuffled_tokens = random.sample(tokens_in_doc, len(tokens_in_doc))
            # print(shuffled_tokens)
            shuffled_doc = " ".join(shuffled_tokens)
            file.write('10800'+row['Id'] + '\t' + str(i) + '\t0\t0\t' + shuffled_doc + '\n')
        file.close()
