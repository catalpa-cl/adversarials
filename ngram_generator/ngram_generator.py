from ngram_generator.preprocess import preprocess
from ngram_generator.language_model import *
import pathlib
import nltk
from numpy.random import choice,seed


def get_text(level, grams, frequency, max_length):
    start = choice(a=grams, size=1, p=frequency)[0]
    while '</s>' in start:
        start = choice(a=grams, size=1, p=frequency)[0]
    current_sentence = start
    current_length = len(start)
    while current_length < max_length:
        next_gram = choice(a=grams, size=1, p=frequency)[0]
        current_sentence = (current_sentence+' '+next_gram).strip()
        if '</s>' in next_gram:
            if(current_length<5):
                continue
            else:
                end = current_sentence.find('</s>')
                current_sentence = current_sentence[0:end]
            if level == 'char':
                space_remove = current_sentence.replace(' ', '')
                current_sentence = space_remove.replace('ß', ' ')
            return current_sentence
        current_length = len(current_sentence)
    if level == 'char':
        space_remove = current_sentence.replace(' ', '')
        current_sentence = space_remove.replace('ß', ' ')
    return current_sentence


max_length = {'1': 776, '2': 918, '3': 742, '4': 631, '5': 1478, '6': 1032, '7': 1103, '8': 1651, '9': 1820, '10': 1188}


def generarate_adversarial(item_nr, level, min_n, max_n, corpus_name, input_path, output_path, max_length):
    index = 0
    path = pathlib.Path(input_path)
    train, test = load_data(path)
    for n in range(min_n, max_n+1):
        tokens = preprocess(train, n)
        n_grams = nltk.ngrams(tokens, n)
        n_vocab = nltk.FreqDist(n_grams)
        grams = []
        frequency = []
        for gram, count in n_vocab.items():
            gram_text = ''
            for t in range(0, n):
                gram_text = (gram_text + ' ' + gram[t]).strip()
                if len(gram_text) == 0:
                    continue
            grams.append(gram_text)
            frequency.append(count / sum(n_vocab.values()))
        '''
            # get certain frequency
            if gram_text == 'panda':
                print('panda='+str(count / sum(n_vocab.values())))
            elif gram_text == 'koala':
                print('koala='+str(count / sum(n_vocab.values())))
            elif gram_text == 'cat':
                print('cat='+str(count / sum(n_vocab.values())))
        '''
        with open(output_path+level[0] + str(n) + '_1000.txt', 'w', encoding='utf-8') as file:
            print(
                'Generating 1000 ' +level + ' level '+ str(n) + '-gram adversarial for ' + corpus_name + ' with max length ' + str(max_length))
            file.write('Id\tEssaySet\tessay_score\tessay_score\tEssayText\n')
            seed(10)
            pre = '1060'
            if level == 'char':
                max_length = max_length*2
                pre = '1050'
            for j in range(item_nr):
                sentence = get_text(level, grams, frequency, max_length)
                file.write(pre+str(n)+'{0:03}'.format(index) + '\t' + '0' + '\t' + '0\t0\t' + sentence + '\n')
                index += 1
        file.close()


#generarate_adversarial(1000, 'word', 1, 5, 'brown', 'data/brown_preprocessed.txt', 'augment/aug_brown_', max_length.get('5'))
#generarate_adversarial(1000, 'char', 1, 5, 'brown', 'data/character_split_brown.txt', 'augment/aug_brown', max_length.get('5'))
#generarate_adversarial(1000, 'char', 1, 5, 'asap', 'data/asap/preprocessed_char/character_split_asap_prompt_3.txt', 'augment/p3/aug_asap_', max_length.get('3'))
#generarate_adversarial(1000, 'word', 1, 5, 'asap', 'data/asap/preprocessed_word/asap_prompt_3_preprocessed.txt', 'augment/p3/aug_asap_', max_length.get('3'))
