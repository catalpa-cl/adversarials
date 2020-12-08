import nltk
from numpy.random import choice,seed


def get_text(grams, frequency, avg_length):
    start = choice(grams)
    current_sentence = start
    current_length = len(start)
    while current_length < avg_length:
        next_gram = choice(a=grams, size=1, p=frequency)[0]
        if len(next_gram)==1:
            continue
        current_sentence = (current_sentence + ' ' + next_gram).strip()
        current_length = len(current_sentence)
    return current_sentence


for i in range(1, 2):
    with open('data/asap/preprocessed_word/asap_prompt_' + str(i) + '_preprocessed.txt', 'r', encoding='utf-8') as in_file:
        sentences = in_file.read().split('\n')
        tagged_list = []
        for s in sentences:
            tokenized = nltk.word_tokenize(s)
            tagged = nltk.pos_tag(tokenized)
            tagged_list += tagged
        print(tagged_list)
        is_noun = lambda pos: pos[:2] == 'NN'
        noun_list = [word for (word, pos) in tagged_list if is_noun(pos)]
        fd = nltk.FreqDist(noun_list)
        nouns = []
        frequency= []
        for word, count in fd.items():
            nouns.append(word)
            frequency.append(count/sum(fd.values()))
        '''
            if word == 'panda':
                print('panda=' + str(count / sum(fd.values())))
            elif word == 'china':
                print('china=' + str(count / sum(fd.values())))
            elif word == 'koala':
                print('koala=' + str(count / sum(fd.values())))
            elif word == 'australia':
                print('australia=' + str(count / sum(fd.values())))
        '''
        index = 0
        with open('augment/p'+str(i)+'/aug_content_burst_1000.txt', 'w', encoding='utf-8') as file:
            print(
                'Generating 1000 content burst adversarial for prompt ' + str(i))
            file.write('Id\tEssaySet\tessay_score\tessay_score\tEssayText\n')
            seed(10)
            for j in range(1000):
                sentence = get_text(nouns, frequency, 44)
                file.write('10700'+'{0:03}'.format(index) + '\t' + str(i) + '\t' + '0\t0\t' + sentence + '\n')
                index += 1
        file.close()


