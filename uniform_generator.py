from nltk.corpus import brown
from numpy.random import choice,seed


# word list from brown corpus
wordlist = list(map(lambda x:x.lower(),brown.words()))
wordlist = list(dict.fromkeys(wordlist))

# character list
charlist = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']


def get_text(grams, avg_length):
    start = choice(grams)
    current_sentence = start
    current_length = len(start)
    while current_length < avg_length:
        next_gram = choice(grams)
        current_sentence = current_sentence+''+next_gram
        current_length = len(current_sentence)
    return current_sentence


index = 0
with open('augment/aug_random_char_1000.txt', 'w', encoding='utf-8') as file:
    file.write('Id\tEssaySet\tessay_score\tessay_score\tEssayText\n')
    seed(10)
    for j in range(1000):
        sentence = get_text(charlist, 44)
        file.write('10100'+'{0:03}'.format(index) + '\t' + '0' + '\t' + '0\t0\t' + sentence + '\n')
        index += 1
file.close()
