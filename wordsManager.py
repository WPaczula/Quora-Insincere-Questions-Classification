import string
from collections import defaultdict
from nltk import ngrams
from nltk.corpus import stopwords


def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def remove_stopwords(text):
    sw = stopwords.words('english')
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    # return " ".join(text) # if you want whole sentences
    return text


def apply_removals(questions):
    cleared_texts = []
    for q in questions:
        cleared_text = remove_punctuation(q[1])
        cleared_text = remove_stopwords(cleared_text)
        cleared_texts.append(cleared_text)
    return cleared_texts


def create_ngrams_dictionary(questions, n=1):
    insincere_dictionary = defaultdict(int)
    sincere_dictionary = defaultdict(int)
    insincere_questions = []
    sincere_questions = []
    for q in questions:
        if q[2] == '0':
            sincere_questions.append(q)
        elif q[2] == '1':
            insincere_questions.append(q)
    insincere_questions = apply_removals(insincere_questions)
    for i in insincere_questions:
        for w in ngrams(i, n):
            insincere_dictionary[" ".join(w)] += 1
    sincere_questions = apply_removals(sincere_questions)
    for s in sincere_questions:
        for w in ngrams(s, n):
            sincere_dictionary[" ".join(w)] += 1
    return insincere_dictionary, sincere_dictionary
