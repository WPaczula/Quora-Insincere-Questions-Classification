import string
from collections import defaultdict
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def removePunctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def removeStopwords(text):
    sw = stopwords.words('english')
    text = [PorterStemmer().stem(word.lower()) for word in text.split() if word.lower() not in sw]
    # return " ".join(text) # if you want whole sentences
    return text

def applyRemovals(questions):
    clearedTexts = []
    for q in questions:
        clearedText = removePunctuation(q[1])
        clearedText = removeStopwords(clearedText)
        clearedTexts.append(clearedText)
    return clearedTexts

def createNgramsDictionary(questions, n = 1):
    insincereDictionary = defaultdict(int)
    sincereDictionary = defaultdict(int)
    insincereQuestions = []
    sincereQuestions = []
    for q in questions:
        if q[2] == '0':
            sincereQuestions.append(q)
        elif q[2] == '1':
            insincereQuestions.append(q)
    insincereQuestions = applyRemovals(insincereQuestions)
    for i in insincereQuestions:
        for w in ngrams(i, n):
            insincereDictionary[" ".join(w)] += 1
    sincereQuestions = applyRemovals(sincereQuestions)
    for s in sincereQuestions:
        for w in ngrams(s, n):
            sincereDictionary[" ".join(w)] += 1
    return insincereDictionary, sincereDictionary
