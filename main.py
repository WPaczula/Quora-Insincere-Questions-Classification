import csv
import string

# import nltk
import operator
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# nltk.download('stopwords')

# Data and output paths:
trainDataPath = r"../input/train.csv"
testDataPath = r"../input/test.csv"
submission_path = r"submission.csv"

# Defining functions to read csv:
def read_csv_data(file, max_rows=None):
    with open(file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')

        if max_rows is not None:
            header = True
            i = 0
            data = []

            for row in reader:
                # skip header
                if header:
                    header = False
                    continue

                data.append(row)
                i += 1
                if i == max_rows:
                    break
        else:
            data = []
            header = True
            for row in reader:
                # skip header
                if header:
                    header = False
                    continue
                data.append(row)

    return data


def parse_train_data(max_rows=None):
    data = read_csv_data(trainDataPath, max_rows)

    questions = []
    for row in data:
        questions.append((row[0], row[1], row[2]))

    return questions


def parse_test_data(max_rows=None):
    data = read_csv_data(testDataPath, max_rows)

    questions = []
    for row in data:
        questions.append((row[0], row[1]))

    return questions

# Defining functions for removing punctuation and stopwords:
# unused in current version
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def remove_stopwords(text):
    sw = stopwords.words('english')
    text = [PorterStemmer().stem(word.lower()) for word in text.split() if word.lower() not in sw]
    return " ".join(text)  # if you want whole sentences
    # return text


def apply_removals(questions):
    cleared_texts = []
    for q in questions:
        cleared_text = remove_punctuation(q[1])
        cleared_text = remove_stopwords(cleared_text)
        cleared_texts.append(cleared_text)
    return cleared_texts

# Function that builds vocabulary from words:
def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

# Function that checks how many words from questions are in embeddings:
def check_coverage(vocab, embeddings_index):
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.3%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.3%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words

# Function adding lowered words to embedding:
def add_lower(embedding, vocab):
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:
            embedding[word.lower()] = embedding[word]
            count += 1
    print(f"Added {count} words to embedding")

# Function clearing texts from contractions:
def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text

# Function for cleaning special characters:
def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])

    for p in punct:
        text = text.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '',
                'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])

    return text

# Function for spelling correction:
def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x

# Function for statistics calculation:
def run_model(model, name):
    classificator = model.fit(x_train_tfidf, y_train)
    predictions = classificator.predict(x_test_tfidf)
    confusionMatrix = confusion_matrix(y_test.values, predictions)
    total = sum(sum(confusionMatrix))
    specificity = confusionMatrix[1, 1] / (confusionMatrix[1, 0] + confusionMatrix[1, 1])
    print(name)
    print("Confusion matrix: ")
    print(confusionMatrix)
    print("Accuracy: ", metrics.accuracy_score(y_test.values, predictions))
    print('Specificity: ', specificity)
    print("Precision: ", metrics.precision_score(y_test.values, predictions, pos_label='1'))
    print("Recall/Sensitivity: ", metrics.recall_score(y_test.values, predictions, pos_label='1'))
    print("F1 Score: ", metrics.f1_score(y_test.values, predictions, pos_label='1'))

# List of most known contractions:
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

# Most common misspells:
mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization', 'pokémon': 'pokemon'}

# Punctuations:
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

# Special characters:
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

# Read data:
train_data = parse_train_data()
test_data = parse_test_data()

# Parse data to DataFrame:
train_df = pd.DataFrame(columns=["qid", "sentence", "target"])
test_df = pd.DataFrame(columns=["qid", "sentence", "shortened", "prediction"])

train_df["qid"] = [x[0] for x in train_data]
train_df["sentence"] = [x[1] for x in train_data]
train_df["target"] = [x[2] for x in train_data]

test_df["qid"] = [x[0] for x in test_data]
test_df["sentence"] = [x[1] for x in test_data]

# Read GloVe embedding:
glove = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr):
        return word, np.asarray(arr, dtype='float32')
glove_embed = dict(get_coefs(*o.split(" ")) for o in open(glove, encoding='latin'))

# Now we check amount of our vocabulary in embedding:
vocab = build_vocab(train_df['sentence'])
print("Glove: ")
oov_glove = check_coverage(vocab, glove_embed)

# Change words to lower case:
train_df["lowered"] = train_df["sentence"].apply(lambda x: x.lower())
test_df["lowered"] = test_df["sentence"].apply(lambda x: x.lower())

# Check embedding covering:
print("Glove: ")
oov_glove = check_coverage(vocab, glove_embed)
add_lower(glove_embed, vocab)
oov_glove = check_coverage(vocab, glove_embed)

# Clean contractions:
train_df['treated_question'] = train_df['lowered'].apply(lambda x: clean_contractions(x, contraction_mapping))

# Check covering after contractions cleaning:
vocab = build_vocab(train_df['treated_question'])
print("Glove : ")
oov_glove = check_coverage(vocab, glove_embed)

# Clean from special characters:
train_df['treated_question'] = train_df['treated_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

# Check covering after cleaning from special characters:
vocab = build_vocab(train_df['treated_question'])
print("Glove : ")
oov_glove = check_coverage(vocab, glove_embed)

# Replace misspelled words:
train_df['treated_question'] = train_df['treated_question'].apply(lambda x: correct_spelling(x, mispell_dict))

# Check coverage:
vocab = build_vocab(train_df['treated_question'])
print("Glove : ")
oov_glove = check_coverage(vocab, glove_embed)

# Clean test set:
test_df['treated_question'] = test_df['sentence'].apply(lambda x: x.lower())
test_df['treated_question'] = test_df['treated_question'].apply(lambda x: clean_contractions(x, contraction_mapping))
test_df['treated_question'] = test_df['treated_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
test_df['treated_question'] = test_df['treated_question'].apply(lambda x: correct_spelling(x, mispell_dict))

# Classification with ngram_range(1, 3) and TFIDF:
count_vect = CountVectorizer(ngram_range=(1, 3))
tfidf_transformer = TfidfTransformer()

#Test values for statistics
X_train, X_test, y_train, y_test = train_test_split(train_df["treated_question"], train_df["target"], random_state=0)

x_train_counts = count_vect.fit_transform(X_train)
x_test_counts = count_vect.transform(X_test)
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
x_test_tfidf = tfidf_transformer.transform(x_test_counts)

X_train_counts = count_vect.fit_transform(train_df["treated_question"])
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
model = LogisticRegression(solver='lbfgs', max_iter=300)
run_model(model, "LogisticRegression")

model = LogisticRegression(solver='lbfgs', max_iter=300)
classificator = model.fit(X_train_tfidf, train_df["target"])

test_df["prediction"] = classificator.predict(tfidf_transformer.transform(count_vect.transform(test_df["treated_question"])))

headers = ["qid", "prediction"]
test_df.to_csv(submission_path, columns=headers, index=False)
