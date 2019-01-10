import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import string
import csv
import nltk
from collections import defaultdict
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

# Any results you write to the current directory are saved as output.

nltk.download('stopwords')

trainDataPath = r"input/train.csv"
testDataPath = r"input/test.csv"


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
            data = list(reader)

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


train_data = parse_train_data(13061)#22)
test_data = parse_test_data(56370)

submission_path = r"submission.csv"

train_df = pd.DataFrame(columns=["sentence", "target"])

test_df = pd.DataFrame(columns=["qid", "sentence", "shortened", "prediction"])

train_df["sentence"] = apply_removals(train_data)
train_df["target"] = [x[2] for x in train_data]
train_df.head()
train_data = []

# tfidf = TfidfVectorizer(ngram_range=(1, 3), stop_words="english", min_df=0.0005)

# features = tfidf.fit_transform(train_df.sentence).toarray()
# labels = train_df.target.items()
# print(features.shape) # produces touple (N, M) - each of N sentences is represented by M features (representing the tf-idf score for different bigrams and trigrams if ngram_range(2,3)

# X_train, X_test, y_train, y_test = train_test_split(train_df["sentence"], train_df["target"], random_state=0)

count_vect = CountVectorizer(ngram_range=(1, 1))

X_train_counts = count_vect.fit_transform(train_df["sentence"])

tfdif_transformer = TfidfTransformer()

X_train_tfdif = tfdif_transformer.fit_transform(X_train_counts)

classificator = MultinomialNB().fit(X_train_tfdif, train_df["target"])

test_df["qid"] = [x[0] for x in test_data]
test_df["sentence"] = [x[1] for x in test_data]
test_df["shortened"] = apply_removals(test_data)
test_data = []

i = 0
for string in test_df["shortened"]:
    predicted = classificator.predict(count_vect.transform([string]))
    test_df.set_value(i, "prediction", predicted[0])
    i += 1

headers = ["qid", "prediction"]
test_df.to_csv(submission_path, columns=headers, index=False)