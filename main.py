import csv
import string

import nltk
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

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


def apply_removals(questions):
    cleared_texts = []
    for q in questions:
        cleared_text = remove_punctuation(q[1])
        cleared_text = remove_stopwords(cleared_text)
        cleared_texts.append(cleared_text)
    return cleared_texts


train_data = parse_train_data(1306122)
test_data = parse_test_data(56370)

submission_path = r"submission.csv"

train_df = pd.DataFrame(columns=["sentence", "target"])

test_df = pd.DataFrame(columns=["qid", "sentence", "shortened", "prediction"])

train_df["sentence"] = apply_removals(train_data)
train_df["target"] = [x[2] for x in train_data]
train_df.head()
train_data = []

X_train, X_test, y_train, y_test = train_test_split(train_df["sentence"], train_df["target"], random_state=0,
                                                    stratify=train_df["target"])
count_vect = CountVectorizer(ngram_range=(1, 1))

X_train_counts = count_vect.fit_transform(X_train)
X_test_counts = count_vect.transform(X_test)

tfdif_transformer = TfidfTransformer()

X_train_tfdif = tfdif_transformer.fit_transform(X_train_counts)
X_test_tfdif = tfdif_transformer.transform(X_test_counts)

classificator = MultinomialNB().fit(X_train_tfdif, y_train)

predictions = classificator.predict(X_test_tfdif)
confusionMatrix = confusion_matrix(y_test, predictions)

total = sum(sum(confusionMatrix))

accuracy1 = (confusionMatrix[0, 0] + confusionMatrix[1, 1]) / total
print('Accuracy : ', accuracy1)
print(confusionMatrix)
sensitivity1 = confusionMatrix[0, 0] / (confusionMatrix[0, 0] + confusionMatrix[0, 1])
print('Sensitivity : ', sensitivity1)

specificity1 = confusionMatrix[1, 1] / (confusionMatrix[1, 0] + confusionMatrix[1, 1])
print('Specificity : ', specificity1)

print("Accuracy: ", metrics.accuracy_score(y_test, predictions))

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
