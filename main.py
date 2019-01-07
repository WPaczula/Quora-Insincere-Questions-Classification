import pandas as pd

from sklearn import model_selection, tree, naive_bayes

from csvParser import parse_train_data, parse_test_data
from wordsManager import create_ngrams_dictionary, apply_removals, create_ngram_set
from sklearn.feature_extraction.text import TfidfVectorizer

train_data = parse_train_data(13061)
test_data = parse_test_data(5637)

# print('********************\nTRAIN:\n********************')
# for q in trainData:
#     q.print()

# print('********************\nTEST:\n********************')
# for q in testData:
#     q.print()

insincere_dictionary, sincere_dictionary = create_ngrams_dictionary(train_data, 2)

test_dictionary = create_ngram_set(test_data, 2)
# i = 0
# for w in sorted(insincere_dictionary.items(), key=lambda x: x[1], reverse=True):
#     if i > 300:
#         break
#     i += 1
#     print(w)

train_df = pd.DataFrame(columns=["ngram_text", "multiplicity", "target"])

for key, value in sorted(insincere_dictionary.items(), key=lambda x: x[1], reverse=True):
    if value <= 1:
        break
    train_df.append({"ngram_text": key,
                     "multiplicity": value,
                     "target": 1},
                    ignore_index=True)

for key, value in sorted(sincere_dictionary.items(), key=lambda x: x[1], reverse=True):
    if value <= 1:
        break
    train_df.append({"ngram_text": key,
                     "multiplicity": value,
                     "target": 0},
                    ignore_index=True)

# train_df = pd.DataFrame(train_data, columns=["qid", "question_text", "target"])
test_df = pd.DataFrame(test_data, columns=["qid", "question_text"])

train_df["question_text"] = apply_removals(train_data)
test_df["question_text"] = apply_removals(test_data)

train_data = []
test_data = []

tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(2, 3))
tfidf_vec.fit(train_df["question_text"].values.tolist() + test_df["question_text"].values.tolist())
train_tfidf = tfidf_vec.transform(train_df["question_text"]).todense()

df = pd.DataFrame(train_tfidf)

features = df.columns.tolist()


classifier = naive_bayes.GaussianNB()
classifier.fit(train_df[features], train_data[2])

test_tfidf = tfidf_vec.transform(test).todense()

results = classifier.predict(test_tfidf)

print("sss")
