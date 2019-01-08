import pandas as pd

from sklearn import model_selection, tree, naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from csvParser import parse_train_data, parse_test_data
from wordsManager import create_ngrams_dictionary, apply_removals, create_ngram_set
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

train_data = parse_train_data(130612)
test_data = parse_test_data(5637)

submission_path = r"/Users/mateuszligus/projects/studia/Quora-Insincere-Questions-Classification/data/submission.csv"

# print('********************\nTRAIN:\n********************')
# for q in trainData:
#     q.print()

# print('********************\nTEST:\n********************')
# for q in testData:
#     q.print()


# i = 0
# for w in sorted(insincere_dictionary.items(), key=lambda x: x[1], reverse=True):
#     if i > 300:
#         break
#     i += 1
#     print(w)

train_df = pd.DataFrame(columns=["sentence",  "target"])

test_df = pd.DataFrame(columns=["qid", "sentence", "shortened", "prediction"])

train_df["sentence"] = apply_removals(train_data)
train_df["target"] = [x[2] for x in train_data]
train_df.head()

tfidf = TfidfVectorizer(ngram_range=(2, 3), stop_words="english")

features = tfidf.fit_transform(train_df.sentence).toarray()
labels = train_df.target.items()
print(features.shape) # produces touple (N, M) - each of N sentences is represented by M features (representing the tf-idf score for different bigrams and trigrams if ngram_range(2,3)

X_train, X_test, y_train, y_test = train_test_split(train_df["sentence"], train_df["target"], random_state=0)

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)

tfdif_transformer = TfidfTransformer()

X_train_tfdif = tfdif_transformer.fit_transform(X_train_counts)

classificator = MultinomialNB().fit(X_train_tfdif, y_train)

test_df["qid"] = [x[0] for x in test_data]
test_df["sentence"] = [x[1] for x in test_data]
test_df["shortened"] = apply_removals(test_data)

i = 0
for string in test_df["shortened"]:
    predicted = classificator.predict(count_vect.transform([string]))
    test_df.set_value(i, "prediction", predicted[0])
    i += 1

headers = ["qid", "prediction"]
test_df.to_csv(submission_path, columns=headers, index=False)
print("sss")
