from csvParser import parse_train_data, parse_test_data
from wordsManager import create_ngrams_dictionary

train_data = parse_train_data(1306120)
test_data = parse_test_data(56370)

# print('********************\nTRAIN:\n********************')
# for q in trainData:
#     q.print()

# print('********************\nTEST:\n********************')
# for q in testData:
#     q.print()

insincere_dictionary, sincere_dictionary = create_ngrams_dictionary(train_data, 2)
i = 0
for w in sorted(insincere_dictionary.items(), key=lambda x: x[1], reverse=True):
    if i > 300:
        break
    i += 1
    print(w)
