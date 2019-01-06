from csvParser import parseTrainData, parseTestData
from wordsManager import createNgramsDictionary

trainData = parseTrainData(1306120)
testData = parseTestData(56370)

# print('********************\nTRAIN:\n********************')
# for q in trainData:
#     q.print()

# print('********************\nTEST:\n********************')
# for q in testData:
#     q.print()

insincereDictionary, sincereDictionary = createNgramsDictionary(trainData, 2)
i = 0
for w in sorted(insincereDictionary.items(), key=lambda x: x[1], reverse=True):
    if i > 300:
        break
    i += 1
    print(w)