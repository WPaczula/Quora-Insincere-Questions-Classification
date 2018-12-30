from csvParser import parseTrainData, parseTestData

trainData = parseTrainData(10)
testData = parseTestData(10)

print('********************\nTRAIN:\n********************')
for q in trainData:
    q.print()

print('********************\nTEST:\n********************')
for q in testData:
    q.print()