import csv

trainDataPath = r"./data/train.csv"
testDataPath = r"./data/test.csv"

def readCsvData(file, maxRows = None):
    with open(file, 'r', encoding = 'utf-8') as f:
        reader = csv.reader(f, delimiter = ',')
        
        if maxRows is not None:
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
                if i == maxRows:
                    break
        else:
            data = list(reader)

    return data

def parseTrainData(maxRows = None):
    data = readCsvData(trainDataPath, maxRows)

    questions = []
    for row in data:
        questions.append((row[0], row[1], row[2]))

    return questions

def parseTestData(maxRows = None):
    data = readCsvData(testDataPath, maxRows)

    questions = []
    for row in data:
        questions.append((row[0], row[1]))

    return questions
