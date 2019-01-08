import csv

trainDataPath = r"/Users/mateuszligus/projects/studia/Quora-Insincere-Questions-Classification/data/train.csv"
testDataPath = r"/Users/mateuszligus/projects/studia/Quora-Insincere-Questions-Classification/data/test.csv"


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
