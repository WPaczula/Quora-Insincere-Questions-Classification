class Question:
    def __init__(self, id, text, target=None):
        self.id = id
        self.text = text
        self.target = target

    def print(self):
        if self.target is None:
            target = '?' 
        else: 
            target = self.target

        print('id: ' + self.id + ' ' +
             'text: ' + self.text + ' ' +
             'target: ' + target + ' ')
