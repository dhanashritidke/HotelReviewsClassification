from collections import Counter
import sys

class NaiveBayesClassifier:
    def __init__(self, file1):
        self.tokens = {}
        self.totalWords = 0
        self.modelParametersFile = "nbmodel.txt"
        self.dataFile = file1
        self.outputFileName = "nboutput.txt"
        self.noOfClasses = 4
        self.classes = ['Fake', 'True', 'Pos', 'Neg']
        self.priors = {}

    def removePunctuation(self, line):
        punctuations = ['.', ',', '"', ';', '/', '!', "'s", '$', '-']
        for p in punctuations:
            if p == '$':
                line = line.replace(p, "$ ")
            elif p == '-' or p == '/':
                line = line.replace(p, " ")
            else:
                line = line.replace(p, "")
        return line

    def readModelParameters(self):
        txt_file = open(self.modelParametersFile)
        self.total_words = int(txt_file.readline().strip())
        for _ in range(self.noOfClasses):
            line = txt_file.readline().strip().split(" ")
            self.priors[line[0]] = float(line[1])
        for _ in range(self.total_words):
            line = txt_file.readline().strip().split(" ")
            self.tokens[line[0]] = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]

    def classify(self):
        txt_file = open(self.dataFile, "r")
        outputFile = open(self.outputFileName, "w")
        for line in txt_file:
            line = line.strip()
            review_id = line.split(' ', 1)[0]
            reviewString = line.split(' ', 1)[1]
            review = Counter(self.removePunctuation(reviewString.lower()).split(" "))
            probFake = self.priors[self.classes[0]]
            probTrue = self.priors[self.classes[1]]
            probPos = self.priors[self.classes[2]]
            probNeg = self.priors[self.classes[3]]
            for word, value in review.items():
                if word in self.tokens:
                    probFake += self.tokens[word][0] * value
                    probTrue += self.tokens[word][1] * value
                    probPos += self.tokens[word][2] * value
                    probNeg += self.tokens[word][3] * value
            classification1 = None
            classification2 = None
            if probTrue > probFake:
                classification1 = self.classes[1]
            else:
                classification1 = self.classes[0]
            if probPos > probNeg:
                classification2 = self.classes[2]
            else:
                classification2 = self.classes[3]
            outputFile.write("%s %s %s\n" % (review_id, classification1, classification2))

def nb_classifier():
    #file_name = "./data/dev-text.txt"
    nbc = NaiveBayesClassifier(sys.argv[1])
    nbc.readModelParameters()
    nbc.classify()

nb_classifier()
