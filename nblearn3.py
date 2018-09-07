from read import read
from collections import Counter
import math
import sys

class NaiveBayesLearn:
    def __init__(self, trainModelFile):
        self.trainModelFile=trainModelFile
        self.classes=['Pos','Neg','Fake','True']
        self.trainClassification = {}
        self.tokens = {}
        self.priors = {}
        self.totalTokens = 0
        self.totalPositiveTokens = 0
        self.totalNegativeTokens = 0
        self.totalTrueTokens = 0
        self.totalFakeTokens = 0
        self.totalWords = 0
        self.outputFileName = "nbmodel.txt"

    def readTrainingData(self):
        self.trainClassification=read(self.trainModelFile)

    def removePunctuation(self, line):
        line = line.replace("...", " ")
        punctuations = ['.', ',', '"', ';', '/', '!', "'s", '$', '-', '?', ':', '(', ')', '\n']
        for p in punctuations:
            if p == '$':
                line = line.replace(p, "$ ")
            elif p == '-' or p == '/' or p == '.':
                line = line.replace(p, " ")
            else:
                line = line.replace(p, "")
        line = line.replace("  ", " ")
        #line = line.replace("  ", " ")
        return line

    def prepareCountsForModel(self):
        stopWords = set(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it',
                     'its', 'on', 'of', 'that', 'the', 'to', 'was', 'were', 'will', 'with', 'this'])
        for entry in self.trainClassification:
            for key,value in entry.items():
                review_id=key
                no_punctuations = self.removePunctuation(value[2].lower().strip()).split(" ")
                tokens = filter(lambda x: x not in stopWords, no_punctuations)
                review= Counter(tokens)
                if value[1] == self.classes[0]:
                    for word in review:
                        if word in stopWords:
                            continue
                        if word != " " and word != '':
                            if word in self.tokens:
                                self.tokens[word][2][0] += review[word]
                            else:
                                self.tokens[word] = [[0, 0], [0, 0], [review[word], 0], [0, 0]]
                            self.totalPositiveTokens += review[word]
                else:
                    for word in review:
                        if word != " " and word != '':
                            if word in self.tokens:
                                self.tokens[word][3][0] += review[word]
                            else:
                                self.tokens[word] = [[0, 0], [0, 0], [0, 0], [review[word], 0]]
                            self.totalNegativeTokens += review[word]

                if value[0] == self.classes[2]:
                    for word in review:
                        if word != " " and word != '':
                            if word in self.tokens:
                                self.tokens[word][0][0] += review[word]
                            else:
                                self.tokens[word] = [[review[word], 0], [0, 0], [0, 0], [0, 0]]
                            self.totalFakeTokens += review[word]
                else:
                    for word in review:
                        if word != " " and word != '':
                            if word in self.tokens:
                                self.tokens[word][1][0] += review[word]
                            else:
                                self.tokens[word] = [[0, 0], [review[word], 0], [0, 0], [0, 0]]
                            self.totalTrueTokens += review[word]

        self.totalTokens = self.totalPositiveTokens + self.totalNegativeTokens + self.totalTrueTokens + self.totalFakeTokens
        self.totalWords = len(self.tokens)

    def calculatePriors(self):
        for class_name in self.classes:
            if class_name == "Pos":
                self.priors[class_name] = math.log(self.totalPositiveTokens / float(self.totalTokens))
            elif class_name == "Neg":
                self.priors[class_name] = math.log(self.totalNegativeTokens / float(self.totalTokens))
            elif class_name == "Fake":
                self.priors[class_name] = math.log(self.totalFakeTokens / float(self.totalTokens))
            elif class_name == "True":
                self.priors[class_name] = math.log(self.totalTrueTokens / float(self.totalTokens))

    def calculateLikelihoodProbabilities(self):
        for word in self.tokens:
            self.tokens[word][0][1] = math.log(
                (self.tokens[word][0][0] + 1) / float(self.totalFakeTokens + self.totalWords))
            self.tokens[word][1][1] = math.log(
                (self.tokens[word][1][0] + 1) / float(self.totalTrueTokens + self.totalWords))
            self.tokens[word][2][1] = math.log(
                (self.tokens[word][2][0] + 1) / float(self.totalPositiveTokens + self.totalWords))
            self.tokens[word][3][1] = math.log(
                (self.tokens[word][3][0] + 1) / float(self.totalNegativeTokens + self.totalWords))

    def writeModelToFile(self):
        outputFile = open(self.outputFileName, "w")
        outputFile.write("%s\n" % self.totalWords)
        for word, prob in self.priors.items():
            line = "%s %s\n" % (word, prob)
            outputFile.write(line)
        for word, value in self.tokens.items():
            line = "%s %s %s %s %s\n" % (word, value[0][1], value[1][1], value[2][1], value[3][1])
            outputFile.write(line)

def runNaiveBayesLearn():
    naiveBayesLearn = NaiveBayesLearn('./data/train-labeled.txt')
    naiveBayesLearn.readTrainingData()
    naiveBayesLearn.prepareCountsForModel()
    naiveBayesLearn.calculatePriors()
    naiveBayesLearn.calculateLikelihoodProbabilities()
    naiveBayesLearn.writeModelToFile()

runNaiveBayesLearn()

