def loadData():
    fileTest = open(r'C:\Users\Mohamed Ahmed\Desktop\study\study 2022\adv ai\KNN on irisDataSet\TestData.txt', 'r')
    testData = []

    for line in fileTest:
        linePoints = line.split(';')
        points = []
        #append every point (minus seperator) to 'points' list
        for i in range(len(linePoints) - 1):
            points.append(float(linePoints[i]))
        lastPoint = linePoints[-1]
        #append extracted points from 'points' list to 'TestData' list
        if lastPoint != '\n':
            points.append(lastPoint[0:6])
            testData.append(points)
    #print(testData)
    fileTest.close()
    
    fileTrain = open(r'C:\Users\Mohamed Ahmed\Desktop\study\study 2022\adv ai\KNN on irisDataSet\TrainData.txt', 'r')
    trainData = []
    for line in fileTrain:
        linePoints = line.split(';')
        points = []
        for i in range(len(linePoints) - 1):
            points.append(float(linePoints[i]))
        lastPoint = linePoints[-1]
        if lastPoint != '\n':
            points.append(lastPoint[0:6])
            trainData.append(points)
    #print(trainData)
    fileTrain.close()
    return trainData , testData

#trainingData , testingData = loadData()
#print (len(trainingData), len(testingData))

import math
def eclideanDistance(x, xi):
    d = 0.0
    for i in range(len(x) - 1):
        d += pow(x[i] - xi[i], 2)
    math.sqrt(d)
    return d


def getAccuracy(predictions, test):
    correct = 0
    for x in range(len(test)):
        if predictions[x] == test[x][-1]:
            correct += 1
    result = (correct/(float(len(test)))) * 100.0
    return correct, result

import operator
def knnPredict(train, test, k):
    predictions = []
    print('K Value: ', k)
    for iTest in test:
        distances = []
        #Apply euclidean to points
        for iTrain in train:
            dist = eclideanDistance(iTest, iTrain)
            distances.append((iTrain, dist))
        #Sort distances after euclidean to choose least base on k value
        distances.sort(key = operator.itemgetter(1))
        iTestKNN = distances[0:k]
        classCounter = {}
        for iDist in iTestKNN:
            if iDist[0][-1] in classCounter:
                classCounter[iDist[0][-1]] += 1
            else:
                classCounter[iDist[0][-1]] = 1
        #print(classCounter)
        #print('##########################################################')
        classCounterSorted = sorted(classCounter.items(), key=operator.itemgetter(1), reverse=True)
        #print(classCounterSorted)
        print('Predicted Class: ', classCounterSorted[0][0], 'Actual Class: ', iTest[-1])
        predictions.append(classCounterSorted[0][0])
    correct, accuracy = getAccuracy(predictions, test)
    print('Number of correctly classified instances : ', correct)
    print('Total number of instances : ', len(test))
    print('Accuracy: ', accuracy)

trainingData , testingData = loadData()
for k in range(1,20):
    knnPredict(trainingData, testingData, k)
    print('##########################################################')
