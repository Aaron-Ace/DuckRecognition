'''
運算公式：

P( h | d) = P ( d | h ) * P( h) / P(d)

這裡：
P ( h | d )：是因子h基於數據d的假設概率,叫做後驗概率
P ( d | h ) : 是假設h為真條件下的數據d的概率
P( h)　: 是假設條件h為真的時候的概率（和數據無關），它叫做h的先驗概率
P(d)　: 數據d的概率，和先驗條件無關．

算法實現分解：

１　數據處理：加載數據並把他們分成訓練數據和測試數據 -> Already Complete
２　匯總數據：匯總訓練數據的概率以便後續計算概率和做預測
３　結果預測：　通過給定的測試數據和匯總的訓練數據做預測
４　評估準確性：使用測試數據來評估預測的準確性
'''

import csv
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import math


def loadCsv(filename):
    lines = csv.reader(open(filename, "r"))  # 將文件以讀的模式打開
    dataset = list(lines)  # 每行存進去一個list
    # print(dataset)
    for i in range(1, len(dataset)):  # 因為標題含有不能轉成數字的字元，所以range從1開始
        dataset[i] = [float(x) for x in dataset[i]]  # 將轉成數字的Data存進去dataset
    # print('Loaded data file {0} with {1} rows'.format(filename, len(dataset)))
    return dataset


def LoadDuckImage(filename):
    colourImg = Image.open(filename)
    colourPixels = colourImg.convert("RGB")
    colourArray = np.array(colourPixels.getdata()).reshape(colourImg.size + (3,))
    indicesArray = np.moveaxis(np.indices(colourImg.size), 0, 2)
    allArray = np.dstack((indicesArray, colourArray)).reshape((-1, 5))
    df = pd.DataFrame(allArray, columns=["x", "y", "red", "green", "blue"])
    df.drop(['x', 'y'], axis=1, inplace=True)
    data = df.to_numpy().astype(float).tolist()
    return data


def separateByClass(dataset):  # 最後一個屬性（-1）為類別值，返回一個類別值到資料樣本清單的映射。
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):  # -1 代表vector倒數第一個
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def mean(numbers):
    return sum(numbers) / float(len(numbers))  # Average number


def stdev(numbers):  # 標準差
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):  # 計算每個屬性的均值和標準差
    summaries = [(mean(attribute), stdev(attribute)) for attribute in
                 zip(*dataset[1:-2])]  # zip函數將資料樣本按照屬性分組為一個個清單，然後可以對每個屬性計算均值和標準差
    del summaries[-1]
    return summaries


def summarizeByClass(dataset):  # 首先將訓練資料集按照類別進行劃分，然後計算每個屬性的摘要
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    # print('Summary by class value: {0}'.format(summaries))
    return summaries


def calculateProbability(x, mean, stdev):  # 計算高斯概率密度函數
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))  # 計算指數部分，然後計算等式的主幹
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):  # 給定一個資料樣本，它所屬每個類別的概率，可以通過將其屬性概率相乘得到,結果是一個類值到概率的映射
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    # print('Probabilities for each class: {0}'.format(probabilities))
    return probabilities


def predict(summaries, inputVector):  # 計算一個資料樣本屬於每個類的概率，找到最大的概率值，並返回關聯的類別
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):  # 通過對測試資料集中每個資料樣本的預測，可以評估模型精度
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def createNewImage(predictions,imagepath):
    Images =[]  #預測結果置換黑與白
    for i in range(len(predictions)):
        if(1 == int(predictions[i])):#如果預測為鴨子(=1)將顏色設為白色
            Images.append([255,255,255])
        else:
            Images.append(([0,0,0]))#如果預測為鴨子(=2)將顏色設為白色
    Images = np.array(Images)
    img = cv2.imread(imagepath)
    size = img.shape
    array = np.reshape(Images, (size[0], -1))#將array大小設定成原始大小的圖片size

    # 利用 Pillow 用上面的新array創造一個新的圖片
    new_image = Image.fromarray(array)
    new_image = new_image.resize((size[0], size[1]))
    new_image.show()
    print("Successful create new images")

def main():
    imagepath = "full_duck_2.jpg"
    train_data = loadCsv("pixel_RGB.csv")  # 輸入data
    test_data = LoadDuckImage(imagepath)

    train_data = train_data[1:]  # 將標題過濾

    # prepare model
    summaries = summarizeByClass(train_data)
    # print(data_through)

    # test model
    predictions = getPredictions(summaries, test_data)

    createNewImage(predictions,imagepath)


main()
