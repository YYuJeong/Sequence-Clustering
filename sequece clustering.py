# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:07:53 2019

@author: YuJeong
"""

'''
# Purchase Pattern 1: Food → Electronics → Fashon
# Purchase Pattern 2: Electronics → Food → Fashon
# Purchase Pattern 3: Fashon → Food → Electronics
'''

import csv
from anytree import Node, RenderTree, findall, util
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import random
import time

def ReadCSV(filename):
    ff = open(filename, 'r', encoding = 'utf-8')
    reader = csv.reader(ff)
    headers = next(reader, None)
    data = {}
    for hh in headers:
        data[hh] = []
    for row in reader: 
        for hh, vv in zip(headers, row):
                data[hh].append(vv)
    return data


def cmp(a, b):
    return (a > b) - (a < b)

def GenerateItemHierarchyTree(treeItem):
    for i in range(len(treeItem['Name'])):
        globals()[treeItem['Name'][i]] =  Node(treeItem['Name'][i], parent = globals()[treeItem['Parent'][i]], data = treeItem['Data'][i])
        item_hierarchy_tree.append(globals()[treeItem['Name'][i]])
    return root

def PrintItemHierarchyTree(root):
    print("=="*30)
    for row in RenderTree(root):
        pre, fill, node = row
        print(f"{pre}{node.name}, data: {node.data}")
    print("=="*30)

def LevenshteinDistance(str1, str2):  #Dynamic Programming 
    str1Len = len(str1)
    str2Len = len(str2)
    matrix = [[0 for x in range(str2Len + 1)] for x in range(str1Len + 1)] 
    for i in range(str1Len + 1): 
        for j in range(str2Len + 1): 
            if i == 0: 
                matrix[i][j] = j    # Min. operations = j 
            elif j == 0: 
                matrix[i][j] = i    # Min. operations = i 
            elif str1[i-1] == str2[j-1]: 
                matrix[i][j] = matrix[i-1][j-1] 
            else: 
                matrix[i][j] = 1 + min(matrix[i][j-1],        # Insert 
                                   matrix[i-1][j],        # Remove 
                                   matrix[i-1][j-1])   # Replace  
    return matrix[str1Len][str2Len]

def NewLevenshteinDistance(str1, str2):
    str1Len = len(str1)
    str2Len = len(str2)
    matrix = [[0 for x in range(str2Len + 1)] for x in range(str1Len + 1)]   
    for i in range(str1Len + 1): 
        for j in range(str2Len + 1): 
            if i == 0: 
                matrix[i][j] = j    
            elif j == 0: 
                matrix[i][j] = i    
            else: # Add Hierarchy Tree 
                if str1[i-1]==str2[j-1]: 
                    cost = 0
                else:
                    cost = ComputeDiagonalCost(matrix, i, j, str1, str2, root)
                matrix[i][j] = round(min(matrix[i][j-1] + 1,        # Insert 
                                   matrix[i-1][j] + 1,        # Remove 
                                   matrix[i-1][j-1] + cost), 3)    # Replace   
    return matrix[str1Len][str2Len]

def ComputeDiagonalCost(matrix, i, j, str1, str2, root):
    maxlength = SearchLongestPath(root)
    if ((matrix[i-1][j] + 1) > matrix[i-1][j-1]) and ((matrix[i][j-1] + 1) > matrix[i-1][j-1]):
        str1char = findall(root, filter_=lambda node: node.name in (str1[i-1]))
        str2char = findall(root, filter_=lambda node: node.name in (str2[j-1]))
        str1char = str(str1char)
        str2char = str(str2char)
        str1lst = str1char.split('/')
        str2lst = str2char.split('/')
        for l in range(min(len(str1lst), len(str2lst))):
            if str1lst[l] != str2lst[l]:
                cmpindex = l
                break
        itempath = (len(str1lst)-cmpindex)+(len(str2lst)-cmpindex)
        cost = round(itempath/maxlength, 3)
    else:
        cost = 1
    return cost

def SearchLongestPath(root):
    toContent = list()
    for ee in root.leaves:
        toContent.append(str(ee))
    toString = list()
    for ee in toContent:
        toString.append(ee[6:-2])
    eachNode = list()
    for ee in toString:
        eachNode.append(ee.split('/'))
    longestPath = list()
    for ee in eachNode:
        longestPath.append(ee[:-1])
    dupliPath = list(set([tuple(set(item)) for item in longestPath]))
    pathLen = list()
    for ee in dupliPath:
        pathLen.append(len(ee)-1)
    pathLen.sort(reverse=True)
    maxlength = pathLen[0] + pathLen[1]
    return maxlength


def ComputeLevenshteinSimilarity(LevenshteinDist, str1, str2):
    maxlen = max(len(str1), len(str2))
    similarity = 1 - LevenshteinDist/maxlen
    return round(similarity, 3)

def PrintMatrix(matrix, str1, str2):
    str1Len = len(str1)
    str2Len = len(str2)
    print('{:4s}'.format('    -   '), end=" ")
    for i in range(str2Len):
        print('{:4s}'.format(str2[i]), end=" ")
    print(" ")
    for i in range(str1Len + 1):
        if i > 0:
            print(str1[i-1], end=" ")
        else:
            print("-", end=" ")
        print("[", end=" ")
        for j in range(str2Len + 1):
          print('{:4s}'.format(str(matrix[i][j])), end=" ")
        print("]")
    print("")
    
def generateRandomSequence(size, chars):
    return ''.join(random.choice(chars) for _ in range(size))

def GenerateRandomSequence():
    foodChars = 'abcdefgh'
    electronicsChars = 'ijklm'
    fashionChars = 'nopqrstuv'
    foodItemArr= []
    electronicsItemArr = []
    fashionItemArr = []
    '''
    foodItemArr = [generateRandomSequence(3, foodChars) for j in range(10)]
    electronicsItemArr = [generateRandomSequence(3, electronicsChars) for j in range(10)]
    fashionItemArr = [generateRandomSequence(3, fashionChars) for j in range(10)]
    '''
    foodItemArr = [[generateRandomSequence(i, foodChars) for j in range(50)] for i in range(1,6)]
    electronicsItemArr = [[generateRandomSequence(i, electronicsChars) for j in range(50)] for i in range(2,6)]
    fashionItemArr = [[generateRandomSequence(i, fashionChars) for j in range(50)] for i in range(2, 7)]

    return foodItemArr, electronicsItemArr, fashionItemArr
    
def GeneratePurchasePattern(foodItemArr, electronicsItemArr, fashionItemArr):
    '''
    # Purchase Pattern 1: Food → Electronics → Fashon
    # Purchase Pattern 2: Electronics → Food → Fashon
    # Purchase Pattern 3: Fashon → Food → Electronics
    '''
    random.shuffle(foodItemArr)
    random.shuffle(electronicsItemArr)
    random.shuffle(fashionItemArr)
    pattern1 = []
    pattern2 = []
    pattern3 = []

    for j in range(len(foodItemArr[0])):
        pattern1.append(foodItemArr[4][j] + electronicsItemArr[1][j] + fashionItemArr[0][j]) # 5:3:2 #
        pattern2.append(electronicsItemArr[2][j] + foodItemArr[3][j] + fashionItemArr[0][j]) # 4:4:2 #
        pattern3.append(fashionItemArr[4][j] + foodItemArr[1][j] + electronicsItemArr[0][j]) # 2:2:6 #          
    clustData = []
    clustData.extend(pattern1)
    clustData.extend(pattern2)
    clustData.extend(pattern3)
    labels = []
    sqlabel = 's'
    for i in range(len(clustData)):
        labels.append(sqlabel + str(i+1))
    clustDf = pd.DataFrame(clustData, columns = ['Sequence'], index = labels)
    return clustDf

def ComputeDistMatrix(clustDf):
    distArr = list()
    for i in range(len(clustDf['Sequence'])):
        for j in range(i+1, len(clustDf['Sequence'])):
            distArr.append(NewLevenshteinDistance(clustDf['Sequence'][i],clustDf['Sequence'][j]))
    dist = squareform(distArr)
    return dist

def AgglomerativeCluster(clustDf):
    dist = ComputeDistMatrix(clustDf)
    aggloClusters = linkage(dist, method='average')
    print("=="*30)
    dendrogram(aggloClusters, labels = clustDf.index)

def PrintClusterObjectLabels(clustDf):
    dist = ComputeDistMatrix(clustDf)
    aggCluster = AgglomerativeClustering(n_clusters=3, affinity='precomputed', linkage='average')
    predict = aggCluster.fit_predict(dist)
    print(predict)
    return predict

def ComputeClusterAccuracy(predict):
    correct = []
    label0 = 0
    label1 = 0
    label2 = 0
    for i in range(150):
        if i%3 == 0:
            correct.append(0)
            label0 += 1
        elif i%3 == 1:
            correct.append(1)
            label1 += 1
        else:
            correct.append(2)
            label2 += 1
    fit = 0
    for i in range(len(correct)):
        if correct[i] == predict[i]:
            fit += 1
    accuracy = round(fit/len(correct), 3)
    print(accuracy)
    print("label0:", label0)
    print("label1:", label1)
    print("label2:", label2)
    #for i in range(60):
        
    return accuracy            

if __name__ == '__main__':
    treeItem = ReadCSV('tree.csv')
    data = ReadCSV('data.csv')
    
    item_hierarchy_tree = []    
    root = Node("R", data = "All Item")
    item_hierarchy_tree.append(root) 
    root = GenerateItemHierarchyTree(treeItem)
    PrintItemHierarchyTree(root)
    
    foodItemArr, electronicsItemArr, fashionItemArr = GenerateRandomSequence()

    clustDf = GeneratePurchasePattern(foodItemArr, electronicsItemArr, fashionItemArr)

    print("< Agglomerative Clustering >")
   # AgglomerativeCluster(clustDf)
    start_time = time.time() 
    predict = PrintClusterObjectLabels(clustDf)
    print("start_time", start_time)
    print("--- %s seconds ---" %(time.time() - start_time))
    print("=="*30)

    print("<Cluster Accuracy>")
    accuracy = ComputeClusterAccuracy(predict)





    