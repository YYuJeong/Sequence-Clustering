# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:07:53 2019

@author: YuJeong
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:46:30 2019

@author: YuJeong
"""
import csv
from anytree import Node, RenderTree, findall, util
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd
from nltk.cluster import KMeansClusterer, euclidean_distance

'''
str1 = "aegi"
str2 = "jdai"
str1Len = len(str1)
str2Len = len(str2)
'''

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
    
    ''' //입력 받아서 트리 생성
    while 1:
        nodeName, nodeData, nodeParent = input("카테고리/ 데이터/ 부모노드: ").split()
        if nodeName == "Exit":
            break
        globals()[nodeName] =  Node(nodeName, parent = globals()[nodeParent], data = nodeData)
        item_hierarchy_tree.append(globals()[nodeName])
    '''
    return root

def PrintItemHierarchyTree(root):
    print("=="*30)
    for row in RenderTree(root):
        pre, fill, node = row
        print(f"{pre}{node.name}, data: {node.data}")
    print("=="*30)


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
    
def ComputeDistMatrix(data):
    distArr = list()
    for i in range(len(data['Transaction'])):
        for j in range(i+1, len(data['Transaction'])):
            distArr.append(NewLevenshteinDistance(data['Transaction'][i], data['Transaction'][j]))
    dist = squareform(distArr)
    print(dist)
    return dist

def AgglomerativeClustering(data):
    distArr = list()
    for i in range(len(data['Transaction'])):
        for j in range(i+1, len(data['Transaction'])):
            distArr.append(NewLevenshteinDistance(data['Transaction'][i], data['Transaction'][j]))
    dist = squareform(distArr)
    aggloClusters = linkage(dist, method='complete')
    
    printcluster=pd.DataFrame(aggloClusters, columns=['clusterID_1', 'clusterID_2', 'distance', 'cluster member num'],
                index=['cluster %d' %(i+1) for i in range(aggloClusters.shape[0])])
    dendro = dendrogram(aggloClusters, labels = data['Id'])
    #print(dendro)

    return aggloClusters

'''
def kMeansClustering(data):
    NUM_CLUSTERS = 3
   # data =  np.array([[0, 1], [1, 0], [2, 0], [3,4]]) 
    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance = NewLevenshteinDistance, repeats=25)
    assigned_clusters = kclusterer.cluster(data, assign_clusters=True)
    return kcluster
    
'''
if __name__ == '__main__':
    treeItem = ReadCSV('tree.csv')
    data = ReadCSV('data.csv')
    
    item_hierarchy_tree = []    
    root = Node("R", data = "All Item")
    item_hierarchy_tree.append(root) 
    root = GenerateItemHierarchyTree(treeItem)
    PrintItemHierarchyTree(root)
    
    print("< New Distance Measure >")
    NewLevenshteinDist = NewLevenshteinDistance(data['Transaction'][0], data['Transaction'][1])
    print("LevenshteinDistance: ", NewLevenshteinDist)
    print("=="*30)

    dist = ComputeDistMatrix(data)    
    print("< Agglomerative Clustering >")
    aggloClusters = AgglomerativeClustering(data)
    #dendro = dendrogram(aggloClusters, labels = data['Id'])
   # print(aggloClusters)
    print("=="*30)








    