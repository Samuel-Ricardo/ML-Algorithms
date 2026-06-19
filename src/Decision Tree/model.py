

import pandas as pd 
import numpy as np 
import pydotplus 
import matplotlib.pyplot as plt
import shutil

from sklearn import tree


def createTree(trainingData): 
    data = trainingData.iloc[:, :-1]
    labels = trainingData.iloc[:, -1]

    trainedTree = tree.DecisionTreeClassifier(criterion='entropy')
    trainedTree.fit(data, labels)

    return trainedTree

def showTree2pdf(trainedTree, filename): 
    if shutil.which("dot") is None:
        plt.figure(figsize=(12, 8))
        tree.plot_tree(trainedTree, filled=True, rounded=True)
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close()
        return

    dot_data = tree.export_graphviz(trainedTree, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(filename)

def data2vector(data):
    names = data.columns[:-1]
    for i in names:
        col = pd.Categorical(data[i])
        data[i] = col.codes 

    return data


data = pd.read_table("../../docs/dataset/tennis.txt", header=None, sep='\t')

trainningvec = data2vector(data)

decisionTree = createTree(trainningvec)

showTree2pdf(decisionTree, "temis.pdf")


