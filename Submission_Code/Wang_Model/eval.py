import h5py
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from itertools import chain
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import csv
import matplotlib.pyplot as plt
from tsai.all import *

from data_process.load_trian_data import prepare_data
from data_process.load_eval_data import prepare_eval_data
from processor import check_condition

def make_predict(epoch, eval_list, file_names):
    probs = []
    preds = []
    # model_path = f"models/TS80_{epoch}.pkl"
    model_path = f"models/gMLP100_0005_{epoch}.pkl"
    model = load_learner(model_path, cpu=False)
    # path = Path('./models/mv_clf.pkl')
    # model = load_learner(PATH, cpu=False)
    for features_array in eval_list:
        tx = np.transpose(features_array,(2,0,1))
        prob, _, pred = model.get_X_preds(tx)   
        proba = prob[:,1].numpy().tolist()
        probs.append(proba)
        pred_float = np.array([float(value) for value in pred])
        pred_int = (pred_float >= 0.5).astype(int)
        preds.append(pred_int)
    predictions = []
    for prob in probs:
        length  = len(prob)//2
        list0 = list(prob[length:])
        predictions.append(check_condition(list0))  
    return  predictions, probs


def main():
    eval_list,file_names = prepare_eval_data(window_size=100)
    for epoch in range(100):
        predictions, probs = make_predict(epoch, eval_list, file_names)
    
    # 检查文件夹是否存在，如果不存在则创建
    folder_path = './pic/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # 保存图片
    plt.plot(probs)
    title = f"max:{max(probs)},epoch:{probs.index(max(probs))}"
    plt.title(title)
    plt.savefig('./pic/gMLP_100_12.png')

if __name__ == "__main__":
    main()