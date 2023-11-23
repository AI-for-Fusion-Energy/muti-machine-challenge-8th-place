import numpy as np
import os
from sklearn.metrics import f1_score
from tsai.all import *

from data_process.load_trian_data import prepare_data

def main():
    window_size = 100
    X,Y = prepare_data(window_size)
    tfms = [None, TSClassification()]
    # tfms = [None, TSMultiLabelClassification()]
    batch_tfms = TSStandardize(by_sample=True)
    mv_clf = TSClassifier(X, Y, path='models', arch="gMLP", tfms=tfms, batch_tfms=batch_tfms, metrics=f1_score, device="cuda:1",verbose=0)

    # 训练时确保数据也在 GPU 上
    # save_callback = SaveModelCallback(monitor='f1_score', every_epoch=True, reset_on_fit=True)
    for epoch in range(100):
        mv_clf.fit(1, 0.0005)
        model_filename = f"gMLP100_10_{epoch}.pkl"
        mv_clf.export(model_filename)
    
if __name__ == "__main__":
    main()


