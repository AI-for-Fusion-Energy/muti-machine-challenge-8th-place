import numpy as np
import os
from processor import load_data_from_hdf5, sliding_window_cmod

def prepare_eval_data(window_size):
    eval_list = []
    evals_list = []
    file_names = []


    # hdata_folder = "/home/minglongwang/239/h2a"
    folder = "/Public/wan_pre/Cleaned_data_10/CMod/CMod_evaluate"
    # folder = "/Public/wan_pre/Cleaned_data_10/CMod/CMod_train"
    files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.hdf5')]
    # files = files[0:1]
    for file_path in files:
        # 生成ID的列表名
    #     if os.path.isfile(os.path.join(folder, file)):
        file_name = os.path.splitext(os.path.basename(file_path))[0]  # 获取文件名（不带后缀）
        id_string = "ID"  # 替换为您需要的ID字符串
        new_file_name = f"{id_string}_{file_name}"
        file_names.append(new_file_name)
        features, _, _ = load_data_from_hdf5(file_path)
    # 拼接特征和标签，进行步长为max_length/window_size的滑动窗口构建
        features_array = np.array(features)
        evals_list.append(features_array)
        stride = 1
        features_3d = sliding_window_cmod(features_array, window_size, stride, 100)
        eval_list.append(features_3d)

    return eval_list,file_names