import numpy as np
import os
from processor import load_data_from_hdf5, sliding_window, sliding_window_cmod, concatenate_arrays


def prepare_data(window_size):

    h_features_list = []
    hdata_folder = "/Public/wan_pre/Cleaned_data_10/HL2A"
    hdata_files = [os.path.join(hdata_folder, file) for file in os.listdir(hdata_folder) if file.endswith('.hdf5')]
    # hdata_filse = hdata_files[:2]
    for file_path in hdata_files:
        features, tags, max_length= load_data_from_hdf5(file_path)
    #     print(max_length)
    # 拼接特征和标签，进行步长为max_length/window_size的滑动窗口构建
        features_array = np.array(features)
        features_tags = np.concatenate((features_array,[tags]))
        if 1 in tags:tag=1
        else:tag=0
    # input分别为输入数组，窗口大小，滑动步长，是否是破裂标签（对应了不同处理方式）
        if tag == 0 and max_length > 1000:
            stride = 20
            # 50
        else:
            stride = 10
        features_3d = sliding_window(features_tags, window_size, stride, tag, 100)
    #     print(features_3d.shape)
    #     print(tag)
        h_features_list.append(features_3d)


    j_features_list = []
    j_is_disrupt = []
    jdata_folder = "/Public/wan_pre/Cleaned_data_10/JText"
    jdata_files = [os.path.join(jdata_folder, file) for file in os.listdir(jdata_folder) if file.endswith('.hdf5')]
    for file_path in jdata_files:
        features, tags, max_length= load_data_from_hdf5(file_path)     
        # j_text里面发现有采样率为5k的数据里长度下采样之后小了一个，在最前面补零
        new_list = [lst + [lst[-1]]*(max_length - len(lst)) for lst in features]
        features_array = np.array(new_list)   
    #     features_array = np.array(features)
        if 1 in tags:
            tag=1
    #         print(file_path)
        else:tag=0
    # 拼接特征和标签，进行步长为max_length/window_size的滑动窗口构建
        features_tags = np.concatenate((features_array,[tags]))  
    # input分别为输入数组，窗口大小，滑动步长，是否是破裂标签（对应了不同处理方式）
    # 窗口不适合调整的太大，构建窗口的函数可能没办法全部读取后面破裂之前的数据（函数构建的不够好）
        if tag == 0:
            stride = 5
            # 20
        else:
            stride = 3
        features_3d = sliding_window(features_tags, window_size, stride, tag, 50)
        j_features_list.append(features_3d)


    cmod_list = []
    cmod_is_disrupt = []
    # file_names = []

    # hdata_folder = "/home/minglongwang/239/h2a"
    # folder = "/Public/wan_pre/Cleaned_data_new/CMod/CMod_evaluate"
    folder = "/Public/wan_pre/Cleaned_data_10/CMod/CMod_train"
    files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.hdf5')]
    # files = files[0:1]
    for file_path in files:
    #     file_name = os.path.splitext(os.path.basename(file_path))[0]  # 获取文件名（不带后缀）
    #     id_string = "ID"  # 替换为您需要的ID字符串
    #     new_file_name = f"{id_string}_{file_name}"
    #     file_names.append(new_file_name)
        features, tags, max_length = load_data_from_hdf5(file_path)
        features_array = np.array(features)
        features_tags = np.concatenate((features_array,[tags]))
        if 1 in tags:
            tag=1
            cmod_is_disrupt.append(1)
        else:
            tag=0
            cmod_is_disrupt.append(0)
        stride = 1
        features_3d = sliding_window_cmod(features_tags, window_size, stride, 100)
        cmod_list.append(features_3d)

    train_data = j_features_list + h_features_list + cmod_list
    all_features = concatenate_arrays(train_data)
    all_features = np.transpose(all_features,(0,2,1))
    all_feature = all_features[:-1,:,:]
    all_tags = all_features[-1,:,:]
    print(all_feature.shape)
    print(all_tags.shape)
    new_list = []

    # for i in range(all_tags.shape[0]):
    #     new_list.append(all_tags[i, 1].max())
    # all_tags = [int(x) for x in new_list]

    # 取窗口中的最右边的值作为这个窗口的标签
    for i in range(all_tags.shape[0]):
        new_list.append(all_tags[i, -1])
    all_tags = [int(x) for x in new_list]
    # print(len(all_tags))

    # print(all_feature.shape)
    # X_reshaped = all_feature.transpose(1, 0, 2).reshape(all_feature.shape[1], -1)
    X = all_feature.transpose(1,0,2)
    Y = all_tags
    print("X.shape:",X.shape)
    print("Y.shape:",len(Y))
    one_count = np.count_nonzero(Y)
    print("1:0 = ",one_count/(len(Y)-one_count))
    return X,Y
            