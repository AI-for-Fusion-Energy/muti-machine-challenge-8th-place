import numpy as np
import h5py

def sliding_window_cmod(array_2d, window_size, stride, tag_length):
    num_features, sequence_length = array_2d.shape
    if sequence_length < window_size:
        array_3d = np.zeros((num_features, window_size, 1))
        zero_numbers = window_size - sequence_length
        for feature in range(num_features):
            array_3d[feature,:zero_numbers,:] = 0
            window_data = array_2d[feature,:]
            array_3d[feature,zero_numbers:,0] = window_data
        return array_3d  
    array_3d = np.zeros((num_features, window_size, 2*sequence_length))
    
    for feature in range(num_features):
        start_position = -1
        position = 0
        while(start_position < sequence_length - window_size - 1):
            start_position += 1
            array_3d[feature, :, position] = array_2d[feature, start_position:start_position+window_size]            
            position += 1
    return array_3d[:,:,:position]


def sliding_window(array_2d, window_size, stride, tag, tag_length):
    num_features, sequence_length = array_2d.shape
    if sequence_length < window_size:
        array_3d = np.zeros((num_features, window_size, 1))
        zero_numbers = window_size - sequence_length
        for feature in range(num_features):
            array_3d[feature,:zero_numbers,:] = 0
            window_data = array_2d[feature,:]
            array_3d[feature,zero_numbers:,0] = window_data
        return array_3d  
    array_3d = np.zeros((num_features, window_size, sequence_length+window_size))
    
    for feature in range(num_features):
        s1 = 0
        s2 = stride
        start_position = 0
        window = 0
        if tag == 1:end = sequence_length-2*tag_length
        else:end = sequence_length-window_size
        while start_position < (sequence_length - 2*tag_length):
            start_position = np.random.randint(s1, s2)
            s2 += stride
            if start_position + window_size > sequence_length:
                continue
            window_data = array_2d[feature, start_position:start_position+window_size]
            array_3d[feature, :, window] = window_data
            window += 1
        position = window
        if tag == 1:
            while(start_position < sequence_length - window_size - 1):
                start_position += 1
                array_3d[feature, :, position] = array_2d[feature, start_position:start_position+window_size]            
                position += 1
    return array_3d[:,:,:position]


def load_data_from_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
#         features = [list(f[feature][:]) for feature in f]  # Load each feature
        features = []
        tags_dataset = f['IsDisrupt']  # Assuming the tag is named 'IsDisrupt'
        length = len(f['AXUV_01'])
        for feature_name in f:
            if feature_name in ('start_time','IsDisrupt','time_1k','time_5k','poloidal Mirnov probes_01','poloidal Mirnov probes_02','poloidal Mirnov probes_03','poloidal Mirnov probes_04','poloidal Mirnov probes_05','poloidal Mirnov probes_06','poloidal Mirnov probes_07','poloidal Mirnov probes_08','poloidal Mirnov probes_09','poloidal Mirnov probes_10','poloidal Mirnov probes_11','poloidal Mirnov probes_12','toroidal Mirnov probes_01','toroidal Mirnov probes_02'):
                continue
#             tags1 = ['plasma current','LM proxy','q95 proxy','IP error fraction','Radiation fraction','Rotating mode proxy','C3 radiation','loop voltage','toroidal magnetic field','horizontal displacement','vertical displacement']
#             tags0 = ['poloidal Mirnov probes_04_1k','poloidal Mirnov probes_13_1k','soft-X-ray_05','soft-X-ray_10','line integral density(center chord)','AXUV_09','AXUV_10','AXUV_11']
#             features_new = tags0+tags1
#             features_new = tags1
            feature = f[feature_name]
#             if feature_name in features_new:
            if feature.shape == None:
                features.append([0]*length)
            elif (len(feature.shape))==0:
                print(file_path)
                print(feature_name)
            else:
                feature_list = list(feature[:])
                features.append(feature_list)
        tags = tags_dataset[()]  # Extract the value from the dataset
    return features, tags, length


def concatenate_arrays(arrays):
    """
    将多个三维数组拼接在一起，保持前两个维度不变
    参数：
    arrays: 三维数组的列表
    返回：
    拼接后的三维数组
    """
    # 检查数组是否为空
    if len(arrays) == 0:
        return None
    # 确保所有数组的前两个维度相等
    for i in range(1, len(arrays)):
        if arrays[i].shape[:2] != arrays[0].shape[:2]:
            raise ValueError("前两个维度不相等")
    # 获取数组的形状信息
    height, width = arrays[0].shape[:2]
    # 计算拼接后的数组形状
    depth = sum(arr.shape[2] for arr in arrays)
    # 创建一个空数组用于存储结果
    result_array = np.empty((height, width, depth))
    # 按顺序将每个数组拼接到结果数组中
    pos = 0
    for arr in arrays:
        result_array[:, :, pos:pos+arr.shape[2]] = arr
        pos += arr.shape[2]
    return result_array


def check_condition(input_list):
    threshold1 = 0.6
    threshold2 = 0.7
    consecutive_length_fraction = 1 / 5  # 五分之一

    # 检查是否有三分之一及以上的点大于0.6
    count_above_threshold1 = sum(1 for value in input_list if value > threshold1)
#     count_above_threshold1 = 0
#     for value in input_list:
#         if value > threshold1:
#             count_above_threshold1 += 1
    if max(input_list)>0.85:
        return 1
    
    if count_above_threshold1 >= len(input_list) / 3:
        return 1

    # 检查是否有连续列表长度五分之一的点大于0.7
    consecutive_count = 0
    for value in input_list:
        if value > threshold2:
            consecutive_count += 1
            if consecutive_count >= len(input_list) * consecutive_length_fraction:
                return 1
        else:
            consecutive_count = 0
    return 0

