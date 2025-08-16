import math

import scipy.signal as signal
import os

# from DE_PSD import *
import numpy as np
import pandas as pd
import torch
from dotmap import DotMap
from mne.decoding import CSP
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from pyriemann.estimation import Covariances
# from rieman import ProjCommonSpace
# from featuring import Riemann
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import torch.nn as nn
# from model_test import *
# from spd import *
# def tangentspace_learning(spoc):
#
#     ''' Learning Processing: from Riemannian Space to Tangent Space '''
#
#     geom = Riemann(n_fb=1, metric='riemann').transform(spoc)
#     scaler = StandardScaler()
#     scaler.fit(geom)
#     sc = scaler.transform(geom)
#     return sc
class signal2spd(nn.Module):# convert signal epoch to SPD matrix
    def __init__(self):
        super().__init__()
        self.dev = torch.device('cpu')
    def forward(self, x):
        x = x.squeeze()
        mean = x.mean(axis=-1).unsqueeze(-1).repeat(1, 1, x.shape[-1])
        x = x - mean
        cov = x @ x.permute(0, 2, 1)
        cov = cov.to(self.dev)
        cov = cov / (x.shape[-1] - 1)
        tra = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        tra = tra.view(-1, 1, 1)
        cov /= tra
        identity = torch.eye(cov.shape[-1], cov.shape[-1], device=self.dev).to(self.dev).repeat(x.shape[0], 1, 1)
        cov = cov + (1e-5 * identity)
        return cov
class E2R(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.signal2spd = signal2spd()
        # self.signal2spd = signal2x()
    def patch_len(self, n, epochs):
        list_len = []
        base = n // epochs
        for i in range(epochs):
            list_len.append(base)
        for i in range(n - base * epochs):
            list_len[i] += 1
        if sum(list_len) == n:
            return list_len
        else:
            return ValueError('check your epochs and axis should be split again')
    def forward(self, x):
        # x with shape[bs, ch, time]
        x = torch.from_numpy(x)  # 将 NumPy 数组转换为 PyTorch 张量
        list_patch = self.patch_len(x.shape[-1], int(self.epochs))
        x_list = list(torch.split(x, list_patch, dim=-1))
        for i, item in enumerate(x_list):
            x_list[i] = self.signal2spd(item)
        x = torch.stack(x_list).permute(1, 0, 2, 3)
        print('-------------------------------------------', x.shape)
        return x
def get_DTU_data(name="S1", timelen=1, data_document_path="E:/听觉注意力/DTU/DATA_preproc"):
    class CustomDatasets(Dataset):
        # initialization: data and label
        def __init__(self, data, adj, label):
            self.data = torch.Tensor(data)
            self.adj = torch.Tensor(adj)
            self.label = torch.tensor(label, dtype=torch.uint8)

        # get the size of data
        def __len__(self):
            return len(self.label)

        # get the data and label
        # def __getitem__(self, index):
        #     return self.data[index], self.label[index]
        def __getitem__(self, index):
            return self.data[index], self.adj[index], self.label[index]

    def get_data_from_mat(mat_path):
        '''
        discription:load data from mat path and reshape
        param{type}:mat_path: Str
        return{type}: onesub_data
        '''
        mat_eeg_data = []
        mat_wavA_data = []
        mat_wavB_data = []
        mat_event_data = []
        matstruct_contents = loadmat(mat_path)
        matstruct_contents = matstruct_contents['data']
        mat_event = matstruct_contents[0, 0]['event']['eeg'].item()
        mat_event_value = mat_event[0]['value']  # 1*60

        print(mat_event_value.shape)
        # mat_event_value = np.where(mat_event_value == 1, 0, mat_event_value)
        # mat_event_value = np.where(mat_event_value == 2, 1, mat_event_value)
        # two_d_array = np.array([np.array([item], dtype='uint8') for item in mat_event_value])
        # mat_event_value = np.array([two_d_array], dtype='object')

        mat_eeg = matstruct_contents[0, 0]['eeg']  # 60 trials 3200*66
        print(mat_eeg.shape)
        mat_wavA = matstruct_contents[0, 0]['wavA']
        mat_wavB = matstruct_contents[0, 0]['wavB']
        for i in range(mat_eeg.shape[1]):
            mat_eeg_data.append(mat_eeg[0, i])
            mat_wavA_data.append(mat_wavA[0, i])
            mat_wavB_data.append(mat_wavB[0, i])
            mat_event_data.append(mat_event_value[i][0][0])
        print(mat_event_data)

        # return mat_eeg_data, mat_wavA_data, mat_wavB_data, mat_event_data

        return mat_eeg_data, mat_event_data

    def sliding_window(eeg_datas, labels, args, eeg_channel):
        window_size = args.window_length
        stride = int(window_size * (1 - args.overlap))

        train_eeg = []
        test_eeg = []
        train_label = []
        test_label = []

        for m in range(len(labels)):
            eeg = eeg_datas[m]
            label = labels[m]
            windows = []
            new_label = []
            for i in range(0, eeg.shape[0] - window_size + 1, stride):
                window = eeg[i:i + window_size, :]
                windows.append(window)
                new_label.append(label)
            # 计算分割点
            train_eeg.append(np.array(windows)[:int(len(windows) * 1)])
            # test_eeg.append(np.array(windows)[int(len(windows) * 0.9):int(len(windows) * 1)])
            train_label.append(np.array(new_label)[:int(len(windows) * 1)])
            # test_label.append(np.array(new_label)[int(len(windows) * 0.9):int(len(windows) * 1)])
            # train_eeg.append(np.array(windows)[int(len(windows) * 0.2):])
            # train_label.append(np.array(new_label)[int(len(windows) * 0.2):])
        # 堆叠和重塑数据
        train_eeg = np.concatenate(train_eeg, axis=0).reshape(-1, window_size, eeg_channel)
        # test_eeg = np.concatenate(test_eeg, axis=0).reshape(-1, window_size, eeg_channel)
        train_label = np.concatenate(train_label, axis=0).reshape(-1, 1)
        # test_label = np.concatenate(test_label, axis=0).reshape(-1, 1)
        # train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, eeg_channel)
        # test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, eeg_channel)
        # train_label = np.stack(train_label, axis=0).reshape(-1, 1)
        # test_label = np.stack(test_label, axis=0).reshape(-1, 1)
        return train_eeg, train_label
        # return train_eeg,test_eeg, train_label,test_label

    print("Num GPUs Available: ", torch.cuda.is_available())
    print(name)
    time_len = timelen
    random_seed = 42
    args = DotMap()
    args.name = name
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 64 * time_len
    args.window_length = math.ceil(args.fs)
    args.overlap = 0.6
    args.batch_size = 32
    args.max_epoch = 200
    args.random_seed = random_seed
    args.people_number = 18
    args.eeg_channel = 64
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 60
    args.cell_number = 3200
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.log_interval = 20
    args.csp_comp = 64
    args.label_col = 0
    args.log_path = "ConvTran-main-DTU/Results/1s"
    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)
    subpath = args.data_document_path + '/' + str(args.name) + '_data_preproc.mat'
    eeg_data, event_data = get_data_from_mat(subpath)
    eeg_data = np.array(eeg_data)
    eeg_data = eeg_data[:, :, 0:64]

    event_data = np.array(event_data)
    print(eeg_data.shape)
    eeg_data = np.vstack(eeg_data)
    eeg_data = eeg_data.reshape([args.trail_number, -1, args.eeg_channel])
    event_data = np.vstack(event_data)
    # print(event_data)
    event_data = np.where(event_data == 1, 0, event_data)
    event_data = np.where(event_data == 2, 1, event_data)
    # print(event_data)
    eeg_data = np.array(eeg_data)
    print(eeg_data.shape)
    #
    # eeg_data = eeg_data.transpose(0, 2, 1)
    # event_data = np.squeeze(event_data - 1)
    # csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
    #           norm_trace=True)
    # eeg_data = csp.fit_transform(eeg_data, event_data)
    # eeg_data = eeg_data.transpose(0, 2, 1)
    train_data, train_label = sliding_window(eeg_data, event_data, args, args.csp_comp)
    # train_data, test_data, train_label, test_label = sliding_window(eeg_data, event_data, args, args.csp_comp)
    del eeg_data
    del event_data


    args.n_test = int(len(train_label) * 0.1)
    args.n_valid = args.n_test
    args.n_train = len(train_label) - args.n_test * 2

    # print(1, data.shape)
    print("len of test_label", args.n_test, len(train_label))
    # del data

    print(train_data.shape, 5)
    train_data = train_data.transpose(0, 2, 1)
    # test_data = test_data.transpose(0, 2, 1)


    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)

    # indices2 = np.arange(test_data.shape[0])
    # np.random.shuffle(indices2)

    train_data, train_label = train_data[indices], train_label[indices]
    print(args.n_train, args.n_valid)
    valid_data, valid_label = train_data[args.n_train:args.n_train + args.n_valid], train_label[
                                                                                    args.n_train:args.n_train + args.n_valid]
    test_data, test_label = train_data[args.n_train + args.n_valid:], train_label[args.n_train + args.n_valid:]
    train_data, train_label = train_data[:args.n_train], train_label[:args.n_train]




    # set the number of training, testing and validating data
    # args.n_test = len(test_label)
    # args.n_valid = args.n_test
    # args.n_train = len(train_label) - args.n_test
    #
    #
    # train_data = train_data.transpose(0, 2, 1)
    # test_data = test_data.transpose(0, 2, 1)
    #
    # indices = np.arange(train_data.shape[0])
    # np.random.shuffle(indices)
    # train_data, train_label = train_data[indices], train_label[indices]
    #
    # valid_data, valid_label = train_data[args.n_train:], train_label[args.n_train:]
    # train_data, train_label = train_data[:args.n_train], train_label[:args.n_train]

    ##--------------------------------------流形空间修改测试
    # cov_train = Covariances('oas').transform(train_data)     ##协方差矩阵    oas重叠平均分割:将时间序列数据分割成重叠的片段，然后计算每个片段的协方差矩阵。
    # cov_train = cov_train[:, None, :, :]
    # cov_valid = Covariances('oas').transform(valid_data)
    # cov_valid = cov_valid[:, None, :, :]
    # cov_test = Covariances('oas').transform(test_data)
    # cov_test = cov_test[:, None, :, :]
    #
    # spoc = ProjCommonSpace(rank_num=64)      #空间映射的对象
    # # spoc = ProjSPoCSpace(rank_num=rank_num, scale='auto')
    #
    # spoc_train = spoc.fit(cov_train).transform(cov_train)# fit 在训练数据上学习空间映射的参数
    # spoc_valid = spoc.fit(cov_train).transform(cov_valid)
    # spoc_test = spoc.fit(cov_train).transform(cov_test)   #将训练数据和测试数据应用到之前学习到的映射参数上，从而将它们转换到切空间中。
    # # sc_train = spoc_train[:,0,:,:]
    # # sc_test  = spoc_test[:,0,:,:]
    # '''Dimensionality Reduction'''
    # sc_train = tangentspace_learning(spoc_train)
    # sc_valid = tangentspace_learning(spoc_valid)
    # sc_test = tangentspace_learning(spoc_test)
    #
    # spoc_train = np.squeeze(spoc_train, axis=1)
    # spoc_valid = np.squeeze(spoc_valid, axis=1)
    # spoc_test = np.squeeze(spoc_test, axis=1)
    model1 = E2R(1)
    spoc_train = model1(train_data)
    spoc_valid = model1(valid_data)
    spoc_test = model1(test_data)
    print('spoc_train', spoc_train.shape)

    spoc_train = np.squeeze(spoc_train, axis=1)
    spoc_valid = np.squeeze(spoc_valid, axis=1)
    spoc_test = np.squeeze(spoc_test, axis=1)

    # 保存测试数据
    np.save(f'./DTU/brainMap/test_data{name}.npy', test_data)
    np.save(f'./DTU/brainMap/spoc_test{name}.npy', spoc_test)
    np.save(f'./DTU/brainMap/test_label{name}.npy', test_label)

    train_loader = DataLoader(dataset=CustomDatasets(train_data, spoc_train, train_label),
                              batch_size=args.batch_size, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(dataset=CustomDatasets(valid_data, spoc_valid, valid_label),
                             batch_size=args.batch_size, drop_last=True, pin_memory=True)
    test_loader = DataLoader(dataset=CustomDatasets(test_data, spoc_test, test_label),
                             batch_size=args.batch_size, drop_last=True, pin_memory=True)

    # train_loader = DataLoader(dataset=CustomDatasets(sc_train, train_label),
    #                           batch_size=args.batch_size, drop_last=True, pin_memory=True)
    # valid_loader = DataLoader(dataset=CustomDatasets(sc_valid, valid_label),
    #                          batch_size=args.batch_size, drop_last=True, pin_memory=True)
    # test_loader = DataLoader(dataset=CustomDatasets(sc_test, test_label),
    #                          batch_size=args.batch_size, drop_last=True, pin_memory=True)

    ##--------------------------------------


    # train_loader = DataLoader(dataset=CustomDatasets(train_data, train_label),
    #                           batch_size=args.batch_size, drop_last=True, pin_memory=True)
    # valid_loader = DataLoader(dataset=CustomDatasets(valid_data, valid_label),
    #                           batch_size=args.batch_size, drop_last=True, pin_memory=True)
    # test_loader = DataLoader(dataset=CustomDatasets(test_data, test_label),
    #                          batch_size=args.batch_size, drop_last=True, pin_memory=True)
    return train_loader, valid_loader, test_loader

def get_KUL_data(name="S1", time_len=1, data_document_path="E:/听觉注意力/KUL/preprocessed_data"):
    class CustomDatasets(Dataset):
        # initialization: data and label
        def __init__(self, data, adj, label):
            self.data = torch.Tensor(data)
            self.adj = torch.Tensor(adj)
            self.label = torch.tensor(label, dtype=torch.uint8)

        # get the size of data
        def __len__(self):
            return len(self.label)

        # get the data and label
        def __getitem__(self, index):
            return self.data[index], self.adj[index], self.label[index]

    def read_prepared_data(args):
        data = []
        target = []
        # for l in range(len(args.ConType)):
        #     label = pd.read_csv(args.data_document_path + "/csv/" + args.name + args.ConType[l] + ".csv")
        #
        #     for k in range(args.trail_number):
        #         filename = args.data_document_path + "/" + args.ConType[l] + "/" + args.name + "Tra" + str(
        #             k + 1) + ".csv"
        #         # KUL_single_single3,contype=no,name=s1,len(arg.ConType)=1
        #         data_pf = pd.read_csv(filename, header=None)
        #         eeg_data = data_pf.iloc[:, 2:]  # KUL,DTU
        #
        #
        #         data.append(eeg_data)
        #         target.append(label.iloc[k, args.label_col])
        # for m in range(len(labels)):
        #     eeg = eeg_datas[m]
        #     label = labels[m]
        #     windows = []
        #     new_label = []
        #     for i in range(0, eeg.shape[0] - window_size + 1, stride):
        #         window = eeg[i:i + window_size, :]
        #         windows.append(window)
        #         new_label.append(label)
        #     train_eeg.append(np.array(windows)[:int(len(windows) * 0.9)])
        #     test_eeg.append(np.array(windows)[int(len(windows) * 0.9):])
        #     train_label.append(np.array(new_label)[:int(len(windows) * 0.9)])
        #     test_label.append(np.array(new_label)[int(len(windows) * 0.9):])
        # train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, eeg_channel)
        # test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, eeg_channel)
        # train_label = np.stack(train_label, axis=0).reshape(-1, 1)
        # test_label = np.stack(test_label, axis=0).reshape(-1, 1)
        file_name = f"{args.name}.mat"
        file_path = os.path.join(args.data_document_path, file_name)
        # all_sessions_data, all_labels = load_and_segment_eeg_data(file_path)
        eegdata = sio.loadmat(file_path)
        trials = eegdata['preproc_trials']
        # all_sessions_data = []
        # all_labels = []
        for i in range(8):
            preproc_trials = trials[0][i]
            # 检查 raw_data 的结构
            # print("raw_data structure:", preproc_trials)
            # 从 RawData 字典中获取 EegData 字段
            raw_data = preproc_trials['RawData']
            # print(raw_data)
            trial_data = raw_data[0][0]['EegData']
            trial_data = trial_data.item()
            trial_label = preproc_trials['attended_ear'].item()
            if trial_label == 'R':
                trial_label = 1
            elif trial_label == 'L':
                trial_label = 0

            target.append(trial_label)
            data.append(trial_data)
        target = np.array(target)
        target = target.reshape(-1, 1)
            # num_samples = trial_data.shape[0]
            # print('num_samples', num_samples)
            # samples_per_segment = fs * window_size
            # num_segments = num_samples // samples_per_segment
            # print(num_segments)
            # segmented_data = np.array([
            #     trial_data[i * samples_per_segment:(i + 1) * samples_per_segment, :]
            #     for i in range(num_segments)
            # ])
            # print(segmented_data)
            # labels = np.full(num_segments, trial_label)
            # # print(labels)
            # labels = labels.reshape(num_segments, 1)
            # all_sessions_data.append(segmented_data)
            # all_labels.append(labels)


        return data, target

    def sliding_window(eeg_datas, labels, args, eeg_channel):
        window_size = args.window_length
        stride = int(window_size * (1 - args.overlap))
        print("----------------")

        train_eeg = []
        test_eeg = []
        train_label = []
        test_label = []

        for m in range(len(labels)):
            eeg = eeg_datas[m]
            label = labels[m]
            windows = []
            new_label = []
            for i in range(0, eeg.shape[0] - window_size + 1, stride):
                # print(i)
                window = eeg[i:i + window_size, :]
                windows.append(window)
                new_label.append(label)
            train_eeg.append(np.array(windows)[:int(len(windows) * 1)])
            # test_eeg.append(np.array(windows)[int(len(windows) * 0.9):])
            train_label.append(np.array(new_label)[:int(len(windows) * 1)])
            # test_label.append(np.array(new_label)[int(len(windows) * 0.9):])
        train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, eeg_channel)
        # test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, eeg_channel)
        train_label = np.stack(train_label, axis=0).reshape(-1, 1)
        # test_label = np.stack(test_label, axis=0).reshape(-1, 1)
        return train_eeg, train_label
        # return train_eeg,test_eeg, train_label,test_label

    print("Num GPUs Available: ", torch.cuda.is_available())
    print(name)
    args = DotMap()
    args.name = name
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 128
    args.window_length = math.ceil(args.fs * time_len)
    args.overlap = 0.5
    args.batch_size = 32
    args.max_epoch = 200
    args.random_seed = 1234
    args.image_size = 32
    args.people_number = 16
    args.eeg_channel = 64
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 8
    args.cell_number = 46080
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.log_interval = 20
    args.label_col = 0
    args.alpha_low = 1
    args.alpha_high = 50
    args.log_path = "result"
    args.frequency_resolution = args.fs / args.window_length
    args.point_low = math.ceil(args.alpha_low / args.frequency_resolution)
    args.point_high = math.ceil(args.alpha_high / args.frequency_resolution) + 1
    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)
    args.csp_comp = 64
    stft_para = {
        'stftn': 128,
        'fStart': [1],  # 频带起始频率
        'fEnd': [32],  # 频带结束频率
        'window': 1,  # 窗口长度为1秒
        'fs': 128  # 采样率
    }
    # load data 和 label

    eeg_data, event_data = read_prepared_data(args)
    # 遍历列表，并对每个数组进行切片
    eeg_data_sliced = [data[0:49792, :] for data in eeg_data]

    data = np.vstack(eeg_data_sliced)
    eeg_data = data.reshape([args.trail_number, -1, args.eeg_channel])
    event_data = np.vstack(event_data)
    print("eeg_data.shape---",eeg_data.shape)

    #-------------------------------------------------------------------------
    # eeg_data1 = eeg_data.transpose(0, 2, 1)
    # event_data1 = np.squeeze(event_data - 1)
    # csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
    #           norm_trace=True)
    # eeg_data1 = csp.fit_transform(eeg_data1, event_data1)
    # eeg_data1 = eeg_data1.transpose(0, 2, 1)

    # -------------------------------------------------------------------------
    # train_data, test_data, train_label, test_label = sliding_window(eeg_data, event_data, args, args.csp_comp)
    train_data, train_label = sliding_window(eeg_data, event_data, args, args.csp_comp)
    # set the number of training, testing and validating data
    args.n_test = int(len(train_label)*0.1)
    args.n_valid = args.n_test
    args.n_train = len(train_label) - args.n_test*2

    print(1, data.shape)
    print("len of test_label", args.n_test, len(train_label))
    del data

    print(train_data.shape, 5)
    train_data = train_data.transpose(0, 2, 1)
    # test_data = test_data.transpose(0, 2, 1)

    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data, train_label = train_data[indices], train_label[indices]

    print(args.n_train, args.n_valid)
    valid_data, valid_label = train_data[args.n_train:args.n_train+args.n_valid], train_label[args.n_train:args.n_train+args.n_valid]
    test_data, test_label = train_data[args.n_train+args.n_valid:], train_label[args.n_train+args.n_valid:]
    train_data, train_label = train_data[:args.n_train], train_label[:args.n_train]
    # set the number of training, testing and validating data
    # args.n_test = len(test_label)
    # args.n_valid = args.n_test
    # args.n_train = len(train_label) - args.n_test
    # print(args.n_train,args.n_valid , args.n_test)
    #
    # train_data = train_data.transpose(0, 2, 1)
    # test_data = test_data.transpose(0, 2, 1)
    #
    # indices = np.arange(train_data.shape[0])
    # np.random.shuffle(indices)
    # train_data, train_label = train_data[indices], train_label[indices]
    #
    # valid_data, valid_label = train_data[args.n_train:], train_label[args.n_train:]
    # train_data, train_label = train_data[:args.n_train], train_label[:args.n_train]

    ##--------------------------------------流形空间修改测试
    # cov_train = Covariances('oas').transform(train_data)     ##协方差矩阵    oas重叠平均分割:将时间序列数据分割成重叠的片段，然后计算每个片段的协方差矩阵。
    # cov_train = cov_train[:, None, :, :]
    # cov_valid = Covariances('oas').transform(valid_data)
    # cov_valid = cov_valid[:, None, :, :]
    # cov_test = Covariances('oas').transform(test_data)
    # cov_test = cov_test[:, None, :, :]
    # #
    # spoc = ProjCommonSpace(rank_num=64)      #空间映射的对象
    # # spoc = ProjSPoCSpace(rank_num=rank_num, scale='auto')
    #
    # spoc_train = spoc.fit(cov_train).transform(cov_train)# fit 在训练数据上学习空间映射的参数
    # spoc_valid = spoc.fit(cov_train).transform(cov_valid)
    # spoc_test = spoc.fit(cov_train).transform(cov_test)   #将训练数据和测试数据应用到之前学习到的映射参数上，从而将它们转换到切空间中。
    # # sc_train = spoc_train[:,0,:,:]
    # # sc_test  = spoc_test[:,0,:,:]
    # # '''Dimensionality Reduction'''
    # # sc_train = tangentspace_learning(spoc_train)
    # # sc_valid = tangentspace_learning(spoc_valid)
    # # sc_test = tangentspace_learning(spoc_test)
    # #
    # spoc_train = np.squeeze(spoc_train, axis=1)
    # spoc_valid = np.squeeze(spoc_valid, axis=1)
    # spoc_test = np.squeeze(spoc_test, axis=1)
    #
    model1 = E2R(1)

    # model2 = adjweight()
    spoc_train = model1(train_data)
    spoc_valid = model1(valid_data)
    spoc_test = model1(test_data)

    print('spoc_train', spoc_train.shape)

    spoc_train = np.squeeze(spoc_train, axis=1)
    spoc_valid = np.squeeze(spoc_valid, axis=1)
    spoc_test = np.squeeze(spoc_test, axis=1)

    # spoc_train = model2(spoc_train)
    # spoc_valid = model2(spoc_valid)
    # spoc_test = model2(spoc_test)

    # model2 = SPDTangentSpace(8)
    # sc_train = model2(spoc_train)
    # sc_valid = model2(spoc_valid)
    # sc_test = model2(spoc_test)
    #
    # train_data_de = []
    # valid_data_de = []
    # test_data_de = []
    # for i in range(train_data.shape[0]):
    #     psd, train_data_de1 = DE_PSD(train_data[i], stft_para)
    #     train_data_de.append(train_data_de1)
    # train_data_de = [torch.from_numpy(item) for item in train_data_de]
    # train_data_de = torch.stack(train_data_de, dim=0)
    # for i in range(valid_data.shape[0]):
    #     psd, valid_data_de1 = DE_PSD(valid_data[i], stft_para)
    #     valid_data_de.append(valid_data_de1)
    # valid_data_de = [torch.from_numpy(item) for item in valid_data_de]
    # valid_data_de = torch.stack(valid_data_de, dim=0)
    # for i in range(test_data.shape[0]):
    #     psd, test_data_de1 = DE_PSD(test_data[i], stft_para)
    #     test_data_de.append(test_data_de1)
    # test_data_de = [torch.from_numpy(item) for item in test_data_de]
    # test_data_de = torch.stack(test_data_de, dim=0)



    # train_loader = DataLoader(dataset=CustomDatasets(train_data, spoc_train, train_label),
    #                           batch_size=args.batch_size, drop_last=True, pin_memory=True)
    # valid_loader = DataLoader(dataset=CustomDatasets(valid_data, spoc_valid, valid_label),
    #                          batch_size=args.batch_size, drop_last=True, pin_memory=True)
    # test_loader = DataLoader(dataset=CustomDatasets(test_data, spoc_test, test_label),
    #                          batch_size=args.batch_size, drop_last=True, pin_memory=True)
    # print(train_data.shape)

    train_loader = DataLoader(dataset=CustomDatasets(train_data, spoc_train, train_label),
                              batch_size=args.batch_size, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(dataset=CustomDatasets(valid_data, spoc_valid, valid_label),
                             batch_size=args.batch_size, drop_last=True, pin_memory=True)
    test_loader = DataLoader(dataset=CustomDatasets(test_data, spoc_test, test_label),
                             batch_size=args.batch_size, drop_last=True, pin_memory=True)
    # 保存测试数据
    np.save(f'./pre_trained_models/test_data{name}.npy', test_data)
    np.save(f'./pre_trained_models/spoc_test{name}.npy', spoc_test)
    np.save(f'./pre_trained_models/test_label{name}.npy', test_label)
    # ##--------------------------------------
    # train_loader = DataLoader(dataset=CustomDatasets(train_data, train_label),
    #                           batch_size=args.batch_size, drop_last=True, pin_memory=True)
    # valid_loader = DataLoader(dataset=CustomDatasets(valid_data, valid_label),
    #                           batch_size=args.batch_size, drop_last=True, pin_memory=True)
    # test_loader = DataLoader(dataset=CustomDatasets(test_data, test_label),
    #                          batch_size=args.batch_size, drop_last=True, pin_memory=True)
    return train_loader, valid_loader, test_loader


def get_all_KUL_data(name="S1", time_len=1, data_document_path="E:/听觉注意力/KUL/preprocessed_data"):
    class CustomDatasets(Dataset):
        # initialization: data and label
        def __init__(self, data, adj, label):
            self.data = torch.Tensor(data)
            self.adj = torch.Tensor(adj)
            self.label = torch.tensor(label, dtype=torch.uint8)

        # get the size of data
        def __len__(self):
            return len(self.label)

        # get the data and label
        def __getitem__(self, index):
            return self.data[index], self.adj[index], self.label[index]

    def read_prepared_data(args):
        data = []
        target = []
        for i in range(1, 17):
            print(i)
            name = 'S' + str(i)
            file_name = f"{name}.mat"
            file_path = os.path.join(args.data_document_path, file_name)
            eegdata = sio.loadmat(file_path)
            trials = eegdata['preproc_trials']
            for i in range(8):
                preproc_trials = trials[0][i]
                # 检查 raw_data 的结构
                # print("raw_data structure:", preproc_trials)
                # 从 RawData 字典中获取 EegData 字段
                raw_data = preproc_trials['RawData']
                # print(raw_data)
                trial_data = raw_data[0][0]['EegData']
                trial_data = trial_data.item()
                trial_label = preproc_trials['attended_ear'].item()
                if trial_label == 'R':
                    trial_label = 1
                elif trial_label == 'L':
                    trial_label = 0
                # print(trial_label)
                target.append(trial_label)
                data.append(trial_data)
        target = np.array(target)
        target = target.reshape(-1, 1)

        return data, target

    def sliding_window(eeg_datas, labels, args, eeg_channel):
        window_size = args.window_length
        stride = int(window_size * (1 - args.overlap))

        train_eeg = []
        test_eeg = []
        train_label = []
        test_label = []

        for m in range(len(labels)):
            eeg = eeg_datas[m]
            label = labels[m]
            windows = []
            new_label = []
            for i in range(0, eeg.shape[0] - window_size + 1, stride):
                window = eeg[i:i + window_size, :]
                windows.append(window)
                new_label.append(label)
            train_eeg.append(np.array(windows)[:int(len(windows) * 1)])
            test_eeg.append(np.array(windows)[int(len(windows) * 1):])
            train_label.append(np.array(new_label)[:int(len(windows) * 1)])
            test_label.append(np.array(new_label)[int(len(windows) * 1):])

        train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, eeg_channel)
        # test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, eeg_channel)
        train_label = np.stack(train_label, axis=0).reshape(-1, 1)
        # test_label = np.stack(test_label, axis=0).reshape(-1, 1)

        return train_eeg, train_label

    print("Num GPUs Available: ", torch.cuda.is_available())
    print(name)
    args = DotMap()
    args.name = name
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 128
    args.window_length = math.ceil(args.fs * time_len)
    args.overlap = 0.5
    args.batch_size = 16
    args.max_epoch = 200
    args.random_seed = 1234
    args.image_size = 32
    args.people_number = 16
    args.eeg_channel = 64
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 128
    args.cell_number = 46080
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.log_interval = 20
    args.label_col = 0
    args.alpha_low = 1
    args.alpha_high = 50
    args.log_path = "result"
    args.frequency_resolution = args.fs / args.window_length
    args.point_low = math.ceil(args.alpha_low / args.frequency_resolution)
    args.point_high = math.ceil(args.alpha_high / args.frequency_resolution) + 1
    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)
    args.csp_comp = 64
    eeg_data, event_data = read_prepared_data(args)

    # 遍历列表，并对每个数组进行切片
    eeg_data_sliced = [data[0:49792, :] for data in eeg_data]

    data = np.vstack(eeg_data_sliced)
    eeg_data = data.reshape([args.trail_number, -1, args.eeg_channel])
    event_data = np.vstack(event_data)
    print("eeg_data.shape---",eeg_data.shape)

    number = int(name[1:])
    test_data, test_label = sliding_window(eeg_data[number*8-8:number*8, :, :], event_data[number*8-8:number*8, :], args, args.csp_comp)
    eeg_data1 = np.concatenate((eeg_data[:number*8-8, :, :], eeg_data[number*8:, :, :]), axis=0)
    event_data1 = np.concatenate((event_data[:number*8-8, :], event_data[number*8:, :]), axis=0)
    train_data, train_label = sliding_window(eeg_data1, event_data1, args, args.csp_comp)

    # train_data, test_data, train_label, test_label = sliding_window(eeg_data, event_data, args, args.csp_comp)

    # set the number of training, testing and validating data
    args.n_test = len(test_label)
    args.n_valid = args.n_test
    args.n_train = len(train_label) - args.n_test


    print("len of test_label", len(test_label), len(train_label))
    del data


    train_data = train_data.transpose(0, 2, 1)
    test_data = test_data.transpose(0, 2, 1)

    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data, train_label = train_data[indices], train_label[indices]


    valid_data, valid_label = test_data, test_label
    # train_data, train_label =

    model1 = E2R(1)
    spoc_train = model1(train_data)
    spoc_valid = model1(valid_data)
    spoc_test = model1(test_data)
    print('spoc_train', spoc_train.shape)

    spoc_train = np.squeeze(spoc_train, axis=1)
    spoc_valid = np.squeeze(spoc_valid, axis=1)
    spoc_test = np.squeeze(spoc_test, axis=1)


    train_loader = DataLoader(dataset=CustomDatasets(train_data, spoc_train, train_label),
                              batch_size=args.batch_size, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(dataset=CustomDatasets(valid_data, spoc_valid, valid_label),
                              batch_size=args.batch_size, drop_last=True, pin_memory=True)
    test_loader = DataLoader(dataset=CustomDatasets(test_data, spoc_test, test_label),
                             batch_size=args.batch_size, drop_last=True, pin_memory=True)


    return train_loader, valid_loader, test_loader

