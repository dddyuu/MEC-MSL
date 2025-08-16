import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
import os
import matplotlib.pyplot  as plt
from graph.layers import GraphConvolution,Linear
from matplotlib.pyplot import MultipleLocator


def stgblsp(A):
    zor = torch.zeros_like(A)
    torch.sign(A) * torch.maximum(abs(A) - 1 / 2 * 3e-4 * 1e-5, zor)
    return A
def normalize_A(A, symmetry=False,gaowei =False):
    A = F.relu(A)
    if symmetry:
        if gaowei:
            A = A + A.permute(0,2,1)
            d = torch.sum(A, 1)
            d = 1 / torch.sqrt(d + 1e-10)
            D = torch.diag_embed(d)
            L = torch.matmul(torch.matmul(D, A), D)
        else:
            A = A + torch.transpose(A,0,1)
            d = torch.sum(A, 1)
            d = 1 / torch.sqrt(d + 1e-10)
            D = torch.diag_embed(d)
            L = torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        # print("D",D.shape)
        # print("A",A.shape)
        L = torch.matmul(torch.matmul(D, A), D)

    return L

def is_symmetric(matrix, tol=1e-5):
    """
    Check if a matrix is symmetric.
    Args:
        matrix (torch.Tensor): Input matrix.
        tol (float): Tolerance for symmetry check.
    Returns:
        bool: True if the matrix is symmetric, False otherwise.
    """
    return torch.allclose(matrix, matrix.T, atol=tol)
def generate_cheby_adj(A, K):
    support = []


    for i in range(K):
        if i == 0:
            support.append(torch.eye(A.shape[1]).cuda())
        elif i == 1:
            support.append(A)
        else:
            temp = torch.matmul(support[-1], A)
            # temp = (temp + temp.T) / 2  # 对称化处理
            support.append(temp)
            # for i, adj in enumerate(temp):
            #     print(f"Matrix {i} is symmetric:", is_symmetric(adj))

    return support

def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.transpose(1,2))
    cov2 = torch.matmul(emb2, emb2.transpose(1,2))
    cost = torch.mean((cov1 - cov2)**2)
    return cost

def graph_coarsening(x):
    #print(x.shape)
    emb1 = torch.mean(x[:,0:5],1).unsqueeze(dim=1) #x[:,0:5]:提取x第二维的下标维0-4的数据
    emb2 = torch.mean(x[:,5:8],1).unsqueeze(dim=1)
    emb3 = torch.mean(torch.cat([x[:,8:11],x[:,17:20]],1),1).unsqueeze(dim=1)
    emb4 = torch.mean(x[:,11:14],1).unsqueeze(dim=1)
    emb5 = torch.mean(x[:,14:17],1).unsqueeze(dim=1)
    emb6 = torch.mean(x[:,20:23],1).unsqueeze(dim=1)
    emb7 = torch.mean(x[:,23:26],1).unsqueeze(dim=1)
    emb8 = torch.mean(x[:,26:29],1).unsqueeze(dim=1)
    emb9 = torch.mean(x[:,29:32],1).unsqueeze(dim=1)
    emb10 = torch.mean(x[:,32:35],1).unsqueeze(dim=1)
    emb11 = torch.mean(torch.cat([x[:,35:38],x[:,44:47]],1),1).unsqueeze(dim=1)
    emb12 = torch.mean(x[:,38:41],1).unsqueeze(dim=1)
    emb13 = torch.mean(x[:,41:44],1).unsqueeze(dim=1)
    emb14 = torch.mean(x[:,47:50],1).unsqueeze(dim=1)
    #print(x[50:52].shape,x[57].shape)
    emb15 = torch.mean(torch.cat([x[:,50:52],x[:,57:58]],1),1).unsqueeze(dim=1)
    emb16 = torch.mean(torch.cat([x[:,52:55],x[:,58:61]],1),1).unsqueeze(dim=1)
    emb17 = torch.mean(torch.cat([x[:,55:57],x[:,61:62]],1),1).unsqueeze(dim=1)
    #print(torch.cat([x[:,55:57],x[:,61:62]],1).shape)
    #print(emb1.shape,emb2.shape,emb3.shape,emb4.shape,emb5.shape,emb6.shape,emb7.shape,emb8.shape,emb9.shape,emb10.shape,emb11.shape,emb12.shape,emb13.shape,emb14.shape,emb15.shape,emb16.shape,emb17.shape)
    x = torch.cat([emb1,emb2,emb3,emb4,emb5,emb6,emb7,emb8,emb9,emb10,emb11,emb12,emb13,emb14,emb15,emb16,emb17],1)
    #print(x.shape)
    return x
def loss_dependence(emb1, emb2, dim):
    sum = 0
    R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
    K1 = torch.matmul(emb1, emb1.transpose(1,2))
    K2 = torch.matmul(emb2, emb2.transpose(1,2))
    #print(K1.shape[0],K1.shape[1],K1.shape[2])
    RK1 = torch.matmul(R, K1)
    RK2 = torch.matmul(R, K2)
    MRK = torch.matmul(RK1, RK2)
    #HSIC = torch.trace(MRK,dim=0)
    #print(torch.trace(MRK[0]))
    #print(RK1.shape[0],RK1.shape[1],RK1.shape[2])
    for i in range(len(MRK)):
        sum += torch.trace(MRK[i])
    HSIC = sum/len(MRK)
    #HSIC =torch.trace(MRK[:,])
    return HSIC

def get_labels(label_path):
    '''
        得到15个 trials 对应的标签
    :param label_path: 标签文件对应的路径
    :return: list，对应15个 trials 的标签，1 for positive, 0 for neutral, -1 for negative
    '''
    return scio.loadmat(label_path, verify_compressed_data_integrity=False)['label'][0]


def label_2_onehot(label_list):
    '''
        将原始-1， 0， 1标签转化为独热码形式
    :param label_list: 原始标签列表
    :return label_onehot: 独热码形式标签列表
    '''

    look_up_table = {-1: [1, 0, 0],
                     0: [0, 1, 0],
                     1: [0, 0, 1]}
    label_onehot = [look_up_table[label] for label in label_list]
    return label_onehot


def get_frequency_band_idx(frequency_band):
    '''
        获得频带对应的索引，仅对 ExtractedFeatures 目录下的数据有效
    :param frequency_band: 频带名称，'delta', 'theta', 'alpha', 'beta', 'gamma'
    :return idx: 频带对应的索引
    '''
    lookup = {'delta': 0,
              'theta': 1,
              'alpha': 2,
              'beta': 3,
              'gamma': 4}
    return lookup[frequency_band]


def build_extracted_features_dataset(folder_path, feature_name):
    '''
        将 folder_path 文件夹中的 ExtractedFeatures 数据转化为机器学习常用的数据集，区分开不同 trial 的数据
        ToDo: 增加 channel 的选择，而不是使用所有的 channel
    :param folder_path: ExtractedFeatures 文件夹对应的路径
    :param feature_name: 需要使用的特征名，如 'de_LDS'，'asm_LDS' 等，以 de_LDS1 为例，维度为 62 * 235 * 5，235为影片长度235秒，每秒切分为一个样本，62为通道数，5为频带数
    :param frequency_band: 需要选取的频带，'delta', 'theta', 'alpha', 'beta', 'gamma'
    :return feature_vector_dict, label_dict: 分别为样本的特征向量，样本的标签，key 为被试名字，val 为该被试对应的特征向量或标签的 list，方便 subject-independent 的测试
    '''
    #frequency_idx = get_frequency_band_idx(frequency_band)
    labels = get_labels(os.path.join(folder_path, 'label.mat'))
    labels = labels+1
    feature_vector_dict = {}
    label_dict = {}
    try:
        all_mat_file = os.walk(folder_path)
        skip_set = {'label.mat', 'readme.txt'}
        file_cnt = 0
        for path, dir_list, file_list in all_mat_file:
            for file_name in file_list:
                file_cnt += 1
                print('当前已处理到{}，总进度{}/{}'.format(file_name, file_cnt, len(file_list)))
                if file_name not in skip_set:
                    all_features_dict = scio.loadmat(os.path.join(folder_path, file_name),
                                                     verify_compressed_data_integrity=False)
                    subject_name = file_name.split('.')[0]
                    feature_vector_trial_dict = {}
                    label_trial_dict = {}
                    for trials in range(1, 16):
                        feature_vector_list = []
                        label_list = []
                        cur_feature = all_features_dict[feature_name + str(trials)]
                        cur_feature = np.asarray(cur_feature[:, :]).T  # 转置后，维度为 N * 62, N 为影片长度
                        feature_vector_list.extend(_ for _ in cur_feature)
                        for _ in range(len(cur_feature)):
                            label_list.append(labels[trials - 1])
                        feature_vector_trial_dict[str(trials)] = feature_vector_list
                        label_trial_dict[str(trials)] = label_list
                    feature_vector_dict[subject_name] = feature_vector_trial_dict
                    label_dict[subject_name] = label_trial_dict
                else:
                    continue
    except FileNotFoundError as e:
        print('加载数据时出错: {}'.format(e))

    return feature_vector_dict, label_dict



def subject_independent_data_split(feature_vector_dict, label_dict, test_subject_set):
    '''
        使用 subject_independent 的方式做数据切分
    :param feature_vector_dict: build_preprocessed_eeg_dataset_CNN 函数返回的 feature_vector_dict
    :param label_dict: build_preprocessed_eeg_dataset_CNN 函数返回的 label_dict
    :param test_subject_set: 留一法，用作测试集的 subject
    :return train_feature, train_label, test_feature, test_label: 训练特征，训练标签，测试特征，测试标签
    '''
    train_feature = []
    train_label = []
    test_feature = []
    test_label = []
    for experiment in feature_vector_dict.keys():
        subject = experiment.split('_')[0]
        for trial in feature_vector_dict[experiment].keys():
            if subject in test_subject_set:
                test_feature.extend(feature_vector_dict[experiment][trial])
                test_label.extend(label_dict[experiment][trial])
            else:
                train_feature.extend(feature_vector_dict[experiment][trial])
                train_label.extend(label_dict[experiment][trial])
    return train_feature, train_label, test_feature, test_label

def draw_fig(list,name,epoch):
    # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
    x1 = range(1, epoch+1)
    print(x1)
    y1 = list
    if name=="loss":
        plt.cla()
        plt.figure(figsize=(100, 100))
        plt.title('Test loss vs. epoch', fontsize=5)
        plt.plot(x1, y1,)
        plt.xlabel('epoch', fontsize=5)
        plt.ylabel('Test loss', fontsize=5)
        plt.grid()
       # plt.savefig("./lossAndacc/Train_loss.png")
        plt.show()
    elif name =="acc":
        plt.cla()
        plt.figure(figsize=(100, 100))
        plt.title('Test accuracy vs. epoch', fontsize=5)
        plt.plot(x1, y1,)
        plt.xlabel('epoch', fontsize=5)
        plt.ylabel('Test accuracy', fontsize=5)
        plt.grid()
        #plt.savefig("./lossAndacc/Train _accuracy.png")
        plt.show()

class GraphUnet(nn.Module):

    def __init__(self, ks, dim, act, drop_p):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.bottom_gcn = GCN(dim, dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim, act, drop_p))
            self.up_gcns.append(GCN(dim, dim, act, drop_p))
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))

    def forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        org_h = h
        for i in range(self.l_n):
            h = self.down_gcns[i](g, h)
            adj_ms.append(g)
            down_outs.append(h)
            g, h, idx = self.pools[i](g, h)
            indices_list.append(idx)
        h = self.bottom_gcn(g, h)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, idx = adj_ms[up_idx], indices_list[up_idx]
            g, h = self.unpools[i](g, h, down_outs[up_idx], idx)
            h = self.up_gcns[i](g, h)
            h = h.add(down_outs[up_idx])
            hs.append(h)
        h = h.add(org_h)
        hs.append(h)
        return hs


class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h):
        h = self.drop(h)
        h = torch.matmul(h,g)
        h = self.proj(h)
        h = self.act(h)
        return h


class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h


def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx




def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g

class Initializer(object):

    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)