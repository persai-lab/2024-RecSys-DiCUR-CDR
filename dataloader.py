import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import TensorDataset
import more_itertools as miter
import matplotlib.pyplot as plt
import seaborn as sns

class DiCURCDR_DataLoader:
    def __init__(self, config, data):
        self.data_name = config['data_name']
        self.batch_size = config["batch_size"]
        self.shuffle = config["shuffle"]
        self.collate_fn = default_collate

        self.num_users = config.num_users
        self.num_items_s = config.num_items_s
        self.num_items_t = config.num_items_t

        self.seed = config['seed']
        self.mode = config["mode"]

        self.generate_train_test_data(data)
        self.unobserved_s = []
        self.unobserved_t = []
        for uid in range(self.num_users):
            self.unobserved_s.append(list(np.where(self.train_data_s[uid] == 0)[1]))
            self.unobserved_t.append(list(np.where(self.train_data_t[uid] == 0)[1]))

        self.data = TensorDataset(torch.Tensor(self.train_data_s).float(),
                                  torch.Tensor(self.train_data_t).float(),
                                  torch.Tensor(self.test_data_s).long(),
                                  torch.Tensor(self.test_data_t).long(),
                                  torch.Tensor(self.negative_data_s).long(),
                                  torch.Tensor(self.negative_data_t).long(),
                                  torch.Tensor(self.mask_data_s).bool(),
                                  torch.Tensor(self.mask_data_t).bool())

        # self.test_data = TensorDataset(torch.Tensor(self.test_data_q).long(),
        self.data_loader = DataLoader(self.data, batch_size=self.batch_size)


    def generate_train_test_data(self, data):
        negative_s = data['negative_s']
        negative_t = data['negative_t']

        if self.mode == 'test':
            trainset_s = data['trainset_s']
            trainset_t = data['trainset_t']
            testset_s = data['testset_s']
            testset_t = data['testset_t']
        elif self.mode == 'validation':
            temp_trainset_s = data['trainset_s']
            temp_trainset_t = data['trainset_t']
            trainset_s, trainset_t, testset_s, testset_t = {}, {}, {}, {}
            for uid, iids in temp_trainset_s.items():
                trainset_s[uid] = iids[:-1]
                trainset_t[uid] = temp_trainset_t[uid][:-1]
                testset_s[uid] = iids[-1]
                testset_s[uid] = temp_trainset_t[uid][-1]

        else:
            raise AttributeError('Undefined mode type')

        self.train_data_s, self.train_data_t, \
        self.test_data_s, self.test_data_t, \
        self.negative_data_s, self.negative_data_t = self.train_test_convert(
            trainset_s, trainset_t, testset_s, testset_t, negative_s, negative_t)

        self.mask_data_s = np.copy(self.train_data_s)
        self.mask_data_t = np.copy(self.train_data_t)


    def train_test_convert(self, trainset_s, trainset_t, testset_s, testset_t, negative_s, negative_t):
        train_data_s, train_data_t = [], []
        test_data_s, test_data_t = [], []
        negative_data_s, negative_data_t = [], []
        for uid, iids in trainset_s.items():
            train_s = np.zeros((1,self.num_items_s))
            train_s[0, iids] = 1

            train_t = np.zeros((1,self.num_items_t))
            iids_t = trainset_t[uid]
            train_t[0, iids_t] = 1

            test_s = testset_s[uid]
            test_t = testset_t[uid]

            nega_s = np.array(negative_s[uid])
            nega_t = np.array(negative_t[uid])

            train_data_s.append(train_s)
            train_data_t.append(train_t)
            test_data_s.append(test_s)
            test_data_t.append(test_t)
            negative_data_s.append(nega_s)
            negative_data_t.append(nega_t)


        return np.array(train_data_s), np.array(train_data_t), np.array(test_data_s), np.array(test_data_t), np.array(
            negative_data_s), np.array(negative_data_t)
