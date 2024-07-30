import logging
import time
import numpy as np
from torch.backends import cudnn
import torch
from torch import nn
import warnings
from model import CCA, Generator_s, Generator_t, Discriminator_s, Discriminator_t
from dataloader import DiCURCDR_DataLoader

warnings.filterwarnings("ignore")
cudnn.benchmark = True


class trainer(object):
    def __init__(self, config, data):
        super(trainer, self).__init__()
        self.config = config
        self.logger = logging.getLogger("trainer")

        self.mode = config.mode
        self.manual_seed = config.seed
        self.device = torch.device("cpu")

        self.current_epoch = 1
        self.current_iteration = 1

        self.data_loader = DiCURCDR_DataLoader(config, data)

        self.current_epoch = 1
        self.current_iteration = 1

        # HR, NDCG, the larger the better.
        self.best_hr_s = 0.
        self.best_hr_t = 0.
        self.best_ndcg_s = 0.
        self.best_ndcg_t = 0.
        self.best_mrr_s = 0.
        self.best_mrr_t = 0.

        # create empty list to store losses and testing evaluation metrics of each epoch
        self.train_loss_gen_s_list = []
        self.train_loss_gen_t_list = []
        self.train_loss_dis_s_list = []
        self.train_loss_dis_t_list = []
        self.train_loss_cca_list = []
        self.test_loss_s_list = []
        self.test_loss_t_list = []
        self.test_hr_s_list = []
        self.test_hr_t_list = []
        self.test_ndcg_s_list = []
        self.test_ndcg_t_list = []
        self.test_mrr_s_list = []
        self.test_mrr_t_list = []

        self.numOfMinibatches = int(self.config.num_users / self.config.batch_size) + 1
        self.numOfLastMinibatch = self.config.num_users % self.config.batch_size

        # build models
        self.model_CCA = CCA(config)
        self.model_Gen_s = Generator_s(config)
        self.model_Gen_t = Generator_t(config)
        self.model_Dis_s = Discriminator_s(config)
        self.model_Dis_t = Discriminator_t(config)

        # define criterion
        self.criterion = nn.BCELoss(reduction='sum')
        self.criterion_MSE = nn.MSELoss(reduction='mean')
        if config.optimizer == "sgd":
            self.optimizer_gen_s = torch.optim.SGD(self.model_Gen_s.parameters(),
                                                   lr=self.config.learning_rate_gen,
                                                   momentum=self.config.momentum,
                                                   weight_decay=self.config.weight_decay)
            self.optimizer_gen_t = torch.optim.SGD(self.model_Gen_t.parameters(),
                                                   lr=self.config.learning_rate_gen,
                                                   momentum=self.config.momentum,
                                                   weight_decay=self.config.weight_decay)
            self.optimizer_CCA = torch.optim.SGD(self.model_CCA.parameters(),
                                                 lr=self.config.learning_rate_gen,
                                                 momentum=self.config.momentum,
                                                 weight_decay=self.config.weight_decay)
            self.optimizer_dis_s = torch.optim.SGD(self.model_Dis_s.parameters(),
                                                   lr=self.config.learning_rate_dis,
                                                   momentum=self.config.momentum,
                                                   weight_decay=self.config.weight_decay)
            self.optimizer_dis_t = torch.optim.SGD(self.model_Dis_t.parameters(),
                                                   lr=self.config.learning_rate_dis,
                                                   momentum=self.config.momentum,
                                                   weight_decay=self.config.weight_decay)
        elif config.optimizer == "adam":
            self.optimizer_gen_s = torch.optim.Adam(self.model_Gen_s.parameters(),
                                                    lr=self.config.learning_rate_gen,
                                                    betas=(config.beta1, config.beta2),
                                                    eps=self.config.epsilon,
                                                    weight_decay=self.config.weight_decay)
            self.optimizer_gen_t = torch.optim.Adam(self.model_Gen_t.parameters(),
                                                    lr=self.config.learning_rate_gen,
                                                    betas=(config.beta1, config.beta2),
                                                    eps=self.config.epsilon,
                                                    weight_decay=self.config.weight_decay)
            self.optimizer_CCA = torch.optim.Adam(self.model_CCA.parameters(),
                                                  lr=self.config.learning_rate_gen,
                                                  betas=(config.beta1, config.beta2),
                                                  eps=self.config.epsilon,
                                                  weight_decay=self.config.weight_decay)
            self.optimizer_dis_s = torch.optim.Adam(self.model_Dis_s.parameters(),
                                                    lr=self.config.learning_rate_dis,
                                                    betas=(config.beta1, config.beta2),
                                                    eps=self.config.epsilon,
                                                    weight_decay=self.config.weight_decay)
            self.optimizer_dis_t = torch.optim.Adam(self.model_Dis_t.parameters(),
                                                    lr=self.config.learning_rate_dis,
                                                    betas=(config.beta1, config.beta2),
                                                    eps=self.config.epsilon,
                                                    weight_decay=self.config.weight_decay)

        self.scheduler_gen_s = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_gen_s,
            mode='min',
            patience=0,
            min_lr=1e-5,
            factor=0.5,
            verbose=True
        )

        self.scheduler_gen_t = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_gen_t,
            mode='min',
            patience=0,
            min_lr=1e-5,
            factor=0.5,
            verbose=True
        )

        self.scheduler_cca = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_CCA,
            mode='min',
            patience=0,
            min_lr=1e-5,
            factor=0.5,
            verbose=True
        )

        self.scheduler_dis_s = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_dis_s,
            mode='min',
            patience=0,
            min_lr=1e-5,
            factor=0.5,
            verbose=True
        )

        self.scheduler_dis_t = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_dis_t,
            mode='min',
            patience=0,
            min_lr=1e-5,
            factor=0.5,
            verbose=True
        )

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")
            print("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            # torch.cuda.set_device(self.config.gpu_device)
            self.model_CCA = self.model_CCA.to(self.device)
            self.model_Gen_s = self.model_Gen_s.to(self.device)
            self.model_Gen_t = self.model_Gen_t.to(self.device)
            self.model_Dis_s = self.model_Dis_s.to(self.device)
            self.model_Dis_t = self.model_Dis_t.to(self.device)
            self.criterion = self.criterion.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print("Program will run on *****GPU-CUDA***** ")
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")
            print("Program will run on *****CPU*****\n")

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(1, self.config.max_epoch + 1):
            self.sample_ZR_s, self.sample_ZP_s = [], []
            self.sample_ZR_t, self.sample_ZP_t = [], []

            seed = int(time.time())
            np.random.seed(seed)
            for uid in range(self.config.num_users):
                self.sample_ZR_s.append(np.random.choice(self.data_loader.unobserved_s[uid], int(len(
                    self.data_loader.unobserved_s[uid]) * self.config.ZR_ratio / 100),
                                                         replace=False))
                self.sample_ZP_s.append(np.random.choice(self.data_loader.unobserved_s[uid], int(len(
                    self.data_loader.unobserved_s[uid]) * self.config.ZP_ratio / 100),
                                                         replace=False))

                self.sample_ZR_t.append(np.random.choice(self.data_loader.unobserved_t[uid], int(len(
                    self.data_loader.unobserved_t[uid]) * self.config.ZR_ratio / 100),
                                                         replace=False))
                self.sample_ZP_t.append(np.random.choice(self.data_loader.unobserved_t[uid], int(len(
                    self.data_loader.unobserved_t[uid]) * self.config.ZP_ratio / 100),
                                                         replace=False))

            print("=" * 50 + "Epoch {}".format(epoch) + "=" * 50)
            print("=" * 25 + "Source Domain".format(epoch) + "=" * 25)
            self.train_one_epoch_dis_s()
            self.train_one_epoch_gen_s()

            print("+" * 25 + "Target Domain".format(epoch) + "+" * 25)
            self.train_one_epoch_dis_t()
            self.train_one_epoch_gen_t()

            print("+" * 30 + "CCA".format(epoch) + "+" * 30)
            self.train_one_epoch_cca()

            print("*" * 25 + "Test Results".format(epoch) + "*" * 25)
            self.validate()  # perform validation or testing
            self.current_epoch += 1

        print("Best Results: {:.05}, {:.05}, {:.05}, {:.05}, {:.05}, {:.05}".format(self.best_hr_s, self.best_ndcg_s,
                                                                                    self.best_mrr_s, self.best_hr_t,
                                                                                    self.best_ndcg_t, self.best_mrr_t))

    def train_one_epoch_gen_s(self):
        """
               One epoch of training source domain generator
               :return:
               """
        self.model_Gen_s.train()
        self.model_Dis_s.eval()
        self.logger.info("\n")
        self.logger.info("Train Generator Source Epoch: {}".format(self.current_epoch))
        self.logger.info("learning rate: {}".format(self.optimizer_gen_s.param_groups[0]['lr']))
        self.train_loss_gen_s = 0
        # train_elements_gen_s = 0


        hidden_gen_s_s_all = np.array([]).reshape(0,self.config.hidden_size_gen_s_s)
        hidden_gen_s_d_all = np.array([]).reshape(0,self.config.hidden_size_gen_s_d)
        for batch_idx, data in enumerate(self.data_loader.data_loader):
            train_data_s, train_data_t, test_data_s, test_data_t, \
            negative_data_s, negative_data_t, mask_data_s, mask_data_t = data

            start = batch_idx * self.config.batch_size
            if batch_idx == self.numOfMinibatches - 1:  # if it is the last minibatch
                numOfBatches = self.numOfLastMinibatch
            else:
                numOfBatches = self.config.batch_size
            end = start + numOfBatches

            ZR_samples = self.sample_ZR_s[start:end]
            ZP_samples = self.sample_ZP_s[start:end]

            train_data_s = train_data_s.squeeze(1).to(self.device)
            train_data_t = train_data_t.squeeze(1).to(self.device)
            mask_data_s = mask_data_s.to(self.device)

            # ZR_mask = torch.zeros_like(mask_data_s).squeeze(1)
            ZR_mask = mask_data_s.squeeze(1).clone()
            ZP_mask = mask_data_s.squeeze(1).clone()
            for u in range(len(ZR_samples)):
                ZR_mask[u][ZR_samples[u]] = True
                ZP_mask[u][ZP_samples[u]] = True

            self.optimizer_gen_s.zero_grad()

            hidden_gen_s_s, hidden_gen_s_d, output_gen_s = self.model_Gen_s(train_data_s)

            fake_s_ZP = torch.concat((train_data_s, output_gen_s * ZP_mask), dim=1)
            output_dis_s_fake = self.model_Dis_s(fake_s_ZP)

            label = torch.ones_like(output_dis_s_fake).to(self.device)

            gen_s_ZP_loss =self.criterion(output_dis_s_fake.float(), label.float())

            # gen_s_ZR_loss = self.criterion_MSE(torch.masked_select(output_gen_s, ZR_mask), torch.masked_select(train_data_s, ZR_mask))
            # gen_s_ZR_loss = self.criterion_MSE(output_gen_s * ZR_mask, train_data_s*ZR_mask)

            gen_s_ZR_loss = torch.sum(torch.sum(((output_gen_s-train_data_s) ** 2) * ZR_mask, dim=1))

            gen_s_loss = gen_s_ZP_loss + self.config.ZR_weight * gen_s_ZR_loss
            # gen_s_loss = gen_s_ZP_loss + gen_s_ZR_loss
            # gen_s_loss = gen_s_ZR_loss

            # hidden_gen_t_s, hidden_gen_t_d, output_gen_t = self.model_Gen_t(train_data_t)
            # Ws_gen_s_s, Wt_gen_t_s, Ws_gen_s_d, Wt_gen_t_d, G, G_s, G_t = self.model_CCA(hidden_gen_s_s, hidden_gen_s_d,
            #                                                                              hidden_gen_t_s, hidden_gen_t_d,
            #                                                                              start, end)

            hidden_gen_t_s, hidden_gen_t_d, output_gen_t = self.model_Gen_t(train_data_t)
            Ws_gen_s_s, Wt_gen_t_s, Ws_gen_s_d, Wt_gen_t_d = self.model_CCA(hidden_gen_s_s, hidden_gen_s_d,
                                                                                         hidden_gen_t_s, hidden_gen_t_d,
                                                                                         start, end)

            # loss_CCA = self.criterion_MSE(Ws_gen_s_s, G) + self.criterion_MSE(Wt_gen_t_s, G)
            # loss_CCA_D = self.criterion_MSE(Ws_gen_s_d, G_s) + self.criterion_MSE(Wt_gen_t_d, G_t)
            loss_CCA = self.criterion_MSE(Ws_gen_s_s, Wt_gen_t_s)
                       # + self.criterion_MSE(Wt_gen_t_s, G)
            loss_CCA_D = - (self.criterion_MSE(Ws_gen_s_d, Ws_gen_s_s) + self.criterion_MSE(Wt_gen_t_d,
                                                                                            Wt_gen_t_s) + self.criterion_MSE(
                Ws_gen_s_d, Wt_gen_t_d))
            # gen_s_loss = gen_s_loss + loss_CCA
            gen_s_loss = gen_s_loss + self.config.CCA_weight * loss_CCA + self.config.CCA_D_weight * loss_CCA_D

            self.train_loss_gen_s += gen_s_loss.item()
            torch.nn.utils.clip_grad_norm_(self.model_Gen_s.parameters(), self.config.max_grad_norm)  # clip gradient to
            # avoid gradient vanishing or exploding
            gen_s_loss.backward()
            self.optimizer_gen_s.step()

            hidden_gen_s_s_all = np.vstack((hidden_gen_s_s_all, hidden_gen_s_s.detach()))
            hidden_gen_s_d_all = np.vstack((hidden_gen_s_d_all, hidden_gen_s_d.detach()))

        self.train_loss_gen_s = self.train_loss_gen_s/self.config.num_users
        # self.scheduler_gen_s.step(self.train_loss_gen_s)
        self.train_loss_gen_s_list.append(self.train_loss_gen_s)
        self.logger.info("Generator Source Train Loss: {:.6f}".format(self.train_loss_gen_s))
        self.hidden_gen_s_s = np.array(hidden_gen_s_s_all)
        self.hidden_gen_s_d = np.array(hidden_gen_s_d_all)
        print("Generator Source Train Loss: {:.6f}".format(self.train_loss_gen_s))

    def train_one_epoch_gen_t(self):
        """
               One epoch of training target domain generator
               :return:
               """
        self.model_Gen_t.train()
        self.model_Dis_t.eval()
        self.logger.info("\n")
        self.logger.info("Train Generator Target Epoch: {}".format(self.current_epoch))
        self.logger.info("learning rate: {}".format(self.optimizer_gen_t.param_groups[0]['lr']))
        self.train_loss_gen_t = 0
        # train_elements_gen_s = 0

        hidden_gen_t_s_all = np.array([]).reshape(0,self.config.hidden_size_gen_t_s)
        hidden_gen_t_d_all = np.array([]).reshape(0,self.config.hidden_size_gen_t_d)
        for batch_idx, data in enumerate(self.data_loader.data_loader):
            train_data_s, train_data_t, test_data_s, test_data_t, \
            negative_data_s, negative_data_t, mask_data_s, mask_data_t = data

            start = batch_idx * self.config.batch_size
            if batch_idx == self.numOfMinibatches - 1:  # if it is the last minibatch
                numOfBatches = self.numOfLastMinibatch
            else:
                numOfBatches = self.config.batch_size
            end = start + numOfBatches

            ZR_samples = self.sample_ZR_t[start:end]
            ZP_samples = self.sample_ZP_t[start:end]

            train_data_t = train_data_t.squeeze(1).to(self.device)
            train_data_s = train_data_s.squeeze(1).to(self.device)
            mask_data_t = mask_data_t.to(self.device)

            # ZR_mask = torch.zeros_like(mask_data_t).squeeze(1)
            ZR_mask = mask_data_t.squeeze(1).clone()
            ZP_mask = mask_data_t.squeeze(1).clone()
            for u in range(len(ZR_samples)):
                ZR_mask[u][ZR_samples[u]] = True
                ZP_mask[u][ZP_samples[u]] = True

            self.optimizer_gen_t.zero_grad()

            hidden_gen_t_s, hidden_gen_t_d, output_gen_t = self.model_Gen_t(train_data_t)

            fake_t_ZP = torch.concat((train_data_t, output_gen_t * ZP_mask), dim=1)
            output_dis_t_fake = self.model_Dis_t(fake_t_ZP)

            label = torch.ones_like(output_dis_t_fake).to(self.device)

            gen_t_ZP_loss = self.criterion(output_dis_t_fake.float(), label.float())

            # gen_t_ZR_loss = self.criterion_MSE(torch.masked_select(output_gen_t, ZR_mask), torch.masked_select(train_data_t, ZR_mask))
            # gen_t_ZR_loss = self.criterion_MSE(output_gen_t*ZR_mask, train_data_t * ZR_mask)

            gen_t_ZR_loss = torch.sum(torch.sum(((output_gen_t-train_data_t)**2) * ZR_mask, dim=1))

            gen_t_loss = gen_t_ZP_loss + self.config.ZR_weight * gen_t_ZR_loss
            # gen_t_loss = gen_t_ZP_loss + gen_t_ZR_loss
            # gen_t_loss = gen_t_ZR_loss

            hidden_gen_s_s, hidden_gen_s_d, output_gen_s = self.model_Gen_s(train_data_s)
            # Ws_gen_s_s, Wt_gen_t_s, Ws_gen_s_d, Wt_gen_t_d, G, G_s, G_t = self.model_CCA(hidden_gen_s_s, hidden_gen_s_d,
            #                                                                              hidden_gen_t_s, hidden_gen_t_d,
            #                                                                              start, end)
            Ws_gen_s_s, Wt_gen_t_s, Ws_gen_s_d, Wt_gen_t_d = self.model_CCA(hidden_gen_s_s, hidden_gen_s_d,
                                                                                         hidden_gen_t_s, hidden_gen_t_d,
                                                                                         start, end)

            # loss_CCA = self.criterion_MSE(Ws_gen_s_s, G) + self.criterion_MSE(Wt_gen_t_s, G)
            # loss_CCA_D = self.criterion_MSE(Ws_gen_s_d, G_s) + self.criterion_MSE(Wt_gen_t_d, G_t)
            loss_CCA = self.criterion_MSE(Ws_gen_s_s, Wt_gen_t_s)
            # + self.criterion_MSE(Wt_gen_t_s, G)
            loss_CCA_D = - (self.criterion_MSE(Ws_gen_s_d, Ws_gen_s_s) + self.criterion_MSE(Wt_gen_t_d,
                                                                                            Wt_gen_t_s) + self.criterion_MSE(
                Ws_gen_s_d, Wt_gen_t_d))

            # gen_t_loss = gen_t_loss + loss_CCA
            gen_t_loss = gen_t_loss + self.config.CCA_weight * loss_CCA + self.config.CCA_D_weight * loss_CCA_D

            self.train_loss_gen_t += gen_t_loss.item()
            torch.nn.utils.clip_grad_norm_(self.model_Gen_t.parameters(), self.config.max_grad_norm)  # clip gradient to
            # avoid gradient vanishing or exploding
            gen_t_loss.backward()
            self.optimizer_gen_t.step()

            hidden_gen_t_s_all = np.vstack((hidden_gen_t_s_all, hidden_gen_t_s.detach()))
            hidden_gen_t_d_all = np.vstack((hidden_gen_t_d_all, hidden_gen_t_d.detach()))

        self.train_loss_gen_t = self.train_loss_gen_t / self.config.num_users
        # self.scheduler_gen_t.step(self.train_loss_gen_t)
        self.train_loss_gen_t_list.append(self.train_loss_gen_t)
        self.logger.info("Generator Target Train Loss: {:.6f}".format(self.train_loss_gen_t))
        self.hidden_gen_t_s = np.array(hidden_gen_t_s_all)
        self.hidden_gen_t_d = np.array(hidden_gen_t_d_all)
        print("Generator Target Train Loss: {:.6f}".format(self.train_loss_gen_t))


    def train_one_epoch_cca(self):
        """
               One epoch of training source domain generator
               :return:
               """
        self.model_CCA.train()
        self.logger.info("\n")
        self.logger.info("Train CCA Epoch: {}".format(self.current_epoch))
        self.logger.info("learning rate: {}".format(self.optimizer_CCA.param_groups[0]['lr']))
        self.train_loss_CCA = 0
        self.train_loss_CCA_D = 0
        self.train_loss_CCA_all = 0


        for batch_idx, data in enumerate(self.data_loader.data_loader):
            train_data_s, train_data_t, test_data_s, test_data_t, \
            negative_data_s, negative_data_t, mask_data_s, mask_data_t = data

            start = batch_idx * self.config.batch_size
            if batch_idx == self.numOfMinibatches - 1:  # if it is the last minibatch
                numOfBatches = self.numOfLastMinibatch
            else:
                numOfBatches = self.config.batch_size
            end = start + numOfBatches

            train_data_t = train_data_t.to(self.device)
            train_data_s = train_data_s.to(self.device)

            hidden_gen_s_s, hidden_gen_s_d, output_gen_s = self.model_Gen_s(train_data_s)
            hidden_gen_t_s, hidden_gen_t_d, output_gen_t = self.model_Gen_t(train_data_t)
            # Ws_gen_s_s, Wt_gen_t_s, Ws_gen_s_d, Wt_gen_t_d, G, G_s, G_t = self.model_CCA(hidden_gen_s_s, hidden_gen_s_d,
            #                                                                              hidden_gen_t_s, hidden_gen_t_d,
            #                                                                              start, end)

            Ws_gen_s_s, Wt_gen_t_s, Ws_gen_s_d, Wt_gen_t_d = self.model_CCA(hidden_gen_s_s, hidden_gen_s_d,
                                                                                         hidden_gen_t_s, hidden_gen_t_d,
                                                                                         start, end)

            # loss_CCA = self.criterion_MSE(Ws_gen_s_s, G) + self.criterion_MSE(Wt_gen_t_s, G) + self.criterion_MSE(G.T @ G, torch.eye(G.shape[1]).to(self.device))
            # loss_CCA_D = self.criterion_MSE(Ws_gen_s_d, G_s) + self.criterion_MSE(Wt_gen_t_d, G_t)
            # # loss_CCA_D = loss_CCA_D + self.criterion_MSE(G_s.T @ G_s, torch.eye(G_s.shape[1]))+self.criterion_MSE(G_t.T @ G_t, torch.eye(G_t.shape[1]))
            # loss_CCA_D = loss_CCA_D - (self.criterion_MSE(G, G_s) + self.criterion_MSE(G, G_t) + self.criterion_MSE(G_s, G_t))

            loss_CCA = self.criterion_MSE(Ws_gen_s_s, Wt_gen_t_s) + self.criterion_MSE(Ws_gen_s_s.T @ Ws_gen_s_s,
                                                                                       torch.eye(
                                                                                           Ws_gen_s_s.shape[1]).to(
                                                                                           self.device)) + self.criterion_MSE(
                Wt_gen_t_s.T @ Wt_gen_t_s, torch.eye(Wt_gen_t_s.shape[1]).to(self.device))
            loss_CCA_D = - (self.criterion_MSE(Ws_gen_s_d, Ws_gen_s_s) + self.criterion_MSE(Wt_gen_t_d, Wt_gen_t_s) +  self.criterion_MSE(Ws_gen_s_d, Wt_gen_t_d))
            # loss_CCA_D = loss_CCA_D + self.criterion_MSE(G_s.T @ G_s, torch.eye(G_s.shape[1]))+self.criterion_MSE(G_t.T @ G_t, torch.eye(G_t.shape[1]))
            # loss_CCA_D = loss_CCA_D - (self.criterion_MSE(G, G_s) + self.criterion_MSE(G, G_t) + self.criterion_MSE(G_s, G_t))

            # loss = loss_CCA + loss_CCA_D
            loss = self.config.CCA_weight * loss_CCA + self.config.CCA_D_weight * loss_CCA_D

            self.train_loss_CCA_all += loss.item()
            self.train_loss_CCA += loss_CCA.item()
            self.train_loss_CCA_D += loss_CCA_D.item()
            torch.nn.utils.clip_grad_norm_(self.model_CCA.parameters(), self.config.max_grad_norm)  # clip gradient to
            # avoid gradient vanishing or exploding
            loss.backward(retain_graph=True)
            self.optimizer_CCA.step()

        self.train_loss_CCA_all = self.train_loss_CCA_all / self.config.num_users
        self.train_loss_CCA = self.train_loss_CCA / self.config.num_users
        self.train_loss_CCA_D = self.train_loss_CCA_D / self.config.num_users
        # self.scheduler_cca.step(self.train_loss_CCA_all)
        self.train_loss_cca_list.append(self.train_loss_CCA_all)
        self.logger.info("CCA all Loss: {:.6f}, CCA Loss: {:.6f}, CCA_D Loss {:.6f}:".format(self.train_loss_CCA_all, self.train_loss_CCA, self.train_loss_CCA_D))
        print("CCA all Loss: {:.6f}, CCA Loss: {:.6f}, CCA_D Loss {:.6f}:".format(self.train_loss_CCA_all, self.train_loss_CCA, self.train_loss_CCA_D))


    def train_one_epoch_dis_s(self):
        """
               One epoch of training source domain discriminator
               :return:
               """
        self.model_Dis_s.train()
        self.model_Gen_s.eval()
        self.logger.info("\n")
        self.logger.info("Train Discriminator Source Epoch: {}".format(self.current_epoch))
        self.logger.info("learning rate: {}".format(self.optimizer_dis_s.param_groups[0]['lr']))
        self.train_loss_dis_s = 0

        for batch_idx, data in enumerate(self.data_loader.data_loader):
            train_data_s, train_data_t, test_data_s, test_data_t, \
            negative_data_s, negative_data_t, mask_data_s, mask_data_t = data

            start = batch_idx * self.config.batch_size
            if batch_idx == self.numOfMinibatches - 1:  # if it is the last minibatch
                numOfBatches = self.numOfLastMinibatch
            else:
                numOfBatches = self.config.batch_size
            end = start + numOfBatches

            ZR_samples = self.sample_ZR_s[start:end]
            ZP_samples = self.sample_ZP_s[start:end]

            train_data_s = train_data_s.squeeze(1).to(self.device)
            train_data_t = train_data_t.to(self.device)
            mask_data_s = mask_data_s.to(self.device)

            ZP_mask = mask_data_s.squeeze(1).clone()
            for u in range(len(ZR_samples)):
                ZP_mask[u][ZP_samples[u]] = True

            self.optimizer_dis_s.zero_grad()

            hidden_gen_s_s, hidden_gen_s_d, output_gen_s = self.model_Gen_s(train_data_s)

            # fake_s = torch.concat((output_gen_s * mask_data_s.squeeze(1), train_data_s.squeeze(1)), dim=1)
            fake_s_ZP = torch.concat((train_data_s, output_gen_s * ZP_mask), dim=1)
            output_dis_s_fake = self.model_Dis_s(fake_s_ZP)
            real_s = torch.concat((train_data_s, train_data_s), dim=1)
            output_dis_s_real = self.model_Dis_s(real_s)

            label_fake = torch.zeros_like(output_dis_s_fake)
            label_real = torch.ones_like(output_dis_s_real)

            loss_fake = self.criterion(output_dis_s_fake.float(), label_fake.float())
            loss_real = self.criterion(output_dis_s_real.float(), label_real.float())

            dis_s_loss = loss_fake + loss_real

            self.train_loss_dis_s += dis_s_loss.item()
            torch.nn.utils.clip_grad_norm_(self.model_Dis_s.parameters(), self.config.max_grad_norm)  # clip gradient to
            # avoid gradient vanishing or exploding
            dis_s_loss.backward(retain_graph=True)
            self.optimizer_dis_s.step()

        self.train_loss_dis_s = self.train_loss_dis_s / (self.config.num_users*2)
        # self.scheduler_dis_s.step(self.train_loss_dis_s)
        self.train_loss_dis_s_list.append(self.train_loss_dis_s)
        self.logger.info("Discriminator Source Train Loss: {:.6f}".format(self.train_loss_dis_s))
        print("Discriminator Source Train Loss: {:.6f}".format(self.train_loss_dis_s))


    def train_one_epoch_dis_t(self):
        """
               One epoch of training target domain discriminator
               :return:
               """
        self.model_Dis_t.train()
        self.model_Gen_t.eval()
        self.logger.info("\n")
        self.logger.info("Train Discriminator Target Epoch: {}".format(self.current_epoch))
        self.logger.info("learning rate: {}".format(self.optimizer_dis_t.param_groups[0]['lr']))
        self.train_loss_dis_t = 0

        for batch_idx, data in enumerate(self.data_loader.data_loader):
            train_data_s, train_data_t, test_data_s, test_data_t, \
            negative_data_s, negative_data_t, mask_data_s, mask_data_t = data

            start = batch_idx * self.config.batch_size
            if batch_idx == self.numOfMinibatches - 1:  # if it is the last minibatch
                numOfBatches = self.numOfLastMinibatch
            else:
                numOfBatches = self.config.batch_size
            end = start + numOfBatches

            ZR_samples = self.sample_ZR_t[start:end]
            ZP_samples = self.sample_ZP_t[start:end]

            train_data_t = train_data_t.squeeze(1).to(self.device)
            train_data_s = train_data_s.to(self.device)
            mask_data_t = mask_data_t.to(self.device)

            ZP_mask = mask_data_t.squeeze(1).clone()
            for u in range(len(ZR_samples)):
                ZP_mask[u][ZP_samples[u]] = True

            self.optimizer_dis_t.zero_grad()

            hidden_gen_t_s, hidden_gen_t_d, output_gen_t = self.model_Gen_t(train_data_t)

            # fake_t = torch.concat((output_gen_t * mask_data_t.squeeze(1), train_data_t.squeeze(1)), dim=1)
            fake_t_ZP = torch.concat((train_data_t, output_gen_t * ZP_mask), dim=1)
            output_dis_t_fake = self.model_Dis_t(fake_t_ZP)
            real_t = torch.concat((train_data_t, train_data_t), dim=1)
            output_dis_t_real = self.model_Dis_t(real_t)

            label_fake = torch.zeros_like(output_dis_t_fake)
            label_real = torch.ones_like(output_dis_t_real)

            loss_fake = self.criterion(output_dis_t_fake.float(), label_fake.float())
            loss_real = self.criterion(output_dis_t_real.float(), label_real.float())

            dis_t_loss = loss_fake + loss_real

            self.train_loss_dis_t += dis_t_loss.item()
            torch.nn.utils.clip_grad_norm_(self.model_Dis_t.parameters(), self.config.max_grad_norm)  # clip gradient to
            # avoid gradient vanishing or exploding
            dis_t_loss.backward(retain_graph=True)
            self.optimizer_dis_t.step()

        self.train_loss_dis_t = self.train_loss_dis_t / (self.config.num_users * 2)
        # self.scheduler_dis_t.step(self.train_loss_dis_t)
        self.train_loss_dis_t_list.append(self.train_loss_dis_t)
        self.logger.info("Discriminator Target Train Loss: {:.6f}".format(self.train_loss_dis_t))
        print("Discriminator Target Train Loss: {:.6f}".format(self.train_loss_dis_t))


    def validate(self):
        """
           One epoch of validate model
           :return:
           """
        self.model_Gen_s.eval()
        self.model_Gen_t.eval()
        self.model_Dis_s.eval()
        self.model_Dis_t.eval()
        self.model_CCA.eval()
        self.logger.info("\n")
        self.logger.info("Test Result at Epoch: {}".format(self.current_epoch))


        hr_s_all, ndcg_s_all, mrr_s_all = [], [], []
        hr_t_all, ndcg_t_all, mrr_t_all = [], [], []

        with torch.no_grad():
            for batch_idx, data in enumerate(self.data_loader.data_loader):
                train_data_s, train_data_t, test_data_s, test_data_t, \
                negative_data_s, negative_data_t, mask_data_s, mask_data_t = data

                train_data_s = train_data_s.to(self.device)
                train_data_t = train_data_t.to(self.device)

                hidden_gen_s_s, hidden_gen_s_d, output_s = self.model_Gen_s(train_data_s)
                hidden_gen_t_s, hidden_gen_t_d, output_t = self.model_Gen_t(train_data_t)

                for i in range(len(output_s)):
                    pred_scores = [output_s[i][itm].item() for itm in negative_data_s[i]]
                    pred_scores.append(output_s[i][test_data_s[i]].item())

                    hr_s, ndcg_s, mrr_s = self.ranking_evaluation(pred_scores, self.config.topk)
                    hr_s_all.append(hr_s)
                    ndcg_s_all.append(ndcg_s)
                    mrr_s_all.append(mrr_s)

                for i in range(len(output_t)):
                    pred_scores = [output_t[i][itm].item() for itm in negative_data_t[i]]
                    pred_scores.append(output_t[i][test_data_t[i]].item())

                    hr_t, ndcg_t, mrr_t = self.ranking_evaluation(pred_scores, self.config.topk)
                    hr_t_all.append(hr_t)
                    ndcg_t_all.append(ndcg_t)
                    mrr_t_all.append(mrr_t)

        hr_s_all = np.mean(hr_s_all)
        ndcg_s_all = np.mean(ndcg_s_all)
        mrr_s_all = np.mean(mrr_s_all)

        hr_t_all = np.mean(hr_t_all)
        ndcg_t_all = np.mean(ndcg_t_all)
        mrr_t_all = np.mean(mrr_t_all)

        self.logger.info('HR_S: {:.05}'.format(hr_s_all))
        self.logger.info('NDCG_S: {:.05}'.format(ndcg_s_all))
        self.logger.info('MRR_S: {:.05}'.format(mrr_s_all))
        print('HR_S, NDCG_S, MRR_S: {:.05}, {:.05}, {:.05}'.format(hr_s_all, ndcg_s_all, mrr_s_all))

        self.logger.info('HR_T: {:.05}'.format(hr_t_all))
        self.logger.info('NDCG_S: {:.05}'.format(ndcg_t_all))
        self.logger.info('MRR_S: {:.05}'.format(mrr_t_all))
        print('HR_T, NDCG_T, MRR_T: {:.05}, {:.05}, {:.05}'.format(hr_t_all, ndcg_t_all, mrr_t_all))

        if hr_s_all > self.best_hr_s:
            self.best_hr_s = hr_s_all
        if ndcg_s_all > self.best_ndcg_s:
            self.best_ndcg_s = ndcg_s_all
        if mrr_s_all > self.best_mrr_s:
            self.best_mrr_s = mrr_s_all

        if hr_t_all > self.best_hr_t:
            self.best_hr_t = hr_t_all
        if ndcg_t_all > self.best_ndcg_t:
            self.best_ndcg_t = ndcg_t_all
        if mrr_t_all > self.best_mrr_t:
            self.best_mrr_t = mrr_t_all

        self.test_hr_s_list.append(hr_s_all)
        self.test_ndcg_s_list.append(ndcg_s_all)
        self.test_mrr_s_list.append(mrr_s_all)

        self.test_hr_t_list.append(hr_t_all)
        self.test_ndcg_t_list.append(ndcg_t_all)
        self.test_mrr_t_list.append(mrr_t_all)


    def ranking_evaluation(self, preds, topK):
        sort = np.argsort(preds)[::-1][:topK]
        hr_arr = 0
        ndcg_arr = 0
        mrr_arr = 0
        if 99 in sort:
            pos = np.where(sort == 99)[0][0]
            hr_arr = 1.0
            ndcg_arr = np.log(2) / np.log(pos + 2.0)
            mrr_arr = 1.0 / (pos + 1.0)
        return hr_arr, ndcg_arr, mrr_arr


