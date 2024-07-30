import torch
import torch.nn as nn


class CCA(nn.Module):
    '''
    DiCURCDR DCCURL
    '''
    def __init__(self, config):
        super(CCA, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & config.cuda
        if self.cuda:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cuda")
            # torch.cuda.set_device(config.gpu_device)
        else:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cpu")

        self.num_users = config.num_users
        self.hidden_size_gen_s = config.hidden_size_gen_s
        self.hidden_size_gen_t = config.hidden_size_gen_t
        # self.hidden_size_gen_s_s = config.hidden_size_gen_s_s
        # self.hidden_size_gen_t_s = config.hidden_size_gen_t_s
        # self.hidden_size_gen_s_d = config.hidden_size_gen_s_d
        # self.hidden_size_gen_t_d = config.hidden_size_gen_t_d
        self.hidden_size_cca = config.hidden_size_cca

        # self.G = torch.Tensor(self.num_users, self.hidden_size_cca).to(self.device)
        # self.G_s = torch.Tensor(self.num_users, self.hidden_size_cca).to(self.device)
        # self.G_t = torch.Tensor(self.num_users, self.hidden_size_cca).to(self.device)

        self.init_std = config.init_std
        # nn.init.normal_(self.G, mean=0, std=self.init_std)
        # nn.init.normal_(self.G_s, mean=0, std=self.init_std)
        # nn.init.normal_(self.G_t, mean=0, std=self.init_std)

        self.Ws = nn.Linear(self.hidden_size_gen_s, self.hidden_size_cca, bias=True)
        self.Wt = nn.Linear(self.hidden_size_gen_t, self.hidden_size_cca, bias=True)
        # self.Ws_s = nn.Linear(self.hidden_size_gen_s_s, self.hidden_size_cca, bias=True)
        # self.Wt_s = nn.Linear(self.hidden_size_gen_t_s, self.hidden_size_cca, bias=True)
        # self.Ws_d = nn.Linear(self.hidden_size_gen_s_d, self.hidden_size_cca, bias=True)
        # self.Wt_d = nn.Linear(self.hidden_size_gen_t_d, self.hidden_size_cca, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, hidden_gen_s_s, hidden_gen_s_d, hidden_gen_t_s, hidden_gen_t_d, start, end):

        Ws_gen_s_s = self.leakyrelu(self.Ws(hidden_gen_s_s))
        Wt_gen_t_s = self.leakyrelu(self.Wt(hidden_gen_t_s))

        Ws_gen_s_d = self.leakyrelu(self.Ws(hidden_gen_s_d))
        Wt_gen_t_d = self.leakyrelu(self.Wt(hidden_gen_t_d))

        # G = self.G[start:end]
        # G_s = self.G_s[start:end]
        # G_t = self.G_t[start:end]

        # return Ws_gen_s_s, Wt_gen_t_s, Ws_gen_s_d, Wt_gen_t_d, G, G_s, G_t
        return Ws_gen_s_s, Wt_gen_t_s, Ws_gen_s_d, Wt_gen_t_d






class Generator_s(nn.Module):
    '''
        generator of source domain
        '''

    def __init__(self, config):
        super(Generator_s, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & config.cuda
        if self.cuda:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cuda")
            # torch.cuda.set_device(config.gpu_device)
        else:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cpu")

        self.num_users = config.num_users
        self.num_items_s = config.num_items_s
        self.hidden_size_gen_s_d = config.hidden_size_gen_s_d
        self.hidden_size_gen_s_s = config.hidden_size_gen_s_s

        self.encoder_gen_s_d = nn.Linear(self.num_items_s, self.hidden_size_gen_s_d, bias=True)
        self.encoder_gen_s_s = nn.Linear(self.num_items_s, self.hidden_size_gen_s_s, bias=True)

        self.decoder_gen_s = nn.Linear(self.hidden_size_gen_s_d + self.hidden_size_gen_s_s, self.num_items_s, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()


    def forward(self, data_s):

        batch_size = data_s.size(0)

        data_s = data_s.squeeze(1)

        hidden_gen_s_d = self.sigmoid(self.encoder_gen_s_d(data_s))
        hidden_gen_s_s = self.sigmoid(self.encoder_gen_s_s(data_s))

        hidden_gen_s = torch.concat((hidden_gen_s_d, hidden_gen_s_s), dim=1)

        # batch_pred = self.sigmoid(self.decoder_gen_s(hidden_gen_s))
        batch_pred = self.decoder_gen_s(hidden_gen_s)

        return hidden_gen_s_s, hidden_gen_s_d, batch_pred



class Generator_t(nn.Module):
    '''
        generator of target domain
        '''

    def __init__(self, config):
        super(Generator_t, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & config.cuda
        if self.cuda:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cuda")
            # torch.cuda.set_device(config.gpu_device)
        else:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cpu")

        self.num_users = config.num_users
        self.num_items_t = config.num_items_t
        self.hidden_size_gen_t_d = config.hidden_size_gen_t_d
        self.hidden_size_gen_t_s = config.hidden_size_gen_t_s

        self.encoder_gen_t_d = nn.Linear(self.num_items_t, self.hidden_size_gen_t_d, bias=True)
        self.encoder_gen_t_s = nn.Linear(self.num_items_t, self.hidden_size_gen_t_s, bias=True)

        self.decoder_gen_t = nn.Linear(self.hidden_size_gen_t_d + self.hidden_size_gen_t_s, self.num_items_t, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, data_t):

        batch_size = data_t.size(0)
        data_t = data_t.squeeze(1)

        hidden_gen_t_d = self.sigmoid(self.encoder_gen_t_d(data_t))
        hidden_gen_t_s = self.sigmoid(self.encoder_gen_t_s(data_t))

        hidden_gen_t = torch.concat((hidden_gen_t_d, hidden_gen_t_s), dim=1)

        # batch_pred = self.sigmoid(self.decoder_gen_t(hidden_gen_t))
        batch_pred = self.decoder_gen_t(hidden_gen_t)

        return hidden_gen_t_s, hidden_gen_t_d, batch_pred



class Discriminator_s(nn.Module):
    '''
        generator of source domain
        '''

    def __init__(self, config):
        super(Discriminator_s, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & config.cuda
        if self.cuda:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cuda")
            # torch.cuda.set_device(config.gpu_device)
        else:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cpu")

        self.num_users = config.num_users
        self.num_items_s = config.num_items_s
        self.hidden_size_dis_s = config.hidden_size_dis_s

        self.encoder_dis_s = nn.Linear(self.num_items_s * 2, self.hidden_size_dis_s, bias=True)

        self.decoder_dis_s = nn.Linear(self.hidden_size_dis_s, 1, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()


    def forward(self, data_s):

        batch_size = data_s.size(0)

        hidden_dis_s = self.sigmoid(self.encoder_dis_s(data_s))

        batch_pred = self.sigmoid(self.decoder_dis_s(hidden_dis_s))

        return batch_pred



class Discriminator_t(nn.Module):
    '''
        discriminator of target domain
        '''

    def __init__(self, config):
        super(Discriminator_t, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & config.cuda
        if self.cuda:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cuda")
            # torch.cuda.set_device(config.gpu_device)
        else:
            torch.cuda.manual_seed(config.seed)
            self.device = torch.device("cpu")

        self.num_users = config.num_users
        self.num_items_t = config.num_items_t
        self.hidden_size_dis_t = config.hidden_size_dis_t

        self.encoder_dis_t = nn.Linear(self.num_items_t * 2, self.hidden_size_dis_t, bias=True)

        self.decoder_dis_t = nn.Linear(self.hidden_size_dis_t, 1, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, data_t):

        batch_size = data_t.size(0)

        hidden_dis_t = self.sigmoid(self.encoder_dis_t(data_t))

        batch_pred = self.sigmoid(self.decoder_dis_t(hidden_dis_t))

        return batch_pred
