from easydict import EasyDict
import pickle
from trainer import trainer

def single_exp(config):
    config = EasyDict(config)
    print(config)

    data = pickle.load(open('datasets/{}/LOO_data/train_test.pkl'.format(config.data_name), 'rb'))
    config.num_items_s = data['num_items_s']
    config.num_items_t = data['num_items_t']
    config.num_users = data['num_users']

    exp_trainner = trainer(config, data)
    exp_trainner.train()

def run_smaple_data():
    config = {
        "data_name": "smaple_data",
        "mode": 'test',
        "shuffle": True,

        "cuda": True,
        "gpu_device": 0,
        "seed": 1024,
        "topk": 5,

        "batch_size": 1024,
        "learning_rate_gen": 0.001,
        "learning_rate_dis": 0.0001,
        "max_epoch": 500, # total number of epoch
        "epochs_G": 1,
        "epochs_D": 1,

        "CCA_weight": 0.0,
        "CCA_D_weight": 0.0,

        "hidden_size_gen_s": 128,
        "hidden_size_gen_s_d": 128,
        "hidden_size_gen_s_s": 128,
        "hidden_size_dis_s": 64,

        "hidden_size_gen_t": 128,
        "hidden_size_gen_t_d": 128,
        "hidden_size_gen_t_s": 128,
        "hidden_size_dis_t": 64,

        "hidden_size_cca": 64,

        "ZR_ratio": 10,
        "ZP_ratio": 10,
        "ZR_weight": 0.2,

        "max_grad_norm": 50,
        "init_std": 0.2,

        "optimizer": 'adam',
        "epsilon": 0.1,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.1,

    }
    single_exp(config)



if __name__ == '__main__':
    run_smaple_data()