import sys
sys.path.append('..')
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from scenario_wise_rec.trainers import CTRTrainer
from scenario_wise_rec.utils.data import DataGenerator
from scenario_wise_rec.models.multi_domain import PRP
from scenario_wise_rec.basic.features import DenseFeature, SparseFeature

def convert_numeric(val):
    return int(val)


def get_kuairand_data_multidomain_multitask(data_path='./data/kuairand/'):
    # ========== load data ==========
    data = pd.read_csv(data_path + '/kuairand_multi_task_sample.csv')
    # data = pd.read_csv(data_path + '/kuairand_sample.csv')
    data = data[data['tab'].apply(lambda x: x in [1, 0, 2])]
    data.reset_index(drop=True, inplace=True)
    data.rename(columns={'tab': 'domain_indicator'}, inplace=True)

    # ========== feature names ==========
    col_names = data.columns.to_list()
    dense_features = ['follow_user_num', 'fans_user_num', 'friend_user_num', 'register_days']
    useless_features = ['play_time_ms', 'duration_ms', 'profile_stay_time', 'comment_stay_time']
    scenario_features = ['domain_indicator']
    target_features = ['is_click', 'long_view']
    sparse_features = [col for col in col_names if col not in dense_features and
                       col not in useless_features and col not in scenario_features and col not in target_features]

    # ========== feature process ==========
    for feature in dense_features:
        data[feature] = data[feature].apply(lambda x: convert_numeric(x))
    if dense_features:
        sca = MinMaxScaler()
        data[dense_features] = sca.fit_transform(data[dense_features])
    for feature in useless_features:
        del data[feature]
    for feature in scenario_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature])
    for feature in tqdm(sparse_features, desc='feature process'):
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature])
    dense_feas = [DenseFeature(feature_name) for feature_name in dense_features]
    sparse_feas = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=16) for feature_name in sparse_features]
    scenario_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in scenario_features]
    y1 = data[target_features[0]]
    y2 = data[target_features[1]]
    del data[target_features[0]]
    del data[target_features[1]]

    # ========== calculate domain_num and task_num ==========
    domain_num = data.domain_indicator.nunique()
    task_num = len(target_features)

    return dense_feas, sparse_feas, scenario_feas, data, y1, y2, domain_num, task_num


def main(args):
    # ========== feature process ==========
    torch.manual_seed(args.seed)
    dense_feas, sparse_feas, scenario_feas, x, y1, y2, domain_num, task_num = get_kuairand_data_multidomain_multitask(args.dataset_path)
    dg = DataGenerator(x, y1, y2)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(split_ratio=[0.8, 0.1], batch_size=args.batch_size)
    print(f'the samples of train : val : test are  {len(train_dataloader)} : {len(val_dataloader)} : {len(test_dataloader)}')

    # ========== train ==========
    model = PRP(
        features=dense_feas + sparse_feas,
        domain_num=domain_num,
        task_num=task_num,
        expert_num=domain_num,
        expert_params={'dims': [32]}, tower_params={'dims': [16]},
        args=args
    )
    # for name, param in model.named_parameters():
    #     print(name)

    ctr_trainer = CTRTrainer(
        model=model,
        dataset_name=args.dataset_name,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        scheduler_fn=None,  # torch.optim.lr_scheduler.StepLR
        scheduler_params={'step_size': args.step_size, 'gamma': args.gamma},
        n_epoch=args.epoch,
        earlystop_patience=args.patience,
        model_path=args.save_dir,
        args=args
    )
    ctr_trainer.fit(train_dataloader, val_dataloader)

    # ========== test ==========
    domain_auc1, domain_logloss1, domain_auc2, domain_logloss2, auc1, logloss1, auc2, logloss2 = \
        ctr_trainer.evaluate_multi_domain_loss(ctr_trainer.model, test_dataloader, domain_num)
    for d in range(domain_num):
        for t in range(task_num):
            print(f'========== domain: {d}, task: {t} ==========')
            if t == 0:  # task 1
                print(f'auc: {domain_auc1[d]} | logloss: {domain_logloss1[d]}')
            else:  # task 2
                print(f'auc: {domain_auc2[d]} | logloss: {domain_logloss2[d]}')
    print(f'test auc1: {auc1} | test logloss1: {logloss1} | test auc2: {auc2} | test logloss2: {logloss2}')


if __name__ == '__main__':
    # ========== data config ==========
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--dataset_name', type=str, default='Kuairand')
    parser.add_argument('--dataset_path', type=str, default='./data/kuairand')
    parser.add_argument('--batch_size', type=int, default=4096)  # 5

    # ========== train config ==========
    parser.add_argument('--model_name', type=str, default='PRP')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--step_size', type=int, default=4)  # 2
    parser.add_argument('--gamma', type=float, default=0.75)  # 0.8
    parser.add_argument('--epoch', type=int, default=5)  # 100, 20
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')  # cuda:0, cpu
    parser.add_argument('--save_dir', type=str, default='./model_weights/')

    # ========== model config ==========
    parser.add_argument('--use_ot', type=bool, default=True)
    parser.add_argument('--use_independence_loss', type=bool, default=True)

    # ========== main ==========
    args = parser.parse_args()
    main(args)

"""
conda activate benchmark
cd /home/ljy/Scenario-Wise-Rec/scripts
CUDA_VISIBLE_DEVICES=3 python run_prp_kuairand.py --model_name PRP
tensorboard --logdir=./logs --port 6006
"""