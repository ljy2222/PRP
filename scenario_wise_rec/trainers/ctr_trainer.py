import os
import time
import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, log_loss

from ..basic.callback import EarlyStopper


class CTRTrainer(object):
    def __init__(self, model, dataset_name, optimizer_fn=torch.optim.Adam, optimizer_params=None, scheduler_fn=None,
                 scheduler_params=None, n_epoch=10, earlystop_patience=10, model_path='./', args=None):
        self.model = model
        self.dataset_name = dataset_name
        self.device = torch.device(args.device)
        self.model.to(self.device)
        if optimizer_params is None:
            optimizer_params = {'lr': 1e-3, 'weight_decay': 1e-5}
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.criterion = torch.nn.BCELoss()
        self.n_epoch = n_epoch
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.model_path = model_path
        self.time_now = time.strftime('%m_%d_%H_%M', time.localtime(int(round(time.time() * 1000)) / 1000))
        self.writer = SummaryWriter(f'./logs/{self.time_now}')
        self.use_ot = args.use_ot
        self.use_independence_loss = args.use_independence_loss

    def train_one_epoch(self, data_loader, epoch_i, log_interval=10):
        self.model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(data_loader, desc='train', smoothing=0, mininterval=1.0)
        for i, (x_dict, y1, y2) in enumerate(tk0):
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
            y1, y2 = y1.to(self.device), y2.to(self.device)
            if self.use_ot or self.use_independence_loss:
                y_pred1, y_pred2, loss_dict = self.model(x_dict)
            else:
                y_pred1, y_pred2 = self.model(x_dict)
            loss1, loss2 = self.criterion(y_pred1, y1.float()), self.criterion(y_pred2, y2.float())
            self.model.zero_grad()
            loss = loss1 + loss2
            if self.use_ot:
                loss = loss + loss_dict['loss_c'] + loss_dict['loss_s'] + loss_dict['loss_c_2'] + loss_dict['loss_s_2']
            if self.use_independence_loss:
                loss = loss + loss_dict['loss_independence_1'] + loss_dict['loss_independence_2']
            loss.backward()

            loss_dict['loss_1'] = loss1
            loss_dict['loss_2'] = loss2
            # print(loss_dict)

            self.optimizer.step()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0
                self.writer.add_scalar('Loss/total_loss', loss.detach(), (epoch_i + 1) * (i + 1))
                self.writer.add_scalar('Loss/loss1', loss1.detach(), (epoch_i + 1) * (i + 1))
                self.writer.add_scalar('Loss/loss2', loss2.detach(), (epoch_i + 1) * (i + 1))
                if self.use_ot:
                    self.writer.add_scalar('Loss/loss_c', loss_dict['loss_c'].detach(), (epoch_i + 1) * (i + 1))
                    self.writer.add_scalar('Loss/loss_s', loss_dict['loss_s'].detach(), (epoch_i + 1) * (i + 1))
                    self.writer.add_scalar('Loss/loss_c_2', loss_dict['loss_c_2'].detach(), (epoch_i + 1) * (i + 1))
                    self.writer.add_scalar('Loss/loss_s_2', loss_dict['loss_s_2'].detach(), (epoch_i + 1) * (i + 1))
                if self.use_independence_loss:
                    self.writer.add_scalar('Loss/loss_independence_1', loss_dict['loss_independence_1'].detach(), (epoch_i + 1) * (i + 1))
                    self.writer.add_scalar('Loss/loss_independence_2', loss_dict['loss_independence_2'].detach(), (epoch_i + 1) * (i + 1))

    def fit(self, train_dataloader, val_dataloader=None):
        for epoch_i in range(self.n_epoch):
            print('epoch:', epoch_i)
            self.train_one_epoch(train_dataloader, epoch_i)
            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print('current lr: {}'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()
            if val_dataloader:
                auc1, logloss1, auc2, logloss2 = self.evaluate(self.model, val_dataloader)
                print(f'epoch: {epoch_i} | val auc1: {auc1} | val logloss1: {logloss1} | val auc2: {auc2} | val logloss2: {logloss2}')
                self.writer.add_scalar('Metric/auc1', auc1, (epoch_i + 1))
                self.writer.add_scalar('Metric/auc2', auc2, (epoch_i + 1))
                if self.early_stopper.stop_training((auc1 + auc2) / 2, self.model.state_dict()):
                    print(f'validation: best auc: {self.early_stopper.best_auc}')
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    break
        self.writer.close()
        name = self.model.__class__.__name__ + '_' + self.dataset_name + '_' + self.time_now + '.pth'
        torch.save(self.model.state_dict(), os.path.join(self.model_path, name))  # save best auc model

    def evaluate(self, model, data_loader):
        model.eval()
        targets1, targets2, predicts1, predicts2 = list(), list(), list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc='validation', smoothing=0, mininterval=1.0)
            for i, (x_dict, y1, y2) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y1, y2 = y1.to(self.device), y2.to(self.device)
                y_pred1, y_pred2 = model(x_dict, mode='eval')
                targets1.extend(y1.tolist())
                targets2.extend(y2.tolist())
                predicts1.extend(y_pred1.tolist())
                predicts2.extend(y_pred2.tolist())
        return roc_auc_score(targets1, predicts1), log_loss(targets1, predicts1), roc_auc_score(targets2, predicts2), log_loss(targets2, predicts2)

    def evaluate_multi_domain_loss(self, model, data_loader, domain_num):
        model.eval()
        targets_domain_specific_list1, targets_domain_specific_list2, predicts_domain_specific_list1, predicts_domain_specific_list2 = list(), list(), list(), list()
        for i in range(domain_num):
            targets_domain_specific_list1.append(list())
            targets_domain_specific_list2.append(list())
            predicts_domain_specific_list1.append(list())
            predicts_domain_specific_list2.append(list())

        targets_all_list1, targets_all_list2, predicts_all_list1, predicts_all_list2 = list(), list(), list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc='validation', smoothing=0, mininterval=1.0)
            for i, (x_dict, y1, y2) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                domain_id = x_dict['domain_indicator'].clone().detach()
                y1, y2 = y1.to(self.device), y2.to(self.device)
                y_pred1, y_pred2 = model(x_dict, mode='eval')
                targets_all_list1.extend(y1.tolist())
                targets_all_list2.extend(y2.tolist())
                predicts_all_list1.extend(y_pred1.tolist())
                predicts_all_list2.extend(y_pred2.tolist())
                for d in range(domain_num):
                    domain_mask_d = (domain_id == d)
                    targets_domain_specific_list1[d].extend(y1[domain_mask_d].tolist())
                    targets_domain_specific_list2[d].extend(y2[domain_mask_d].tolist())
                    predicts_domain_specific_list1[d].extend(y_pred1[domain_mask_d].tolist())
                    predicts_domain_specific_list2[d].extend(y_pred2[domain_mask_d].tolist())

        domain_auc_list1, domain_logloss_list1, domain_auc_list2, domain_logloss_list2 = list(), list(), list(), list()
        for d in range(domain_num):
            domain_auc_list1.append(roc_auc_score(targets_domain_specific_list1[d], predicts_domain_specific_list1[d]))
            domain_logloss_list1.append(log_loss(targets_domain_specific_list1[d], predicts_domain_specific_list1[d]))
            domain_auc_list2.append(roc_auc_score(targets_domain_specific_list2[d], predicts_domain_specific_list2[d]))
            domain_logloss_list2.append(log_loss(targets_domain_specific_list2[d], predicts_domain_specific_list2[d]))
        total_auc_val1 = roc_auc_score(targets_all_list1, predicts_all_list1)
        total_logloss_val1 = log_loss(targets_all_list1, predicts_all_list1)
        total_auc_val2 = roc_auc_score(targets_all_list2, predicts_all_list2)
        total_logloss_val2 = log_loss(targets_all_list2, predicts_all_list2)

        return domain_auc_list1, domain_logloss_list1, domain_auc_list2, domain_logloss_list2, total_auc_val1, total_logloss_val1, total_auc_val2, total_logloss_val2
