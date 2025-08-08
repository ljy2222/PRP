import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss

from ...basic.layers import MLP, EmbeddingLayer
"""
To prevent premature disclosure of the code, we have hidden a portion of it. However, if the paper is accepted, we will make the complete code publicly available.
"""

class PRP(nn.Module):
    def __init__(self, features, domain_num, task_num, expert_num, expert_params, tower_params, args):
        super(PRP, self).__init__()
        # ========== model config ==========
        self.features = features
        self.domain_num = domain_num
        self.task_num = task_num
        self.expert_num = expert_num
        self.embedding = EmbeddingLayer(features)
        self.input_dims = sum([fea.embed_dim for fea in features])
        self.device = args.device
        self.K = 8
        self.loss_dict = {}
        self.use_ot = args.use_ot
        self.use_independence_loss = args.use_independence_loss
        self.lambda_1 = 0.01

        # ========== s-prp: commonality ==========
        self.experts_c = nn.ModuleList(
            MLP(self.input_dims, output_layer=False, **expert_params) for _ in range(self.expert_num)
        )
        self.gate_c = MLP(self.input_dims, output_layer=False, **{'dims': [self.expert_num], 'activation': 'softmax'})

        # ========== s-prp: specificity ==========
        self.experts_s = nn.ModuleList(
            MLP(self.input_dims, output_layer=False, **expert_params) for _ in range(self.expert_num)
        )

        # ========== t-prp: commonality ==========
        self.experts_c_2 = nn.ModuleList(
            MLP(expert_params['dims'][-1] * 2, output_layer=False, **expert_params) for _ in range(self.expert_num)
        )
        self.gate_c_2 = MLP(expert_params['dims'][-1] * 2, output_layer=False, **{'dims': [self.expert_num], 'activation': 'softmax'})

        # ========== t-prp: specificity ==========
        self.experts_s_2 = nn.ModuleList(
            MLP(expert_params['dims'][-1] * 2, output_layer=False, **expert_params) for _ in range(self.expert_num)
        )

        # ========== towers ==========
        self.towers = nn.ModuleList(
            MLP(expert_params['dims'][-1] * 2, **tower_params) for _ in range(self.task_num)
        )

        # ========== domain_ot ==========
        self.domain_prototypes = nn.ModuleList(
            Prototype(self.K, expert_params['dims'][-1]) for _ in range(self.domain_num + 1)
        )
        self.domain_alphas = nn.ModuleList(
            MLP(expert_params['dims'][-1], output_layer=False, **{'dims': [self.K], 'activation': 'softmax'}) for _ in range(self.domain_num + 1)
        )

        # ========== task_ot ==========
        self.task_prototypes = nn.ModuleList(
            Prototype(self.K, expert_params['dims'][-1]) for _ in range(self.task_num + 1)
        )
        self.task_alphas = nn.ModuleList(
            MLP(expert_params['dims'][-1], output_layer=False, **{'dims': [self.K], 'activation': 'softmax'}) for _ in range(self.task_num + 1)
        )

    def optimal_transport(self, instance, prototype, alpha):
        instance = instance.to(self.device)  # [B, D]
        prototype = prototype.to(self.device)  # [K, D]
        
        loss_func = SamplesLoss(loss='sinkhorn', p=2, blur=.05)
        loss = loss_func(instance, prototype)
        return output, loss

    def cal_independence_loss(self, prototypes, type_name):
        prototypes = prototypes.to(self.device)
        num = self.domain_num if type_name == 'domain' else self.task_num
        loss = 0
        for i in range(num):
            for j in range(i + 1, num):
                loss += torch.sum(torch.pow(F.cosine_similarity(prototypes[i + 1].prototype, prototypes[j + 1].prototype, dim=-1), 2))
        cnt = (num * (num - 1)) // 2
        loss /= cnt
        return loss

    def forward(self, x, mode='train'):
        # ========== input process ==========
        embed_x = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size, input_dims]

        # ========== get domain mask ==========
        domain_id = x['domain_indicator'].clone().detach()
        mask = []
        for d in range(self.domain_num):
            mask.append((domain_id == d))

        # ========== s-prp: commonality ==========
        expert_outs_c = [expert_c(embed_x).unsqueeze(1) for expert_c in self.experts_c]  # expert_outs_c[i]: [batch_size, 1, expert_dims[-1]]
        expert_outs_c = torch.cat(expert_outs_c, dim=1)  # [batch_size, n_expert, expert_dims[-1]]
        gate_out_c = self.gate_c(embed_x).unsqueeze(-1)  # [batch_size, n_expert, 1]
        expert_weights_c = torch.mul(gate_out_c, expert_outs_c)  # [batch_size, n_expert, expert_dims[-1]]
        expert_pooling_c = torch.sum(expert_weights_c, dim=1)  # [batch_size, expert_dims[-1]]

        # ========== s-prp: specificity ==========
        expert_outs_s = [expert_s(embed_x).unsqueeze(1) for expert_s in self.experts_s]  # expert_outs_s[i]: [batch_size, 1, expert_dims[-1]]
        expert_weights_s = torch.zeros_like(expert_outs_s[0])  # [batch_size, 1, expert_dims[-1]]
        for d in range(self.domain_num):
            domain_mask = mask[d].unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]
            expert_weights_s = torch.where(domain_mask, expert_outs_s[d], expert_weights_s)  # [batch_size, 1, expert_dims[-1]]
        expert_pooling_s = torch.sum(expert_weights_s, dim=1)  # [batch_size, expert_dims[-1]]

        # ========== s-prp: combine commonality with specificity ==========
        if self.use_ot:
            expert_pooling_c_ot, loss_c = self.optimal_transport(expert_pooling_c, self.domain_prototypes[0].prototype, self.domain_alphas[0])  # [batch_size, expert_dims[-1]]
            expert_pooling_s_ots, loss_s = [], 0
            for d in range(1, self.domain_num + 1):

                
                
                loss_s += tmp2
            self.loss_dict['loss_c'] = loss_c
            self.loss_dict['loss_s'] = loss_s
            expert_pooling_s_ot = torch.zeros_like(expert_pooling_s_ots[0])  # [batch_size, expert_dims[-1]]
            for d in range(self.domain_num):
                domain_mask = mask[d].unsqueeze(1)
                expert_pooling_s_ot = torch.where(domain_mask, expert_pooling_s_ots[d], expert_pooling_s_ot)
            
        else:
            s_prp_emb = torch.cat([expert_pooling_c, expert_pooling_s], dim=-1)  # [batch_size, expert_dims[-1] * 2]

        # ========== t-prp: commonality ==========
        expert_outs_c_2 = [expert_c_2(s_prp_emb).unsqueeze(1) for expert_c_2 in self.experts_c_2]  # expert_outs_c_2[i]: [batch_size, 1, expert_dims[-1]]
        expert_outs_c_2 = torch.cat(expert_outs_c_2, dim=1)  # [batch_size, n_expert, expert_dims[-1]]
        gate_out_c_2 = self.gate_c_2(s_prp_emb).unsqueeze(-1)  # [batch_size, n_expert, 1]
        expert_weights_c_2 = torch.mul(gate_out_c_2, expert_outs_c_2)  # [batch_size, n_expert, expert_dims[-1]]
        expert_pooling_c_2 = torch.sum(expert_weights_c_2, dim=1)  # [batch_size, expert_dims[-1]]

        # ========== t-prp: specificity ==========
        expert_outs_s_2 = [expert_s_2(s_prp_emb).unsqueeze(1) for expert_s_2 in self.experts_s_2]  # expert_outs_s_2[i]: [batch_size, 1, expert_dims[-1]]
        expert_outs_s_2 = torch.cat(expert_outs_s_2, dim=1)  # [batch_size, n_expert, expert_dims[-1]]
        expert_pooling_s_2 = torch.mean(expert_outs_s_2, dim=1)  # [batch_size, expert_dims[-1]]

        # ========== t-prp: combine commonality with specificity ==========
        if self.use_ot:


            
            self.loss_dict['loss_c_2'] = loss_c_2
            self.loss_dict['loss_s_2'] = loss_s_2_1 + loss_s_2_2
            t_prp_emb1 = torch.cat([expert_pooling_c_2_ot, expert_pooling_s_2_ot_1], dim=-1)  # [batch_size, expert_dims[-1] * 2]
            t_prp_emb2 = torch.cat([expert_pooling_c_2_ot, expert_pooling_s_2_ot_2], dim=-1)  # [batch_size, expert_dims[-1] * 2]
            tower_out1 = torch.sigmoid(self.towers[0](t_prp_emb1))  # [batch_size, 1]
            tower_out2 = torch.sigmoid(self.towers[1](t_prp_emb2))  # [batch_size, 1]
        else:
            t_prp_emb = torch.cat([expert_pooling_c_2, expert_pooling_s_2], dim=-1)  # [batch_size, expert_dims[-1] * 2]
            tower_out1 = torch.sigmoid(self.towers[0](t_prp_emb))  # [batch_size, 1]
            tower_out2 = torch.sigmoid(self.towers[1](t_prp_emb))  # [batch_size, 1]

        # ========== output process ==========
        output1 = torch.zeros_like(tower_out1)  # [batch_size, 1]
        output2 = torch.zeros_like(tower_out2)  # [batch_size, 1]
        for d in range(self.domain_num):
            output1 = torch.where(mask[d].unsqueeze(1), tower_out1, output1)
            output2 = torch.where(mask[d].unsqueeze(1), tower_out2, output2)
        output1 = output1.squeeze(1)  # [batch_size]
        output2 = output2.squeeze(1)  # [batch_size]

        if self.use_independence_loss:
            self.loss_dict['loss_independence_1'] = self.lambda_1 + self.cal_independence_loss(self.domain_prototypes, type_name='domain')
            self.loss_dict['loss_independence_2'] = self.lambda_1 + self.cal_independence_loss(self.task_prototypes, type_name='task')

        if (self.use_ot or self.use_independence_loss) and mode == 'train':
            return output1, output2, self.loss_dict
        else:
            return output1, output2


class Prototype(nn.Module):
    def __init__(self, K, D):
        super(Prototype, self).__init__()
        self.K = K
        self.D = D
        self.prototype = nn.Parameter(torch.randn(self.K, self.D), requires_grad=True)
