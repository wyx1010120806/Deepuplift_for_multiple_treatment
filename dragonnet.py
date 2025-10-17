import torch.nn as nn
import torch
from basemodel import BaseModel
from baseunit import TowerUnit

class Dragonnet(BaseModel):
    def __init__(self, input_dim=100,discrete_size_cols=[2,3,4,5,2],embedding_dim=64,share_dim=6,
                 share_hidden_dims =[64,64,64,64,64], share_hidden_func = torch.nn.ELU(),
                 base_hidden_dims=[100,100,100,100],output_activation_base=torch.nn.Sigmoid(),base_hidden_func = torch.nn.ELU(),
                 ipw_hidden_dims=[100,100,100,100],output_activation_ipw=torch.nn.Sigmoid(),ipw_hidden_func = torch.nn.ELU(),
                 epsilons_hidden_dims=[100,100,100,100],output_activation_epsilons=torch.nn.Sigmoid(),epsilons_hidden_func = torch.nn.ELU(),
                 task = 'classification',classi_nums=2, treatment_label_list=[0,1,2,3],model_type='Tarnet',device='cpu'):
        super(Dragonnet, self).__init__()
        self.model_type = model_type
        self.layers = []
        self.treatment_nums = len(treatment_label_list)
        self.treatment_model = nn.ModuleDict()
        self.treatment_label_list = treatment_label_list
        input_dim = input_dim - len(discrete_size_cols) + len(discrete_size_cols)*embedding_dim
        if len(treatment_label_list) > 2:
            output_activation_ipw = torch.nn.Softmax(dim=1)

        # embedding 
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim).to(device) for size in discrete_size_cols
        ]).to(device)
        
        # share tower
        self.share_tower = TowerUnit(input_dim = input_dim, 
                 hidden_dims=share_hidden_dims, 
                 share_output_dim=share_dim, 
                 activation=share_hidden_func, 
                 use_batch_norm=True, 
                 use_dropout=True, 
                 dropout_rate=0.3, 
                 task='share', 
                 classi_nums=None, 
                 device=device, 
                 use_xavier=True)

        for treatment_label in self.treatment_label_list:
            # treatment tower
            self.treatment_model[str(treatment_label)] = TowerUnit(input_dim = share_dim, 
                 hidden_dims=base_hidden_dims, 
                 share_output_dim=None, 
                 activation=base_hidden_func, 
                 use_batch_norm=True, 
                 use_dropout=True, 
                 dropout_rate=0.3, 
                 task=task, 
                 classi_nums=classi_nums, 
                 output_activation=output_activation_base,
                 device=device, 
                 use_xavier=True)
        
        # ipw tower
        self.ipw_tower = TowerUnit(input_dim = share_dim, 
                 hidden_dims=ipw_hidden_dims, 
                 share_output_dim=None, 
                 activation=ipw_hidden_func, 
                 use_batch_norm=True, 
                 use_dropout=True, 
                 dropout_rate=0.3, 
                 task='classification', 
                 classi_nums=self.treatment_nums, 
                 output_activation=output_activation_ipw,
                 device=device, 
                 use_xavier=True)

        # epsilons tower
        self.epsilons_tower = TowerUnit(input_dim = share_dim, 
                 hidden_dims=epsilons_hidden_dims, 
                 share_output_dim=None, 
                 activation=epsilons_hidden_func, 
                 use_batch_norm=True, 
                 use_dropout=True, 
                 dropout_rate=0.3, 
                 task='regression', 
                 classi_nums=self.treatment_nums, 
                 output_activation=output_activation_epsilons,
                 device=device, 
                 use_xavier=True)
        

    def forward(self, X, t, X_discrete=None, X_continuous=None):
        embedded = [emb(X_discrete[:, i].long()) for i, emb in enumerate(self.embeddings)]
        X_discrete_emb = torch.cat(embedded, dim=1)  # 拼接所有embedding
        x = torch.cat((X_continuous,X_discrete_emb), dim=1)

        share_out = self.share_tower(x)

        pre = []

        for treatment_label in self.treatment_label_list:
            predcit_pro = self.treatment_model[str(treatment_label)](share_out).squeeze().unsqueeze(1)
            pre.append(predcit_pro)

        pre.append(self.epsilons_tower(share_out).squeeze().unsqueeze(1))

        pre.append(self.ipw_tower(share_out))

        ate = []
        if not self.training:
            base_predcit_pro = self.treatment_model['0'](share_out).squeeze().unsqueeze(1)
            for treatment_label in self.treatment_label_list:
                predcit_pro = self.treatment_model[str(treatment_label)](share_out).squeeze().unsqueeze(1)
                if treatment_label != 0:
                    ate.append(predcit_pro -base_predcit_pro)
        return torch.cat(ate, dim=1) if len(ate) !=0 else None,pre,None

def dragonnet_loss(y_preds,t, y_true,task='regression',loss_type=None,classi_nums=2, treatment_label_list=None,X_true=None):
    if task is None:
        raise ValueError("task must be 'classification' or 'regression'")

    t = t.squeeze().unsqueeze(1).long()
    y_true = y_true.squeeze().unsqueeze(1)
    y_pred = torch.gather(torch.cat(y_preds[:len(treatment_label_list)], dim=1), dim=1, index=t.long()).squeeze().unsqueeze(1)
    epsilons = y_preds[len(treatment_label_list)].squeeze().unsqueeze(1)
    t_pred = y_preds[-1]

    y_true_dict = {}
    y_pred_dict = {}
    t_pred_dict = {}
    t_true_dict = {}
    epsilons_dict = {}
    for treatment in treatment_label_list:
        mask = (t == treatment)
        t_true_dict[treatment] = t[mask]
        y_true_dict[treatment] = y_true[mask]
        y_pred_dict[treatment] = y_pred[mask]
        epsilons_dict[treatment] = epsilons[mask]
        if len(treatment_label_list) == 2:
            if treatment == 0:
                t_pred_dict[treatment] = 1 - t_pred.squeeze().unsqueeze(1)[mask]
            else:
                t_pred_dict[treatment] = t_pred.squeeze().unsqueeze(1)[mask]
        else:
            t_pred_dict[treatment] = t_pred[:,treatment][mask]


    # 计算每个treatment的损失
    if task == 'classification':
        if loss_type == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss()
        elif loss_type =='BCELoss':
            criterion = nn.BCELoss()
        else:
            raise ValueError("loss_type must be 'BCEWithLogitsLoss' or 'BCELoss'")
    elif task == 'regression':
        if loss_type == 'mse':
            criterion = nn.MSELoss()
        elif loss_type =='huberloss':
            criterion = nn.SmoothL1Loss() 
        else:
            raise ValueError("loss_type must be 'mse' or 'huberloss'")
    else:
        raise ValueError("task must be 'classification' or'regression'")
    
    # loss y
    loss_y = 0
    for treatment in treatment_label_list:
        loss_y += criterion(y_pred_dict[treatment], y_true_dict[treatment])

    # loss ipw
    loss_ipw = 0
    if len(treatment_label_list) == 2:
        ipw_criterion = nn.BCELoss()
        loss_ipw = ipw_criterion(t_pred, t.float())

    if len(treatment_label_list) > 2: 
        ipw_criterion = nn.CrossEntropyLoss()
        loss_ipw = ipw_criterion(t_pred, t.squeeze())

    # loss_regularization
    loss_regularization = 0
    for treatment in treatment_label_list:
        t_pre = (t_pred_dict[treatment] + 0.01) / 1.02
        y_pre = y_pred_dict[treatment]
        t_true = t_true_dict[treatment]

        h = t_true / t_pre
        y_pert = y_pre + epsilons_dict[treatment] * h

        if task == 'classification':
                criterion = nn.BCEWithLogitsLoss(reduction='sum')
        elif task == 'regression':
            if loss_type == 'mse':
                criterion = nn.MSELoss(reduction='sum')
            elif loss_type =='huberloss':
                criterion = nn.SmoothL1Loss(reduction='sum') 
            else:
                raise ValueError("loss_type must be 'mse' or 'huberloss'")
        else:
            raise ValueError("task must be 'classification' or'regression'")
        
        loss_regularization += criterion(y_pert, y_true_dict[treatment])

    loss = loss_y + loss_ipw + loss_regularization
    return loss, loss_y, None



        
        