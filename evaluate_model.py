#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from evaluate import *
import random
import numpy as np
import pandas as pd
import torch
import pickle
from torch.nn import *
from urllib.parse import urlparse, urlunparse
from pathlib import Path
from uplifttree import *
from functools import partial
from esn_tarnet import *
from feature_select import *
from s_learner import *
from t_learner import *
from tarnet import *
from dragonnet import *
from x_learner import *
from descn import *
from uplifttree import *
from cfrnet import *
import subprocess
import json

def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


def run_shell(hdfs_ls_cmd):
    print(hdfs_ls_cmd)
    result = subprocess.run(hdfs_ls_cmd, shell=True, capture_output=True, text=True)     
    if result.returncode != 0:
        print(f"Error listing files in HDFS: {result.stderr}")
    print(result)
    print(result.returncode)


# In[ ]:


# 读取JSON文件并转换为字典
with open('/opt/tiger/rh2_params/custom_template_vars.json', 'r') as file:
    data = json.load(file)

# data = {
#     "test_data_path": "hdfs://harunasg/home/byte_ecom_product_ds_sg/xh/mall_test_VN",
#     "feature_path": "hdfs://harunasg/home/byte_ecom_product_ds_sg/xh/VN_feature_list_mall_gmv_int_label_selected.pkl",
#     "y": "gmv_int_label",
#     "treatment_col": "is_treatment",
#     "treatment_label_list": "0,1,2,3",
#     "task": "classification",
#     "model_type":["tarnet","tlearner"],
#     "model_path": ["hdfs://harunasg/home/byte_ecom_product_ds_sg/xh/VN_mall_gmv_int_label_tarnet.pkl","hdfs://harunasg/home/byte_ecom_product_ds_sg/xh/VN_mall_gmv_int_label_tlearner.pkl"],
#     "model_params": [{},{}]
# }


# 打印字典
print(data)
feature_path = data['feature_path']
feature = Path(feature_path).name
p = urlparse(feature_path)
parent_path = p.path.rsplit('/', 1)[0]  
selected_path = urlunparse((p.scheme, p.netloc, parent_path, '', '', ''))

feature_list_discrete_path = os.path.basename(feature_path).split('.')[0] + '_discrete.pkl'
feature_list_discrete_hdfs_path = selected_path + '/' + feature_list_discrete_path
discrete_size_cols_path = os.path.basename(feature_path).split('.')[0] + '_discrete_size.pkl'
discrete_size_cols_hdfs_path = selected_path + '/' + discrete_size_cols_path

test_data_path = data['test_data_path']
test_data_path_name = Path(test_data_path).name

s = data.get('treatment_label_list', '')
treatment_label_list = [int(x.strip()) for x in s.split(',') if x.strip()]

y = data['y']
treatment_col = data['treatment_col']
s = data.get('treatment_label_list', '')
treatment_label_list = [int(x.strip()) for x in s.split(',') if x.strip()]

task = data['task']
model_path_list = data['model_path']
model_params_list = data['model_params']
model_type_list = data['model_type']


# In[ ]:


run_shell(f"rm -r ./{test_data_path_name}")
run_shell(f"hdfs dfs -get {test_data_path} ./")

run_shell(f"rm -r ./{feature}")
run_shell(f"hdfs dfs -get {feature_path} ./")

run_shell(f"rm -r ./{feature_list_discrete_path}")
run_shell(f"hdfs dfs -get {feature_list_discrete_hdfs_path} ./")

run_shell(f"rm -r ./{discrete_size_cols_path}")
run_shell(f"hdfs dfs -get {discrete_size_cols_hdfs_path} ./")


# In[ ]:


#读取特征列表
with open(f"./{feature}", 'rb') as f:
    feature_list = pickle.load(f)
print(len(feature_list))

with open(f"./{feature_list_discrete_path}", 'rb') as f:
    feature_list_discrete = pickle.load(f)
print(len(feature_list_discrete))

with open(f"./{discrete_size_cols_path}", 'rb') as f:
    discrete_size_cols = pickle.load(f)
print(len(discrete_size_cols))


# In[ ]:


model_list = {}
for i in range(len(model_path_list)):
    model_type = model_type_list[i]
    model_path = model_path_list[i]
    model_params = model_params_list[i]

    if not model_params:
        if model_type == 'tarnet':
            model_params = dict(
                embedding_dim=3,share_dim=64,
                share_hidden_dims =[256,128,128,64],
                base_hidden_dims = [64,32,32,16],output_activation_base=None,
                share_hidden_func = torch.nn.ELU(),base_hidden_func = torch.nn.ELU()
            )
        elif model_type == 'esn_tarnet':
            model_params = dict(
                embedding_dim=3,share_dim=64,
                share_hidden_dims =[256,128,64,64],
                base_hidden_dims=[64,32,32,16],output_activation_base=torch.nn.Sigmoid(),
                ipw_hidden_dims=[256,128,64,64],output_activation_ipw=None,
                share_hidden_func = torch.nn.ELU(),base_hidden_func = torch.nn.ELU(), ipw_hidden_func = torch.nn.ELU()
            )
        elif model_type == 'slearner':
            model_params = dict(
                embedding_dim=3,
                base_hidden_dims=[64,32,32,16],output_activation_base=None,base_hidden_func = torch.nn.ELU()
            )
        elif model_type == 'tlearner':
            model_params = dict(
                embedding_dim=3,
                base_hidden_dims=[64,32,32,16],output_activation_base=None,base_hidden_func = torch.nn.ELU()
            )
        elif model_type == 'xlearner':
            model_params = dict(
                embedding_dim=3,
                base_hidden_dims=[64,32,32,16],output_activation_base=torch.nn.Sigmoid(),base_hidden_func = torch.nn.ELU()
                ,lift_hidden_dims=[64,32,32,16],lift_activation_base=None,lift_hidden_func = torch.nn.ELU()
                ,treatment_label_weight=[0.25,0.25,0.25,0.25],lift_weight=[0.5,0.5,0.5,0.5]
            )
        elif model_type == 'descn':
            model_params = dict(
                embedding_dim=3,share_dim=64,
                        share_hidden_dims =[256,128,64,64],
                        base_hidden_dims=[64,32,32,16],output_activation_base=None,
                        ipw_hidden_dims=[64,32,32,16],output_activation_ipw=None,
                        pseudo_hidden_dims=[64,32,32,16],output_activation_pseudo=None,
                        share_hidden_func = torch.nn.ELU(),base_hidden_func = torch.nn.ELU(), ipw_hidden_func = torch.nn.ELU(),pseudo_hidden_func = torch.nn.ELU()
            )
        elif model_type == 'dragonnet':
            model_params = dict(
                embedding_dim=3,share_dim=64,
                share_hidden_dims =[512,256,256,128], share_hidden_func = torch.nn.ELU(),
                base_hidden_dims=[64,32,32,16],output_activation_base=torch.nn.Sigmoid(),base_hidden_func = torch.nn.ELU(),
                ipw_hidden_dims=[64,32,32,16],output_activation_ipw=torch.nn.Sigmoid(),ipw_hidden_func = torch.nn.ELU(),
                epsilons_hidden_dims=[64,32,32,16],output_activation_epsilons=torch.nn.Sigmoid(),epsilons_hidden_func = torch.nn.ELU()
            )
        elif model_type == 'cfrnet':
            model_params = dict(
                embedding_dim=3,share_dim=64,
                share_hidden_dims =[256,128,128,64],
                base_hidden_dims = [64,32,32,16],output_activation_base=None,
                share_hidden_func = torch.nn.ELU(),base_hidden_func = torch.nn.ELU()
            )
        elif model_type == 'uplifttree':
            model_params = dict(
                max_depth=5
            )
        elif model_type == 'upliftforest':
            model_params = dict(
                max_depth=5,n_estimators=100
            )
        elif model_type == 'causalforestdml':
            model_params = dict(
                max_depth=5,n_estimators=100
            )   
        else:
            raise ValueError("model_type must be 'tarnet', 'cfrnet', 'esn_tarnet', 'slearner', 'tlearner', 'xlearner', 'descn', 'dragonnet', 'uplifttree','upliftforest','causalforestdml'")

    if model_type == 'tarnet':
        model = Tarnet(
            input_dim=len(feature_list), discrete_size_cols=discrete_size_cols,
            treatment_label_list=treatment_label_list,device=device,model_type = model_type,task=task,classi_nums=2,
            **model_params
        ).to(device)
        loss_f = partial(tarnet_loss)
    elif model_type == 'esn_tarnet':
        model = ESN_Tarnet(
            input_dim=len(feature_list), discrete_size_cols=discrete_size_cols,
            treatment_label_list=treatment_label_list,device=device,model_type = model_type,task=task,classi_nums=2,
            **model_params
        ).to(device)
        loss_f = partial(esn_tarnet_loss)
    elif model_type == 'slearner':
        model = Slearner(
            input_dim=len(feature_list), discrete_size_cols=discrete_size_cols,
            treatment_label_list=treatment_label_list,device=device,model_type = model_type,task=task,classi_nums=2,
            **model_params
        ).to(device)
        loss_f = partial(slearn_loss)
    elif model_type == 'tlearner':
        model = Tlearner(
            input_dim=len(feature_list), discrete_size_cols=discrete_size_cols,
            treatment_label_list=treatment_label_list,device=device,model_type = model_type,task=task,classi_nums=2,
            **model_params
        ).to(device)
        loss_f = partial(tlearn_loss)
    elif model_type == 'xlearner':
        model = Xlearner(
            input_dim=len(feature_list), discrete_size_cols=discrete_size_cols,
            treatment_label_list=treatment_label_list,device=device,model_type = model_type,task=task,classi_nums=2,
            **model_params
        ).to(device)
        loss_f = partial(xlearn_loss)
    elif model_type == 'descn':
        model = Descn(
            input_dim=len(feature_list), discrete_size_cols=discrete_size_cols,
            treatment_label_list=treatment_label_list,device=device,model_type = model_type,task=task,classi_nums=2,
            **model_params
        ).to(device)
        loss_f = partial(descn_loss)
    elif model_type == 'dragonnet':
        model = Dragonnet(
            input_dim=len(feature_list), discrete_size_cols=discrete_size_cols,
            treatment_label_list=treatment_label_list,device=device,model_type = model_type,task=task,classi_nums=2,
            **model_params
        ).to(device)
        loss_f = partial(dragonnet_loss)
    elif model_type == 'cfrnet':
        model = Cfrnet(
            input_dim=len(feature_list), discrete_size_cols=discrete_size_cols,
            treatment_label_list=treatment_label_list,device=device,model_type = model_type,task=task,classi_nums=2,
            **model_params
        ).to(device)
        loss_f = partial(cfrnet_loss)
    elif model_type == 'uplifttree':
        model = UpliftTreeModel(task=task,model_type='tree',treatment_list=treatment_label_list,features_list=feature_list,**model_params)
    elif model_type == 'upliftforest':
        model = UpliftTreeModel(task=task,model_type='forest',treatment_list=treatment_label_list,features_list=feature_list,**model_params)
    elif model_type == 'causalforestdml':
        model = UpliftTreeModel(task=task,model_type='causalforestdml',treatment_list=treatment_label_list,features_list=feature_list,**model_params)
    else:
        raise ValueError("model_type must be 'tarnet', 'cfrnet', 'esn_tarnet', 'slearner', 'tlearner', 'xlearner', 'descn', 'dragonnet', 'uplifttree','upliftforest','causalforestdml'")
    
    model_path_name = Path(model_path).name
    run_shell(f"rm -r ./{model_path_name}")
    run_shell(f"hdfs dfs -get {model_path} ./")

    checkpoint = torch.load(f"./{model_path_name}", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model_list[model_type] = model


# In[ ]:


run_shell(f"rm -r ./{test_data_path_name}")
run_shell(f"hdfs dfs -get {test_data_path} ./")


# In[ ]:


test_df = pd.read_parquet(f"./{test_data_path_name}",columns=feature_list+[treatment_col,y]).fillna(0)


# In[ ]:


X_discrete_test = torch.tensor(test_df[feature_list_discrete].values, dtype=torch.float32).to(device)
X_continuous_test = torch.tensor(test_df[[_ for _ in feature_list if _ not in feature_list_discrete]].values, dtype=torch.float32).to(device)


# In[ ]:


uplift_prediction = {}
for each in model_list.keys():
    model = model_list[each]
    model.to(device)
    model.eval()
    with torch.no_grad(): 
        uplift_predictions,y_preds,*eps = model(None, None, X_discrete=X_discrete_test, X_continuous=X_continuous_test)
    uplift_prediction[each] = uplift_predictions.detach().cpu().numpy()


# In[ ]:

import warnings
warnings.filterwarnings("ignore")
for treatment in treatment_label_list[1:]:
    print(f'*********************** treatment标识:{treatment} ***********************')
    for each in uplift_prediction.keys():
        test_df[each] = uplift_prediction[each][:,treatment-1]
 
    df_tmp = test_df[test_df[treatment_col].isin([0,treatment])]
    df_tmp[treatment_col] = df_tmp[treatment_col].apply(lambda x: 1 if x == treatment else 0)
    df_tmp['random'] = np.random.rand(df_tmp.shape[0])

    evaluate(df=df_tmp[[y,treatment_col,'random'] + [_ for _ in uplift_prediction.keys()]],y_true=y,treatment=treatment_col,divide_feature=None,n=100)

    

