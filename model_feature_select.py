import random
import numpy as np
import pandas as pd

from urllib.parse import urlparse, urlunparse
from functools import partial
from esn_tarnet import *
from feature_select import *
from s_learner import *
from t_learner import * 
from tarnet import *
import subprocess
import json
from pathlib import Path

def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)



def run_shell(hdfs_ls_cmd):
    print(hdfs_ls_cmd)
    result = subprocess.run(hdfs_ls_cmd, shell=True, capture_output=True, text=True)     
    if result.returncode != 0:
        print(f"Error listing files in HDFS: {result.stderr}")
    print(result)
    print(result.returncode)


# 读取JSON文件并转换为字典
with open('/opt/tiger/rh2_params/custom_template_vars.json', 'r') as file:
    data = json.load(file)
print('全局',data)


# 打印字典
print(data)
selected_features_num = int(data['selected_features_num'])
task = data['task']
y = data['y']

train_data_path = data['train_data_path']
train_data = Path(train_data_path).name

test_data_path = data['test_data_path']
test_data = Path(test_data_path).name

feature_path = data['feature_path']
feature = Path(feature_path).name

selected_features_path = data['selected_features_path']
selected_features = Path(selected_features_path).name

p = urlparse(selected_features_path)
parent_path = p.path.rsplit('/', 1)[0]  
selected_path = urlunparse((p.scheme, p.netloc, parent_path, '', '', ''))
print(selected_path)

feature_list_discrete_path = os.path.basename(selected_features_path).split('.')[0] + '_discrete.pkl'
discrete_size_cols_path = os.path.basename(selected_features_path).split('.')[0] + '_discrete_size.pkl'


print("selected_features_num:", selected_features_num)
print("task:", task)
print("y:", y)
print("train_data_path:", train_data_path)
print("test_data_path:", test_data_path)
print("feature_path:", feature_path)
print("selected_features_path:", selected_features_path)
print("feature_list_discrete_path:", selected_path + '/' + feature_list_discrete_path)
print("discrete_size_cols_path:", selected_path + '/' + discrete_size_cols_path)


run_shell(f"rm -r ./{train_data}")
run_shell(f"hdfs dfs -get {train_data_path} ./")

run_shell(f"rm -r ./{test_data}")
run_shell(f"hdfs dfs -get {test_data_path} ./")

run_shell(f"rm -r ./{feature}")
run_shell(f"hdfs dfs -get {feature_path} ./")

#读取特征列表
import pickle
with open(f'./{feature}', 'rb') as f:
    feature_list = pickle.load(f)
print(len(feature_list))

df = pd.read_parquet(f'./{train_data}',columns=feature_list+[y]).fillna(0)
df_test = pd.read_parquet(f'./{test_data}',columns=feature_list+[y]).fillna(0)

top_features,feature_list_discrete,discrete_size_cols = features_select(df,df_test,feature_list,y,selected_features_num,task)

import pickle
# 将列表保存到文件
with open(f"./{selected_features}", 'wb') as f:
    pickle.dump(top_features, f)
tmp = selected_path + '/' + selected_features
run_shell(f"hdfs dfs -rm -r {tmp}")
run_shell(f"hdfs dfs -mkdir {selected_path}")
run_shell(f"hdfs dfs -put ./{selected_features} {selected_path}")

with open(f"./{feature_list_discrete_path}", 'wb') as f:
    pickle.dump(feature_list_discrete, f)
tmp = selected_path + '/' + feature_list_discrete_path
run_shell(f"hdfs dfs -rm -r {tmp}")
run_shell(f"hdfs dfs -mkdir {selected_path}")
run_shell(f"hdfs dfs -put ./{feature_list_discrete_path} {selected_path}")

with open(f"./{discrete_size_cols_path}", 'wb') as f:
    pickle.dump(discrete_size_cols, f)
tmp = selected_path + '/' + discrete_size_cols_path
run_shell(f"hdfs dfs -rm -r {tmp}")
run_shell(f"hdfs dfs -mkdir {selected_path}")
run_shell(f"hdfs dfs -put ./{discrete_size_cols_path} {selected_path}")


