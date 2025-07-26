import xgboost as xgb
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def features_select(df=None,features=None,label=None,n_top=None,selected_path=None):
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

    # 2. 训练原始模型（使用全部特征）
    model_full = xgb.XGBClassifier(n_estimators = 150,use_label_encoder=False, eval_metric='logloss',n_jobs=-1)
    model_full.fit(df_train[features].values, df_train[label].values)

    # 3. 预测并计算原始模型AUC
    y_pred_proba_full = model_full.predict_proba(df_test[features].values)[:, 1]
    auc_full = roc_auc_score(df_test[label].values, y_pred_proba_full)
    print(f"AUC with all features: {auc_full:.4f}")

     # 4. 计算特征重要性
    importance = model_full.feature_importances_
    importance_df = pd.DataFrame({'feature': features, 'importance': importance})
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    top_features = importance_df['feature'].head(n_top).tolist()
    print(len(top_features))
    # 5. 用选出的特征重新训练模型
    model_selected = xgb.XGBClassifier(n_estimators = 150,use_label_encoder=False, eval_metric='logloss',n_jobs=-1)
    model_selected.fit(df_train[top_features].values, df_train[label].values)

    # 6. 预测测试集，计算AUC
    y_pred_proba = model_selected.predict_proba(df_test[top_features])[:, 1]
    auc_score = roc_auc_score(df_test[label].values, y_pred_proba)
    print(f"Selected top {n_top} features: {top_features}")
    print(f"AUC on test set with selected features: {auc_score:.4f}")

    import pickle
    # 将列表保存到文件
    with open(selected_path, 'wb') as f:
        pickle.dump(top_features, f)

    return top_features

    
    