from causalml.inference.tree import UpliftTreeClassifier,UpliftRandomForestClassifier,CausalTreeRegressor,CausalRandomForestRegressor
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot
from IPython.display import Image
from causalml.inference.tree.plot import plot_causal_tree
import matplotlib.pyplot as plt

class UpliftTreeModel():
    def __init__(self,task,model_type,treatment_list,features_list):
        self.task = task
        self.model_type = model_type
        self.treatment_list = treatment_list
        self.features_list = features_list
        self.model = None
        
    def fit(self,X,y,treatment):
        if self.task == 'classification':
            if self.model_type == 'tree':
                self.model = UpliftTreeClassifier(control_name='0',max_depth=3)
                self.model.fit(X=X,y=y,treatment=treatment)
            elif self.model_type == 'forest':
                self.model = UpliftRandomForestClassifier(control_name='0',max_depth=3,n_estimators=100)
                self.model.fit(X=X,y=y,treatment=treatment)
            else:
                raise ValueError("model_type must be 'tree' or 'forest'")
        elif self.task == 'regression' and len(self.treatment_list) == 2:
            if self.model_type == 'tree':
                self.model = CausalTreeRegressor(control_name='0',max_depth=3)
                self.model.fit(X=X,y=y,treatment=treatment)
            elif self.model_type == 'forest':
                self.model = CausalRandomForestRegressor(control_name=0,max_depth=3,n_estimators=100)
                self.model.fit(X=X,y=y,treatment=treatment.astype(int))
            else:
                raise ValueError("model_type must be 'tree' or 'forest'")
        elif self.task == 'regression' and len(self.treatment_list) > 2:
            self.model = {}

            for treatment_name in self.treatment_list:
                if treatment_name == 0:
                    continue

                mask = (treatment == '0') | (treatment == str(treatment_name))   # 选择 treatment 为 0 或 1 的行
                X_sub = X[mask]
                y_sub = y[mask]
                t_sub = treatment[mask]

                if self.model_type == 'tree':
                    model = CausalTreeRegressor(control_name='0',max_depth=3)
                    model.fit(X=X_sub,y=y_sub,treatment=t_sub)
                elif self.model_type == 'forest':
                    model = CausalRandomForestRegressor(control_name=0,max_depth=3,n_estimators=100)
                    model.fit(X=X_sub,y=y_sub,treatment=t_sub.astype(int))
                else:
                    raise ValueError("model_type must be 'tree' or 'forest'")
                
                self.model[treatment_name] = model
        else:
            raise ValueError("task must be 'regression' or 'classification'")
        
    def predict(self,X):
        if len(self.treatment_list) == 2 or self.task == 'classification':
            if self.task == 'classification' and self.model_type == 'forest':
                return self.model.predict(X, full_output=True)
            else:
                return self.model.predict(X)
        else:
            predictions = {}
            for treatment_name in self.treatment_list:
                if treatment_name == 0:
                    continue
                predictions[treatment_name] = self.model[treatment_name].predict(X)
            return predictions
    
    # def visualize(self):
    #     if len(self.treatment_list) == 2 or self.task == 'classification':
    #         print('***************字符串形式展示***************')
    #         result = uplift_tree_string(self.model.fitted_uplift_tree, self.features_list)
    #         print(result)
    #         print('***************树图形展示***************')
    #         graph = uplift_tree_plot(self.model.fitted_uplift_tree, self.features_list)
    #         print(graph)
    #     else:
    #         for treatment_name in self.treatment_list:
    #             if treatment_name == 0:
    #                 continue
    #             print(f'***************{treatment_name}***************')
    #             print('***************字符串形式展示***************')
    #             result = uplift_tree_string(self.model[treatment_name].fitted_uplift_tree, self.features_list)
    #             print(result)
    #             print('***************树图形展示***************')
    #             graph = uplift_tree_plot(self.model[treatment_name].fitted_uplift_tree, self.features_list)
    #             Image(graph.create_png())
                

