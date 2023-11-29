from sklearn import datasets
import warnings
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
import shap
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import ast
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score

class autohypothesis_utils(object):
 
    @staticmethod    
    def create_dataset(n, k, c, b):

        # Generate a dataset with b correlated features
        X, Y = make_classification(n_samples=k, n_features=n, n_informative=b, n_redundant=0, n_classes=c, n_clusters_per_class=1, random_state=42)


        # Create a DataFrame
        columns = [f"feature_{i}" for i in range(1, n+1)]
        df = pd.DataFrame(X, columns=columns)

        df["target"] = Y

        return df
    @staticmethod   
    def opti_loop(df, N_TRIAL, optimize_obj="dual"):
        X = df.drop(columns=['target'])
        y = df['target']
        le = LabelEncoder()
        y = le.fit_transform(y)
        X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y, random_state=42)

        n_startup = 11*df.shape[1]-1

        if optimize_obj == "dual":
            directions = ["maximize", "maximize"]
            sampler = optuna.samplers.MOTPESampler(n_startup_trials=n_startup)
        else:
            directions = ["maximize"]
            sampler = optuna.samplers.TPESampler()

        motpe_experiment = optuna.create_study(sampler=sampler, directions=directions)
        motpe_experiment.optimize(lambda trial: autohypothesis_utils.train(trial, X_train, y_train, X_dev, y_dev, optimize_obj=optimize_obj), n_trials=N_TRIAL)

        return motpe_experiment, X_train, X_dev, y_train, y_dev
    @staticmethod  
    def split_responses(df, column_name):
        # Get unique modes of transport from the column by splitting by ', '
        modes = set()
        for row in df[column_name].dropna():
            modes.update(row.split(", "))

            # Create a new column for each mode and set to False by default
        for mode in modes:
            df[mode] = False

        # Iterate over the rows and update the corresponding mode columns to True if the mode is mentioned
        for index, row in df.iterrows():
            if pd.notna(row[column_name]):
                for mode in row[column_name].split(", "):
                    df.at[index, mode] = True    

        # Drop the original column
        df.drop(columns=[column_name], inplace=True)   

        return df
    @staticmethod  
    def normalize_shap(I_c):

        
        sum_I_c = np.sum(I_c)
        I_c_normalized = I_c / sum_I_c
        return I_c_normalized
    @staticmethod  
    def clusterEntropy(pipeline, X, y_pred):
        model = pipeline.named_steps["classifier"]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        epsilon = 1e-20
        N = len(y_pred)
        F = X.shape[1]
        total_entropy = 0

        for class_idx in np.unique(y_pred):
            class_shap_values = shap_values[class_idx][y_pred == class_idx]
            I_c = np.mean(np.abs(class_shap_values), axis=0)

            if np.all(I_c == 0):
                return 0
            else:
                
                I_c_prime = autohypothesis_utils.normalize_shap(I_c)
                class_entropy = -np.sum(I_c_prime * np.log2(I_c_prime + epsilon))

            total_entropy += (np.sum(y_pred == class_idx) * class_entropy)

        E_prime = total_entropy / (N * np.log2(F))
        return 1 - E_prime

    @staticmethod  
    def clusterEntropy_vanilla(model, X, y_pred):

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        epsilon = 1e-20
        N = len(y_pred)
        F = X.shape[1]
        total_entropy = 0

        for class_idx in np.unique(y_pred):
            class_shap_values = shap_values[class_idx][y_pred == class_idx]
            I_c = np.mean(np.abs(class_shap_values), axis=0)

            if np.all(I_c == 0):
                return 0
            else:
        
                I_c_prime = autohypothesis_utils.normalize_shap(I_c)
                class_entropy = -np.sum(I_c_prime * np.log2(I_c_prime + epsilon))

            total_entropy += (np.sum(y_pred == class_idx) * class_entropy)

        E_prime = total_entropy / (N * np.log2(F))
        return 1 - E_prime


    @staticmethod  
    def train(trial, X_train, y_train, X_dev, y_dev, optimize_obj="dual"):

        
        classifier_name = trial.suggest_categorical("classifier", ["RandomForest"])



        classifier_obj = RandomForestClassifier(
            n_estimators = trial.suggest_int("rf_n_estimators", 10, 1000),
            max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True),
            min_samples_split = trial.suggest_float("rf_min_samples_split", 0.1, 1),
            min_samples_leaf = trial.suggest_float("rf_min_samples_leaf", 0.1, 0.5),
            max_features = trial.suggest_categorical("rf_max_features", ["sqrt", "log2"]),
            random_state = trial.suggest_int("rf_random_state", 42, 42)

        )


        pipeline = Pipeline([

            ('classifier', classifier_obj)
        ])

        pipeline.fit(X_train, y_train)


        obj1 = accuracy_score(y_dev, pipeline.predict(X_dev))
        obj2 = autohypothesis_utils.clusterEntropy(pipeline, X_dev, pipeline.predict(X_dev))
        if optimize_obj == "dual":

            
            return obj1, obj2
        else:
            return obj1,
    @staticmethod  
    def get_trial_hyperparams(study, trial_number):
        trial = study.trials[trial_number]
        return trial.params
    @staticmethod
    def rebuild_pipeline_with_hyperparams(params):
        if params['classifier'] == "XGB":
            xgb_params = {key.replace('xgb_', ''): params[key] for key in params if key.startswith('xgb_')}
            classifier_obj = xgb.XGBClassifier(**xgb_params)

        elif params['classifier'] == "RandomForest":
            rf_params = {key.replace('rf_', ''): params[key] for key in params if key.startswith('rf_')}
            classifier_obj = RandomForestClassifier(**rf_params)
        elif params['classifier'] == "LGBM":
            lgbm_params = {key.replace('lgbm_', ''): params[key] for key in params if key.startswith('lgbm_')}
            classifier_obj = lgb.LGBMClassifier(**lgbm_params)

        # Add further conditions for other classifiers if needed

        pipeline = Pipeline([

            ('classifier', classifier_obj)
        ])
        return pipeline
    @staticmethod
    def is_dominated(a, b):
        """Retourne True si a est dominé par b."""
        return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))
    @staticmethod
    def pareto_front(objectives):
        """Retourne les solutions du front de Pareto."""
        front = []
        for i in range(len(objectives)):
            dominated = False
            for j in range(len(objectives)):
                if autohypothesis_utils.is_dominated(objectives[i], objectives[j]):
                    dominated = True
                    break
            if not dominated:
                front.append(objectives[i])
        return np.array(front)
    @staticmethod
    def add_shap_std_columns(row, X_train, y_train, X_dev, y_dev):
        # Reconstruct the pipeline with hyperparameters
        pipeline = autohypothesis_utils.rebuild_pipeline_with_hyperparams(ast.literal_eval(row['params']))
        pipeline.fit(X_train, y_train)
        
        explainer = shap.Explainer(pipeline.named_steps['classifier'])
        # Calculate the SHAP values
        shap_values = explainer.shap_values(X_train)
        
        # Initialize a dictionary to store the average standard deviations of SHAP values for each class
        shap_std_dict = {}
        
        for class_index in range(len(shap_values)):
            # For each class, calculate the standard deviation of SHAP values for all instances, then take the average
            shap_std_dict[f'shap_std_avg_class_{class_index}'] = np.mean(np.std(shap_values[class_index], axis=0))
            
        return pd.Series(shap_std_dict)

    def compute_shap_std_for_models(df, X_train, y_train, X_dev, y_dev):
        # Apply the function to each row of the DataFrame to obtain the average standard deviations of SHAP values for each class
        shap_std_df = df.apply(lambda row: autohypothesis_utils.add_shap_std_columns(row, X_train, y_train, X_dev, y_dev), axis=1)
        
        # Concatenate the results to the original DataFrame
        df = pd.concat([df, shap_std_df], axis=1)
        
        return df
    @staticmethod
    def add_shap_columns(row, X_train, y_train, X_dev, y_dev):
        # Reconstruct the pipeline with hyperparameters
        pipeline = autohypothesis_utils.rebuild_pipeline_with_hyperparams(ast.literal_eval(row['params']))
        pipeline.fit(X_train, y_train)
        explainer = shap.Explainer(pipeline.named_steps['classifier'])
        # Calculate the SHAP values
        shap_values = explainer.shap_values(X_train)
        selected_shap_values = np.zeros((X_dev.shape[0], X_dev.shape[1]))
        
        for i in range(X_dev.shape[0]):
            selected_shap_values[i] = np.abs(shap_values[y_dev[i]][i]) * X_dev.iloc[i]
            
        df_shap = pd.DataFrame(selected_shap_values, columns=X_dev.columns)
        acc = accuracy_score(y_dev, pipeline.predict(df_shap))
        entropy = autohypothesis_utils.clusterEntropy(pipeline, df_shap, pipeline.predict(df_shap))
        
        return pd.Series([acc, 1-entropy], index=['accuracy_shap', 'entropy_shap'])