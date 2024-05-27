import numpy as np
import optuna
import sklearn.metrics
import xgboost as xgb
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


train_set = pd.read_csv('data/train_set60.csv')
balanced_small_train = pd.read_csv("data/balanced_small_train_set.csv")
balanced_large_train = pd.read_csv("data/balanced_large_train_set.csv")
validation_set = pd.read_csv('data/validate_set20.csv')

def objective_dtree(trial):

    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 100),
        'max_features': trial.suggest_int('max_features', 1, 7),
        "random_state": 1,
    }
    
    model = DecisionTreeRegressor(**params)
    model.fit(train_x, train_y)
    
    preds = model.predict(valid_x)
    pred_labels = np.rint(preds)
    rmse = sklearn.metrics.root_mean_squared_error(valid_y, pred_labels)
    
    return rmse

def objective_rf(trial):

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1, 1000),
        'max_depth': trial.suggest_int('max_depth', 1, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 100),
        'max_features': trial.suggest_int('max_features', 1, 7),
        'max_samples': trial.suggest_float('max_samples', 0.1, 1.0),
        "n_jobs": -1,
        "random_state": 1
    }
    
    model = RandomForestRegressor(**params)
    model.fit(train_x, train_y)
    
    preds = model.predict(valid_x)
    pred_labels = np.rint(preds)
    rmse = sklearn.metrics.root_mean_squared_error(valid_y, pred_labels)
    
    return rmse

def objective_xgb(trial):

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)

    param = {
        "n_jobs": -1,
        "random_state": 1,
        "objective": "reg:squarederror",
        "tree_method": "exact",
        "booster": "gbtree",
        "eta": trial.suggest_float("eta", 0.001, 0.1),
        "max_depth": trial.suggest_int("max_depth", 1, 15),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "lambda": trial.suggest_float("lambda", 1e-7, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-7, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    num_boost_round = 1000000
    early_stopping_rounds = 250

    bst = xgb.train(param, dtrain, num_boost_round=num_boost_round,
                    evals=[(dvalid, "Validation")], 
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=False)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    rmse = sklearn.metrics.root_mean_squared_error(valid_y, pred_labels)

    return rmse


model_objectives = [objective_dtree, objective_rf, objective_xgb]
training_sets = [train_set, balanced_small_train, balanced_large_train]
model_names = ['DecisionTreeRegressor', 'RandomForestRegressor', 'XGBRegressor']
names_iterator = 0

for model_objective in model_objectives:
    
    for training_set in training_sets:
        train_y = training_set['TOTAL_ACCIDENTS']
        train_x = training_set.drop(['TOTAL_ACCIDENTS'], axis=1)

        valid_y = validation_set['TOTAL_ACCIDENTS']
        valid_x = validation_set.drop(['TOTAL_ACCIDENTS'], axis=1)

        study = optuna.create_study(direction='minimize')
        study.optimize(model_objective, n_trials=1000)

        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        best_trial = study.best_trial
        print("  Value: {}".format(best_trial.value))
        print("  Params: ")
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))

        if training_set.equals(train_set):
            name = 'train_set'
        elif training_set.equals(balanced_small_train):
            name = 'balanced_small_train'
        else:
            name = 'balanced_large_train'
        df = study.trials_dataframe()
        df.to_csv(f"parameters/{model_names[names_iterator]}_{name}.csv", index=False)
    names_iterator += 1


test = pd.read_csv("parameters/XGBRegressor_train_set.csv")
test = test.sort_values(by='value')
print(test.head())