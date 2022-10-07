import numpy as np
import pandas as pd
from catboost import *

# out-of-sample R^2
def r2_oos(true, pred):
    if sum(true**2) == 0:
        result = 'zero division error'
    else:
        result = 1 - sum((true - pred)**2) / sum(true**2)
        result = round(result, 4)
    return result
    
def refit_catboost(df, features, train_start=1996, train_end=2011, valid_size=5):
    cat_features = ['exchcd','shrcd','ffi49']
    r2_list = []
    for i in range(valid_size):
        start_year = train_start
        end_year = train_end + i
        train = df[(df['year']<=end_year)&(df['year']>=start_year)] ; print('train: ', start_year, end_year)
        valid = df[df['year']==end_year+1] ; print('valid: ', end_year+1)
        

        X_train, y_train = train[features], np.array(train['ret']) ; print('shape: ', X_train.shape)
        X_valid, y_valid = valid[features], np.array(valid['ret']) ; print('shape: ', X_valid.shape)

        #     cat_features : LabelEncoder
        for data in [X_train, X_valid]:
            data[cat_features] = data[cat_features].astype(str)
        #         catboost에서는 cat_feature를 파라미터로 지정하는데 이때 범주형 변수가 실수형이라면 돌아가지 않음. 
        #         에러 방지를 위해서 문자형으로 바꿔주기  

        # Set up 
        cat_model = CatBoostRegressor(
            cat_features=cat_features,
            verbose = False, eval_metric="RMSE"
        )
        
        fit_model = cat_model.fit(X_train, y_train, 
                                   eval_set=[(X_valid, y_valid)],
                                   use_best_model=True
                                 )
        cat_pred = fit_model.predict(X_valid)
        r2 = r2_oos(y_valid, cat_pred) ; print(r2)
        r2_list.append(r2)
    print(r2_list)    
    return np.mean(r2_list)       
    
# XGBoost
def refit_xgb(df, features, train_start=1996, train_end=2011, valid_size=5):
    r2_list = []
    for i in range(valid_size):
        start_year = train_start
        end_year = train_end + i
        train = df[(df['year']<=end_year)&(df['year']>=start_year)] ; print('train: ', start_year, end_year)
        valid = df[df['year']==end_year+1] ; print('valid: ', end_year+1)

        X_train, y_train = train[features], np.array(train['ret'])
        X_valid, y_valid = valid[features], np.array(valid['ret'])

        # model fitting
        import xgboost
        xgb = xgboost.XGBRegressor(learning_rate=0.1, max_depth=5, 
                                         n_estimators=100)
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_valid)
        r2 = r2_oos(y_valid, xgb_pred) ; print(r2)
        r2_list.append(r2)

    print(r2_list)    
    return np.mean(r2_list)

# LightGBM
def refit_lgb(df, features, train_start=1996, train_end=2011, valid_size=5):

    r2_list = []
    for i in range(valid_size):
        start_year = train_start
        end_year = train_end + i
        train = df[(df['year']<=end_year)&(df['year']>=start_year)] ; print('train: ', start_year, end_year)
        valid = df[df['year']==end_year+1] ; print('valid: ', end_year+1)

        X_train, y_train = train[features], np.array(train['ret'])
        X_valid, y_valid = valid[features], np.array(valid['ret'])

        # model fitting
        import lightgbm as lgbm 
        train_ds = lgbm.Dataset(X_train, label = y_train) 
        test_ds = lgbm.Dataset(X_valid, label = y_valid)

        params = {'learning_rate': 0.01, 'max_depth': 2, 
                  'boosting': 'gbdt', 'objective': 'regression', 
                  'metric': 'mse'} 
        lgb = lgbm.train(params, train_ds, 1000, test_ds, verbose_eval=False) 
        importance_plot = lgbm.plot_importance(lgb, figsize=(8,15), height=0.5, grid=False)
        lgb_pred = lgb.predict(X_valid)
        r2 = r2_oos(y_valid, lgb_pred) ; print(r2)
        r2_list.append(r2)

    print(r2_list)
    return np.mean(r2_list), importance_plot

