import numpy as np
import pandas as pd


# out-of-sample R^2
def r2_oos(true, pred):
    if sum(true**2) == 0:
        result = 'zero division error'
    else:
        result = 1 - sum((true - pred)**2) / sum(true**2)
        result = round(result, 4)
    return result

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

# DNN3
def refit_nn3(df, features, train_start=1996, train_end=2010, valid_size=5):

    r2_list = []
    for i in range(valid_size):
        start_year = train_start
        end_year = train_end + i
        train = df[(df['year']<=end_year)&(df['year']>=start_year)] ; print('train: ', start_year, end_year)
        valid = df[df['year']==end_year+1] ; print('valid: ', end_year+1)

        X_train, y_train = train[features], np.array(train['ret'])
        X_valid, y_valid = valid[features], np.array(valid['ret'])

        # model fitting
        from sklearn.neural_network import MLPRegressor
        # batch_size='auto'로 해야 잘 나옴..
        # 논문에는 batch_size=10000
        nn3_model = MLPRegressor(hidden_layer_sizes=(len(features),3), 
                         activation='relu', solver='adam', 
                          alpha=0.0001, batch_size='auto', 
                          learning_rate_init=0.01, 
                          random_state=None, 
                          tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
                          nesterovs_momentum=True, early_stopping=False
                          )
        
        nn3_model.fit(X_train, y_train)
        nn3_pred = nn3_model.predict(X_valid)
        r2 = r2_oos(y_valid, nn3_pred) ; print(r2)
        r2_list.append(r2)

    print(r2_list)
    return np.mean(r2_list)