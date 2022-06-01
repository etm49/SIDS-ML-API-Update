#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os

import time
#Data Manipulation
import pandas as pd
import numpy as np

# Propcessing and training
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor,GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_predict


#!pip install pca
from pca import pca
from tqdm import tqdm


def get_inputs(text: str, expect: [str] = None, default=None):
    ret = None
    while ret is None or ret == "":
        if expect is None:
            ret = input(text + " : ")
        else:
            while ret not in expect:
                ret = input(text + " " + str(expect) + " : ")

        if default is not None and ret is None or ret == "":
            ret = default
            break

    if ret is None:
        return default
    return ret




# In[3]:


# Inputs


seed = 100

#Maximum percentage of missingness to consider in target (target threshold) 
percent = 90 

#Maximum number of missingess to consider in predictors (predictor threshold)
measure = 40
star_year = get_inputs("year to start from? e.g. 2010")
end_year = get_inputs("year to end at? e.g. 2019")

supported_years = [str(x) for x in list(range(int(star_year), int(end_year)))]


DATASETS_PATH = "/Volumes/My Passport for Mac/jobs/UNDP/ML-IndicatorData/"
savepath = "../mlResults"


model =get_inputs("Select model name",['rfr','gbr','etr'])

model_code = get_inputs("Input model code in format Model 1, Model 2,...")
if model_code in os.scandir(savepath):
    response = get_inputs("model code already present in the mlResults. Would you like to continue with this code and replace existing folder?", ['y','n'])
    if response == 'n':
        model_code = get_inputs("Please enter different model code (format of Model 1, Model 2,...)")

SIDS = ['ASM', 'AIA', 'ATG', 'ABW', 'BHS', 'BRB', 'BLZ', 'BES', 'VGB', 'CPV', 'COM', 'COK', 'CUB', 'CUW', 'DMA', 'DOM',
        'FJI', 'PYF',
        'GRD', 'GUM', 'GNB', 'GUY', 'HTI', 'JAM', 'KIR', 'MDV', 'MHL', 'MUS', 'FSM', 'MSR', 'NRU', 'NCL', 'NIU', 'MNP',
        'PLW', 'PNG', 'PRI',
        'KNA', 'LCA', 'VCT', 'WSM', 'STP', 'SYC', 'SGP', 'SXM', 'SLB', 'SUR', 'TLS', 'TON', 'TTO', 'TUV', 'VIR', 'VUT']


# #### Helper Functions

# In[4]:


########## All functions for Two Level imputation model #########

# Import from disk
def cou_ind_miss(Data):
    """
        Returns the amount of missingness across the years in each indicator-country pair 
    """
    absolute_missing = Data.drop(columns=["Country Code","Indicator Code"]).isnull().sum(axis=1)
    total = Data.drop(columns=["Country Code","Indicator Code"]).count(axis=1)

    percent_missing = absolute_missing * 100 / Data.drop(columns=["Country Code","Indicator Code"]).shape[1]
    missing_value_df = pd.DataFrame({'row_name': Data["Country Code"]+"-"+Data["Indicator Code"],
                                 'Indicator Code':Data["Indicator Code"],
                                 'absolute_missing':absolute_missing,
                                 'total':total,
                                 'percent_missing': percent_missing})
    countyIndicator_missingness = missing_value_df.sort_values(["percent_missing","row_name"])

    return countyIndicator_missingness

def data_importer(percent=90,model_type="non-series",path = DATASETS_PATH):
    """
        
        Import csv files and restrructure the data into a country by indcator format. Model_type will be expanded upon.
        precent: the most tolerable amount of missingness in a column for an indicator  accross the years
        model_type: type of model data imported for
        path: path on disk the raw data is stored
        
        wb_data: indicatorData restructed to a (country x year) by Indicator Code format
        indicatorMeta: indicator meta dataset (as is)
        indicatorData: indicator data dataset (as is)
        datasetMeta: dataset meta data (as is)
    """
    #


    indicatorMeta = pd.read_csv(path + "indicatorMeta.csv")

    datasetMeta = pd.read_csv(path + "datasetMeta.csv")

    indicatorData = pd.read_csv(path + "indicatorData.csv")


    #### Remove rows with missing country or indicator names
    indicatorData["Country/Indicator Code"] = indicatorData["Country Code"]+"-"+indicatorData["Indicator Code"]
    indicatorData= indicatorData[indicatorData["Country/Indicator Code"].notna()].drop(columns="Country/Indicator Code")

    if model_type == "series":
        # Indicators measured less than 5 times for each country are removed
        countyIndicator_missingness = cou_ind_miss(indicatorData)
        indicator_subset = set(countyIndicator_missingness[countyIndicator_missingness.percent_missing >= percent]["Indicator Code"])- set(countyIndicator_missingness[countyIndicator_missingness.percent_missing<percent]["Indicator Code"])

        indicatorData=indicatorData[~indicatorData["Indicator Code"].isin(indicator_subset)]

    wb_data = indicatorData.set_index(['Country Code', 'Indicator Code'])
    wb_data = wb_data.stack()
    wb_data = wb_data.unstack(['Indicator Code'])
    wb_data = wb_data.sort_index()

    indicatorMeta=indicatorMeta[indicatorMeta["Indicator Code"].isin(indicatorData["Indicator Code"].values)]
    indicatorMeta=indicatorMeta[indicatorMeta.Indicator.notna()]

    datasetMeta=datasetMeta[datasetMeta["Dataset Code"].isin(indicatorMeta["Dataset"].values)]
    datasetMeta=datasetMeta[datasetMeta["Dataset Name"].notna()]

    return wb_data,indicatorMeta, datasetMeta, indicatorData


# Preprocess
def missingness(df):
    "Rank the columns of df by the amount of missing observations"
    absolute_missing = df.isnull().sum()
    percent_missing = absolute_missing * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'absolute_missing': absolute_missing,
                                     'percent_missing': percent_missing})
    return missing_value_df.sort_values(["percent_missing","column_name"])


def preprocessing(data,target, target_year,interpolator,SIDS, percent=measure):

    """
    Preprocess data into a format suitable for the two step imputation model by filling the most complete
    Args:
        data: indicatorData dataset 
        target: indicator whose values will be imputed
        target_year: the year under consideration
        interpolator: type of imputer to use for interpolation
        precent: the most tolerable amount of missingness in a column

    Returns:
        X_train: training data
        X_test:  testing data (observation with missing values for target variable)
        y_train: training target series
    """

    #Subset data for target and target_year
    data_sub = data[["Country Code","Indicator Code",str(target_year)]]
    data_sub = data_sub.set_index(["Country Code","Indicator Code"])[str(target_year)].unstack(level=1)
    
    X_train = data_sub[data_sub[target].notna()]
    X_test = data_sub[(data_sub[target].isna())&(data_sub.index.isin(SIDS))]
    
    y_train = X_train.pop(target)
    y_test = X_test.pop(target)
    
    #Find how much missing values are there for each indicator
    rank = missingness(X_train)

    #interpolation for indcators missing less than percent% using KNN imputer 
    most_complete = rank[rank.percent_missing < percent]["column_name"].values 
    
    X_train = X_train[most_complete]
    X_test = X_test[most_complete]
    
    # How muc does fiting only on X_train affect fits (perhaps another layer of performance via CV)
    if interpolator == 'KNNImputer':
        scaler = MinMaxScaler()
        imputer = KNNImputer(n_neighbors=5) #Hard Coded
        scaler.fit(X_train)
        imputer.fit(scaler.transform(X_train))
        
        X_train = pd.DataFrame(data=scaler.inverse_transform(imputer.transform(scaler.transform(X_train)))
                               ,columns=X_train.columns,
                              index=X_train.index)
        
        X_test = pd.DataFrame(data=scaler.inverse_transform(imputer.transform(scaler.transform(X_test)))
                       ,columns=X_test.columns,
                      index=X_test.index)


    elif interpolator == 'SimpleImputer':
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean') #Hard coded
        imp_mean.fit(X_train)
        X_train = pd.DataFrame(data=imp_mean.transform(X_train)
                               ,columns=X_train.columns,
                              index=X_train.index)
        X_test = pd.DataFrame(data=imp_mean.transform(X_test)
                               ,columns=X_test.columns,
                              index=X_test.index)

    else:
        imp = IterativeImputer(missing_values=np.nan,random_state=0, estimator=ExtraTreesRegressor(n_estimators=10, random_state=0),n_nearest_features=100, 
                         add_indicator=False,sample_posterior=False) # Hard Coded values
        imp.fit(X_train)
        X_train = pd.DataFrame(data=imp.transform(X_train)
                               ,columns=X_train.columns,
                              index=X_train.index)
        X_test = pd.DataFrame(data=imp.transform(X_test)
                               ,columns=X_test.columns,
                              index=X_test.index)

    return X_train,X_test, y_train

# Select features/ reduce dimensionality
def feature_selector(X_train,y_train,manual_predictors):
    """
        Implement the Recursive feature selection for automatic selection of predictors for model

        returns: A boolean list of which features should be considered for prediction
    """
    # ESTIMATOR STILL UNDER INVESTIGATION: FOR NOW TAKE ON WITH HIGH DIMENSIONALITY TOLERANCE
    estimator= RandomForestRegressor()

    # STEP SIZE UNDER INVESTIGATION: FOR NOW TAKE ONE THAT REDUCES COMPUTATION TIME WITHOUT JUMPING
    selector = RFE(estimator,n_features_to_select=manual_predictors,step= manual_predictors)
    selector.fit(X_train,y_train)
    return selector.support_

def feature_selection(X_train,X_test,y_train,target, manual_predictors,scheme, meta):
    """
        Returns the training and testing/prediction data with a reduced feature space
    Args:
        X_train: training data
        X_test: prediction data
        y_train: training target array
        target: name of target array
        manual_predictors: number of predictors (for Automatic) or list of predictors (for Manual)
        scheme: feature selection method selected by user
    Returns:
        X_train: reduced training data
        X_test: reduced testing data
    """

    if scheme == "Automatic via feature selection":
        
        # Take the most import predictor_number number of independent variables (via RFE) and plot correlation
        importance_boolean = feature_selector(X_train=X_train,y_train=y_train,manual_predictors=manual_predictors)
        prediction_features = (X_train.columns[importance_boolean].tolist())
    if scheme == "PCA":
        PCA =pca()
        
        out = PCA.fit_transform(X_train)
        prediction_features = list(out['topfeat'].iloc[list(range(manual_predictors)),1].values)
    

    else:
        prediction_features= manual_predictors

    X_train = X_train[prediction_features]
    X_test = X_test[prediction_features]
    
    correlation = X_train[prediction_features].corr()
    correlation.index = prediction_features
    correlation.columns = prediction_features
    #names = meta[meta["Indicator Code"].isin(prediction_features)].Indicator.values

    #correlation.index = names
    #correlation.columns = names
        
    return X_train,X_test,correlation.to_dict(orient ='split')

# Train model and predict
def model_trainer(X_train,X_test,y_train,seed,n_estimators, model,interval):
    """
    Train the selected model, cross validate, score and generate a 90% prediction interval based on bootstrapped residuals.
    Args:
        X_train: training data
        X_test: prediction data
        y_train: training target array
        seed: random state setter
        n_estimators: number of trees for tree based models
        model: type of model to be trained
        interval: type of prediction interval
    Returns:
        prediction: dataframe with prediction values and confidence interval
        rmse: root mean squared error of the model
        gs: trained GridSearchCV model
        best_model: the best model
    """
    sample_weight = []#X_train.pop("sample_weight")
    sids_weights = (X_train.index.isin(SIDS)).sum()
    total = X_train.shape[0]

    # Inverse class weighting for SIDS and non-SIDS
    for i in X_train.index:
        if i in SIDS:
            sample_weight.append(1/sids_weights)
        else:
            sample_weight.append(1/(total-sids_weights))


    if model == "all":
        model = ["rfr","etr","gbr","svr"]
    model_instances=[]
    params= []

    num_folds = 5 # Hard coded
    #scoring = ['neg_root_mean_squared_error','neg_median_absolute_error','explained_variance']
    scoring = 'neg_mean_squared_error'
    if "rfr" in  model:
        clf1 = RandomForestRegressor(random_state = seed, n_jobs=-1)
        param1 = {}
        param1['regressor__n_estimators'] = [500]#[n_estimators]
        param1['regressor__max_depth'] = [5, 10, 20,100, None] # Hard coded
        param1['regressor'] = [clf1]
        model_instances.append(clf1)
        params.append(param1)
    if "etr" in model:
        clf2 = ExtraTreesRegressor(random_state = seed, n_jobs=-1)
        param2 = {}
        param2['regressor__n_estimators'] = [500]#[n_estimators]
        param2['regressor__max_depth'] = [5, 10, 20,100, None]# Hard coded
        param2['regressor'] = [clf2]
        model_instances.append(clf2)
        params.append(param2)

    if "gbr" in model:
        clf3 = GradientBoostingRegressor(random_state = seed)
        param3 = {}
        if interval == "quantile":
            param3['regressor__loss'] = ['quantile']
            param3['regressor__alpha'] = [0.5] # hard coded
        param3['regressor__n_estimators'] = [100]#[n_estimators]
        param3['regressor__max_depth'] = [3,5, 10, 20, None]# Hard coded
        param3['regressor'] = [clf3]
        model_instances.append(clf3)
        params.append(param3)
    pipeline = Pipeline([('regressor', model_instances[0])])

    kwargs = {pipeline.steps[-1][0] + '__sample_weight': sample_weight}


    gs = GridSearchCV(pipeline, params, cv=num_folds, n_jobs=-1, scoring=scoring, refit=True, verbose=10).fit(X_train, y_train,**kwargs)
    rmse = np.sqrt(-gs.best_score_)

    best_model = gs.best_estimator_["regressor"]

    prediction = pd.DataFrame(gs.predict(X_test), columns=["prediction"], index=X_test.index)


    if interval == "bootstrap":

        #Residual Bootsrapping  on validation data
        pred_train = cross_val_predict(best_model,X_train, y_train, cv=3)

        res = y_train - pred_train

        ### BOOTSTRAPPED INTERVALS ###

        alpha = 0.1 #(90% prediction interval) #Hard Coded

        bootstrap = np.asarray([np.random.choice(res, size=res.shape) for _ in range(100)])
        q_bootstrap = np.quantile(bootstrap, q=[alpha/2, 1-alpha/2], axis=0)

        #prediction = pd.DataFrame(gs.predict(X_test), columns=["prediction"], index=X_test.index)
        prediction["upper"]= prediction["prediction"] + q_bootstrap[1].mean()
        prediction["lower"]= prediction["prediction"] + q_bootstrap[0].mean()

    else:
        if str(type(best_model))== "<class 'sklearn.ensemble._gb.GradientBoostingRegressor'>":
            all_models = {}
            for alpha in [0.05, 0.95]: # Hard Coded
                gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha,max_depth=gs.best_params_['regressor__max_depth'],n_estimators=gs.best_params_['regressor__n_estimators'])
                all_models["q %1.2f" % alpha] = gbr.fit(X_train, y_train)
                #For prediction

            prediction["lower"]= all_models["q 0.05"].predict(X_test)
            prediction["upper"]= all_models["q 0.95"].predict(X_test)
        else:
            pred_Q = pd.DataFrame()
            for pred in best_model.estimators_:
                temp = pd.Series(pred.predict(X_test))
                pred_Q = pd.concat([pred_Q,temp],axis=1)
            quantiles = [0.05, 0.95] # Hard Coded

            for q in quantiles:
                s = pred_Q.quantile(q=q, axis=1)
                prediction[str(q)] = s.values
            prediction.rename(columns={"0.05":"lower","0.95":"upper"}, inplace=True) # Column names are hard coded

    # Predict for SIDS countries with missing values
    prediction = prediction[prediction.index.isin(SIDS)]
    prediction = prediction.reset_index().rename(columns={"index":"country"})

    return prediction,rmse,gs, best_model


# In[5]:


def sids_top_ranked(target_year,data,sids, percent,indicator_type="target"):
    """Return a list with indicators with less than "percent" amount of missingness over sids COUNTRIES only """
    sub_data = data[["Country Code","Indicator Code",str(target_year)]]
    sub_data = sub_data[sub_data["Country Code"].isin(sids)]
    sub_data = sub_data.set_index(["Country Code","Indicator Code"])[str(target_year)].unstack(level=1)
    rank = missingness(sub_data)
    if indicator_type == "target":
        top_ranked = rank[(rank.percent_missing < percent)&(rank.percent_missing > 0)]["column_name"].values
    else:
        top_ranked = rank[(rank.percent_missing < percent)]["column_name"].values
    return top_ranked
def total_top_ranked(target_year,data,sids, percent,indicator_type="target"):
    """Return a list with indicators with less than "percent" amount of missingness over ALL COUNTRIES (sids input not used)"""
    sub_data = data[["Country Code","Indicator Code",str(target_year)]]
    sub_data = sub_data.set_index(["Country Code","Indicator Code"])[str(target_year)].unstack(level=1)
    rank = missingness(sub_data)
    if indicator_type == "target":
        top_ranked = rank[(rank.percent_missing < percent)&(rank.percent_missing > 0)]["column_name"].values
    else:
        top_ranked = rank[(rank.percent_missing < percent)]["column_name"].values
    return top_ranked


# In[6]:


def query_and_train(model,supported_years,SIDS =SIDS,percent=percent,measure=measure,seed=seed):
    predictions = pd.DataFrame()
    indicator_importance = pd.DataFrame()
    category_importance = pd.DataFrame()
    performance = pd.DataFrame()
    for i in tqdm(supported_years):
        targets = total_top_ranked(target_year=i,data=indicatorData,sids=SIDS, percent=percent,indicator_type="target")
        #predictors = total_top_ranked(target_year=i,data=indicatorData,SIDS=SIDS, percent=measure,indicator_type="predictor")
        print("target selected")
        for j in targets:
            manual_predictors=10
            target_year=i
            target=j
            interpolator='KNNImputer'
            scheme="PCA"

            t0 = time.time()
            # Train,test (for prediction not validation) split and preprocess
            X_train,X_test,y_train = preprocessing(data=indicatorData,target=target, target_year=target_year,interpolator=interpolator,SIDS=SIDS,percent=percent)
            # Dimension reduction based on scheme
            X_train,X_test,correlation = feature_selection(X_train,X_test,y_train,target, manual_predictors,scheme,indicatorMeta)
            data_code = indicatorMeta[indicatorMeta["Indicator Code"]==j].Dataset.values[0]
            #t1 = time.time()
            #timer
            #sel_time = t1 - t0    
            print(i+"_"+j)
            #for k in ["rfr"]:
            #t0= time.time()
            estimators=100
            k=model # delete when looping
            interval="quantile"
            SIDS=SIDS
            seed=seed
            percent=measure

            # training and prediction for X_test
            prediction,rmse,gs, best_model = model_trainer(X_train,X_test,y_train,seed,estimators, model,interval)
            #feature_importance_bar = dict()

            t1 = time.time()
            #timer
            train_time = t1 - t0  
            #print("feature_importance_bar")
            feature_importance_bar = pd.DataFrame()
            feature_importance_bar["names"] = best_model.feature_names_in_.tolist()
            feature_importance_bar["values"] = best_model.feature_importances_.tolist()
            feature_importance_bar["year"] = i
            feature_importance_bar["target"] = j
            feature_importance_bar["model"] = k
            feature_importance_bar.set_index(["model","year","target"],inplace=True)
            indicator_importance=pd.concat([indicator_importance,feature_importance_bar])
            # Make dataframes of feature importances for bar and pie visuals
            features = indicatorMeta[indicatorMeta["Indicator Code"].isin(X_train.columns)]
            feature_importance_pie =pd.DataFrame(data={"category":features.Category.values,"values":gs.best_estimator_._final_estimator.feature_importances_}).groupby("category").sum().reset_index()#.to_dict(orient="list")
            #print("feature_importance_pie")
            feature_importance_pie["year"] = i
            feature_importance_pie["target"] = j
            feature_importance_pie["model"] = k
            feature_importance_pie.set_index(["model","year","target"],inplace=True)
            category_importance=pd.concat([category_importance,feature_importance_pie])


            #t1 = time.time()
            #timer
            #exec_time = t1 - t0

            # Calculate mean normalized root mean squared error
            value_for_si = y_train.mean()
            rmse_deviation = rmse/value_for_si
            if ((rmse_deviation <0) | (rmse_deviation>1)):
                rmse_deviation = rmse/(y_train.max()-y_train.min())

            perfor = pd.DataFrame()
            print(perfor.shape)
            perfor["rmse"]=[rmse]
            perfor["rmse_deviation"]=[rmse_deviation]
            perfor["time"]=[train_time]
            perfor["year"] = [i]
            perfor["target"] = [j]
            perfor["model"] = [k]

            perfor.set_index(["model","year","target"])

            performance = pd.concat([performance,perfor])

            prediction["year"] = i
            prediction["target"] = j
            prediction["model"] = k
            prediction.set_index(["model","year","target"])
            predictions=pd.concat([predictions,prediction])

    #predictions.to_csv(DATASETS_PATH+supported_years[0]+"_"+model+"_predictions.csv")
    #indicator_importance.to_csv(DATASETS_PATH+supported_years[0]+"_"+model+"_indicator_importance.csv")
    #category_importance.to_csv(DATASETS_PATH+supported_years[0]+"_"+model+"_category_importance.csv")
    #performance.to_csv(DATASETS_PATH+supported_years[0]+"_"+model+"_performance.csv")
    return predictions,indicator_importance,category_importance,performance


# In[7]:


def replacement(dataset,year, save_path, ind_data, ind_meta, sids, pred):
    pred["dataset"] = pred.target.apply(lambda x: ind_meta[ind_meta["Indicator Code"]==x].Dataset.values[0])
    idx= pd.IndexSlice
    dataset_codes = ind_meta[ind_meta.Dataset==dataset]["Indicator Code"].values.tolist()
    subset_data = ind_data[ind_data["Indicator Code"].isin(dataset_codes)][["Country Code","Indicator Code",str(year)]].set_index(["Country Code","Indicator Code"]).stack().unstack("Indicator Code")
    subset_data = subset_data.loc[idx[sids,:],:]
    sub_pred = pred[(pred.year == year)&(pred.dataset==dataset)]#[["Country Code","prediction","target","year","dataset"]]
    #sub_pred = sub_pred.drop(columns="dataset").set_index(["target","Country Code","year"]).stack().unstack("target")#.index.droplevel(2)
    columns = np.unique(sub_pred.target).tolist()
    subset_data = subset_data[columns]
    #print(subset_data)
    try:
        assert subset_data.isna().sum().sum() > 0, f"number of missing in subset_data is 0 for " + dataset + " in " + str(year)
    except:
        print("number of missing in subset_data is 0 for " + dataset + " in " + str(year))
    results = subset_data.copy()
    lower = subset_data.copy()                                                                                                                                  
    upper = subset_data.copy()                                                                                                                                 
    for i in subset_data.index:
                for j in subset_data.columns:
                    value = subset_data.loc[i,j]
                    if np.isnan(value):

                        results.loc[i,j] = sub_pred[(sub_pred["Country Code"]==i[0])&(sub_pred["target"]==j)].prediction.values[0]#sub_data.loc[i,j]
                        lower.loc[i,j] = sub_pred[(sub_pred["Country Code"]==i[0])&(sub_pred["target"]==j)].lower.values[0]
                        upper.loc[i,j] = sub_pred[(sub_pred["Country Code"]==i[0])&(sub_pred["target"]==j)].upper.values[0]
                                                                                                                                  
                    else:
                        lower.loc[i,j] = np.nan
                        upper.loc[i,j] = np.nan
    for i in np.unique(sub_pred["Country Code"]).tolist():
        try:
            assert i in results.index.levels[0], f"cannot find "+ i + " for dataset " + dataset + " and year " + str(year)
        except:
            print(f"cannot find "+ i + " for dataset " + dataset + " and year " + str(year))
            missed = pred[(pred.year == year)&(pred.dataset==dataset)&(pred["Country Code"]==i)]
            p = pd.DataFrame(data = [missed.prediction.values], columns=missed.target.values, index=[(i,year)])
            l = pd.DataFrame(data = [missed.lower.values], columns=missed.target.values, index=[(i,year)])
            u = pd.DataFrame(data = [missed.upper.values], columns=missed.target.values, index=[(i,year)])
            results=pd.concat([results,p])
            lower=pd.concat([lower,l])
            upper=pd.concat([upper,u])
    path = save_path+model_code
    if  not os.path.exists(path):    
        os.makedirs(path)

    if not os.path.exists(path+"/predictions"): 
        os.makedirs(path+"/predictions") 

    if not os.path.exists(path+"/prediction intervals"): 
        os.makedirs(path+"/prediction intervals")

    results.to_csv(save_path+model_code+"/"+"predictions"+"/"+dataset+"_predictions_"+str(year)+".csv")
    lower.to_csv(save_path+model_code+"/"+"prediction intervals"+"/"+dataset+"_lower_"+str(year)+".csv")
    upper.to_csv(save_path+model_code+"/"+"prediction intervals"+"/"+dataset+"_upper_"+str(year)+".csv")
        
                                                               
                        


# In[8]:


def importance_script(save_path,importance):
    importance["dataset"] = importance.target.apply(lambda x: indicatorMeta[indicatorMeta["Indicator Code"]==x].Dataset.values[0])
    importance.rename(columns={"target":"predicted indicator","names":"feature indicator","values":"feature importance"},inplace=True)
    path = save_path+model_code
    if  not os.path.exists(path):    
        os.makedirs(path)
    if not os.path.exists(path+"/"+"feature importances"):
        os.makedirs(path+"/"+"feature importances")
    for d in np.unique(importance.dataset.values):
        possible_years = np.unique(importance[importance.dataset==d].year.values)
        for y in possible_years:
            data = importance[((importance.year == y)&(importance.dataset==d))][["predicted indicator","feature indicator","feature importance"]]
            data.to_csv(path+"/"+"feature importances/" + d+"_feature_importance_"+str(y)+".csv")


# # Run

# In[9]:


wb_data,indicatorMeta, datasetMeta, indicatorData = data_importer()
import warnings
warnings.filterwarnings("ignore")


# In[10]:


predictions,indicator_importance,category_importance,performance = query_and_train(model,supported_years)


# In[ ]:
targets = np.unique(predictions.target.values)
datasets = np.unique(indicatorMeta[indicatorMeta["Indicator Code"].isin(targets)].Dataset.values)

for d in datasets:
    print(d)
    for y in np.unique(predictions[predictions.dataset == d].year.values):
        replacement(d,y,savepath, ind_data = indicatorData, ind_meta=indicatorMeta, sids=SIDS, pred=predictions)


# In[ ]:


importance_script(savepath,indicator_importance)


# In[ ]:




