import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import pickle
from joblib import dump, load
from git import Repo

import os
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor,GradientBoostingRegressor

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from constants import mlMetadata, DATASETS_PATH, SIDS, savepath, mlResults, PATH_OF_GIT_REPO
from utils import data_importer, get_inputs, folderChecker, metaUpdater,git_push, NpEncoder
from enums import Model, Interval





import json

from constants import mlMetadata, DATASETS_PATH, SIDS, savepath, mlResults, PATH_OF_GIT_REPO
from utils import data_importer, model_trainer, get_inputs, folderChecker, metaUpdater,git_push
from enums import Model, Interval, Interpolator, Schema


def intevalExtractor(model,interval,X_train, y_train, X_test, l, u, y_test):
    """
        Generate bootstrap or quantile prediction interval
        Args:
            model: trained model
            interval: name of interval type according enums.py definition
            X_train: training data frame
            X_test: test data frame
            y_train: training data target
            y_test: test data prediction
            l: data frame where the lower prediction interval bounds are to be placed
            u: data frame where the upper prediction interval bounds are to be placed
        Returns:
            l,u : after bounds are placed
    
    """
    if interval == Interval.bootstrap.name:
        try:

            # Residual Bootsrapping  on validation data
            cv_value = min(X_train.shape[0],3)
            pred_train = cross_val_predict(model, X_train, y_train, cv=cv_value)

            res = y_train - pred_train

            ### BOOTSTRAPPED INTERVALS ###

            alpha = 0.1  # (90% prediction interval) #Hard Coded

            bootstrap = np.asarray([np.random.choice(res, size=res.shape) for _ in range(100)])
            q_bootstrap = np.quantile(bootstrap, q=[alpha / 2, 1 - alpha / 2], axis=0)


            l.loc[X_test.index, y_train.name] = y_test.values + q_bootstrap[0].mean()
            u.loc[X_test.index, y_train.name] = y_test.values + q_bootstrap[1].mean()
        except:
            if str(type(model)) == "<class 'sklearn.ensemble._gb.GradientBoostingRegressor'>":
                all_models = {}
                for alpha in [0.05, 0.95]:  # Hard Coded
                    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha,
                                                    max_depth=model.get_params['regressor__max_depth'],
                                                    n_estimators=model.get_params['regressor__n_estimators'])
                    all_models["q %1.2f" % alpha] = gbr.fit(X_train, y_train)
                    # For prediction

                lowVals = all_models["q 0.05"].predict(X_test)
                upVals = all_models["q 0.95"].predict(X_test)
            else:
                pred_Q = pd.DataFrame()
                for pred in model.estimators_:
                    temp = pd.Series(pred.predict(X_test))
                    pred_Q = pd.concat([pred_Q, temp], axis=1)
                quantiles = [0.05, 0.95]  # Hard Coded

                lowVals = pred_Q.quantile(q=quantiles[0], axis=1)
                upVals = pred_Q.quantile(q=quantiles[-1], axis=1)

            l.loc[X_test.index, y_train.name] = lowVals.values
            u.loc[X_test.index, y_train.name] = upVals.values

    else:
        if str(type(model)) == "<class 'sklearn.ensemble._gb.GradientBoostingRegressor'>":
            all_models = {}
            for alpha in [0.05, 0.95]:  # Hard Coded
                gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha,
                                                max_depth=model.get_params['regressor__max_depth'],
                                                n_estimators=model.get_params['regressor__n_estimators'])
                all_models["q %1.2f" % alpha] = gbr.fit(X_train, y_train)
                # For prediction

            lowVals = all_models["q 0.05"].predict(X_test)
            upVals = all_models["q 0.95"].predict(X_test)
        else:
            pred_Q = pd.DataFrame()
            for pred in model.estimators_:
                temp = pd.Series(pred.predict(X_test))
                pred_Q = pd.concat([pred_Q, temp], axis=1)
            quantiles = [0.05, 0.95]  # Hard Coded
            
            lowVals = pred_Q.quantile(q=quantiles[0], axis=1)
            upVals = pred_Q.quantile(q=quantiles[-1], axis=1)
            
            l.loc[X_test.index, y_train.name] = lowVals.values
            u.loc[X_test.index, y_train.name] = upVals.values
    return l,u
            
def posprocessor(imputer,originalData,transformedData,model):
    """
        Generates prediction interval and feature importances and OOB scores for each fit of iterative imputer. 
        Args:
            imputer: trained iterative imputer
            originalData: indicatorData subsetted according to selected years for imputation
            tansformedData: data transformed by imputer
            model: baseEstimator name
        Returns:
            lower: data frame of lower prediction intervals (similar format to transformedData)
            upper: data frame of upper prediction intervals
            performance: data frame of R2 and NRMSE scores
            indiator_importance: data frame of feature importance for each imputed indiactor


    """
    lower = pd.DataFrame(index = transformedData.index, columns = transformedData.columns)
    upper = pd.DataFrame(index = transformedData.index, columns = transformedData.columns)
    performance = pd.DataFrame(columns = ["target","rmse_deviation","model","R2"])
    indicator_importance = pd.DataFrame()

    interval=Interval.bootstrap.name
    for obj in tqdm(imputer.imputation_sequence_[-originalData.dropna(axis=1, how='all').shape[1]:]):
        targetColumn = transformedData.columns[obj[0]]
        print(targetColumn)
        trainIndex = originalData[originalData[targetColumn].notna()][targetColumn].index
        testIndex = originalData[originalData[targetColumn].isna()][targetColumn].index
        predColumns = transformedData.columns[obj[1]]
        best_model = obj[2]#.feature_importances_
        X_train = transformedData.loc[trainIndex,predColumns]
        y_train = transformedData.loc[trainIndex,targetColumn]
        X_test = transformedData.loc[testIndex,predColumns]
        y_test = transformedData.loc[testIndex,targetColumn]
        l,u = intevalExtractor(model = best_model,interval = interval,X_train = X_train, y_train = y_train, X_test = X_test, l = lower, u = upper, y_test = y_test)
        mse = mean_squared_error(y_train.values,best_model.oob_prediction_)
        rmse = np.sqrt(mse)
        nrmse = rmse/y_train.mean()
        if ((nrmse >1) | (nrmse <0)):
            nrmse = rmse/(y_train.max()-y_train.min())
        performance.loc[len(performance)] = [targetColumn,nrmse,"etr",best_model.oob_score_]
        
        feature_importance_bar = pd.DataFrame()
        feature_importance_bar["names"] = predColumns
        feature_importance_bar["values"] = best_model.feature_importances_
        feature_importance_bar["target"] = targetColumn
        feature_importance_bar["model"] = model
        #eature_importance_bar.set_index(["model","year","target"],inplace=True)
        indicator_importance=pd.concat([indicator_importance,feature_importance_bar])
        
    return lower, upper,performance,indicator_importance
    
def base_estiamtor(model,seed):
    """
        Return an instance of the base estimator (can be expanded according to model defintion in enums.py)
    """
    if model == Model.rfr.name:
        baseEstimator = RandomForestRegressor(max_features = 'auto', n_jobs = -1, warm_start = True, random_state = seed,bootstrap = True, oob_score = True)
    if model == Model.etr.name:
        baseEstimator = ExtraTreesRegressor(max_features = 'auto', n_jobs = -1, warm_start = True, random_state = seed,bootstrap = True, oob_score = True)
    return baseEstimator


# Turn into a API JSON format
def processMLData(predDictionary):
    modelCode = model_code[-1]
    jsonDict = dict()
    for datasetCode in predDictionary.keys():
        jsonDict[datasetCode]=dict()
        print(datasetCode)

        indicatorCodes = []
        for i in predDictionary[datasetCode].keys():
            indicatorCodes.extend(predDictionary[datasetCode][i]["prediction"].columns.tolist())
        indicatorCodes = list(set(indicatorCodes))
        for indicator in indicatorCodes:
            if indicator =="year":
                continue
            # If json file already exists, update that (important if running script in segments)
            if (os.path.exists(savepath+'model'+str(modelCode)+'/'+datasetCode+'/'+indicator+'.json') & (response == 'update')):
                with open(savepath+'model'+str(modelCode)+'/'+datasetCode+'/'+indicator+'.json') as json_file:
                        indicatorJson = json.load(json_file)

            else: 
                indicatorJson={"data":{},"upperIntervals":{},"lowerIntervals":{},"categoryImportances":{},"featureImportances":{}}
            for year in predDictionary[datasetCode].keys():
                    predictionsDf = predDictionary[datasetCode][year]["prediction"].reset_index().rename(columns={"level_1":"year"})
                    lowerIntervalsDf = predDictionary[datasetCode][year]["lower"].reset_index().rename(columns={"level_1":"year"})
                    upperIntervalsDf = predDictionary[datasetCode][year]["upper"].reset_index().rename(columns={"level_1":"year"})
                    featureImportancesDf = predDictionary[datasetCode][year]["importance"]
                    yearValues={}
                    upperIntervals={}
                    lowerIntervals={}

                    if indicator in predictionsDf.columns:

                        countries=predictionsDf["Country Code"].unique().tolist()
                        for country in countries:
                            value=predictionsDf[predictionsDf["Country Code"]==country][indicator].iloc[0]
                            lower=lowerIntervalsDf[lowerIntervalsDf["Country Code"]==country][indicator].iloc[0]
                            upper=upperIntervalsDf[upperIntervalsDf["Country Code"]==country][indicator].iloc[0]

                            if not pd.isna(value):
                                yearValues[country]=value

                            if not pd.isna(lower):
                                lowerIntervals[country]=lower

                            if not pd.isna(upper):
                                upperIntervals[country]=upper                       

                        indicatorFeaturesDf=featureImportancesDf[featureImportancesDf["predicted indicator"]==indicator]
                        features=indicatorFeaturesDf["feature indicator"].unique().tolist()
                        featureImportances={}
                        for feature in features:
                            featureImportance=indicatorFeaturesDf[indicatorFeaturesDf["feature indicator"]==feature]["feature importance"].iloc[0]
                            featureImportances[feature]=featureImportance

                        featuresMeta=indicatorMeta[indicatorMeta["Indicator Code"].isin(features)]
                        categories=featuresMeta["Category"].unique().tolist()

                        categoryImportances={}

                        for category in categories:
                            categoryTotal=0
                            for feature in featuresMeta[featuresMeta["Category"]==category]["Indicator Code"].unique().tolist():
                                importance=featureImportances[feature]
                                categoryTotal+=importance
                            categoryImportances[category]=categoryTotal

                        indicatorJson["data"][year]=yearValues
                        indicatorJson["upperIntervals"][year]=upperIntervals
                        indicatorJson["lowerIntervals"][year]=lowerIntervals
                        indicatorJson["featureImportances"][year]=featureImportances
                        indicatorJson["categoryImportances"][year]=categoryImportances
            if not os.path.exists(savepath+'model'+str(modelCode)+'/'+datasetCode):
                os.makedirs(savepath+'model'+str(modelCode)+'/'+datasetCode)
            with open(savepath+'model'+str(modelCode)+'/'+datasetCode+'/'+indicator+'.json', 'w') as outfile:
                json.dump(indicatorJson, outfile,cls=NpEncoder)
            #jsonDict[datasetCode][indicator] = indicatorJson
             
    #return jsonDict


if __name__ == '__main__':

    seed = 1

    predictors = 10

    #Important metadata
    with open(mlMetadata) as json_file:
        mlMetajson = json.load(json_file)


    # pull updated info from repo
    repo = Repo(PATH_OF_GIT_REPO)
    repo.remotes.origin.pull()

    #Inputs to guide modelling
    model =get_inputs("Select model name",[Model.rfr.name,Model.etr.name])#[e.name for e in Model if e != Model.all])

    start_year = get_inputs("year to start from? e.g. 2010")
    end_year = get_inputs("year to end at? e.g. 2019")
    
    iter = get_inputs("number of iterations to run")
    model_code,response = folderChecker()
    if (response in  ['replace','new']):
        mlMetajson = metaUpdater(mlMetajson, model_code,"iterative")

    # IMport data
    wb_data,indicatorMeta, datasetMeta, indicatorData = data_importer()
    idx = pd.IndexSlice

    #Slice data to required years
    data = wb_data.loc(axis=0)[idx[SIDS,[str(i) for i in range(start_year,end_year)]]]
    
    #Train iterative imputer
    rfr = base_estiamtor(model,seed)
    imp_mean = IterativeImputer(estimator = rfr, n_nearest_features = predictors, random_state = seed, max_iter = int(iter), verbose = 2)#update
    imp_mean.fit(data)
    transformed = imp_mean.transform(data)
    #dump(imp_mean, DATASETS_PATH+"Untitled Folder/"+'imp_mean_'+str(i)+'.joblib') 
    imputedData = pd.DataFrame(data = transformed, index = data.index, columns = data.dropna(axis=1, how='all').columns)
    if not os.path.exists(mlResults+ model_code + "/raw data from model"):
        os.makedirs(mlResults+ model_code+ "/raw data from model")
    imputedData.to_csv(mlResults + model_code + "/raw data from model" + "/"+start_year+"_"+end_year+"_"+model+"_predictions.csv",mode='a', header=not os.path.exists(mlResults + model_code + "/raw data from model" + "/"+start_year+"_"+end_year+"_"+model+"_predictions.csv"))

    # Create prediction intervals
    lower, upper,performance,importance = posprocessor(imp_mean,data,imputedData,"etr")
    
    lower.to_csv(mlResults + model_code + "/raw data from model" + "/"+start_year+"_"+end_year+"_"+model+"_lower.csv")
    upper.to_csv(mlResults + model_code + "/raw data from model" + "/"+start_year+"_"+end_year+"_"+model+"_upper.csv")
    performance.to_csv(mlResults + model_code + "/raw data from model" + "/"+start_year+"_"+end_year+"_"+model+"_performance.csv")
    importance.to_csv(mlResults + model_code + "/raw data from model" + "/"+start_year+"_"+end_year+"_"+model+"_indicator_importance.csv")

    # Setup data for processMLData function to match API format 
    # (This code along with processMLData should be refactored to remove unnecessary for-loop splits)
    large_dict = dict()
    datasets = indicatorMeta.groupby("Dataset")["Indicator Code"].apply(list)#.reset_index(name='indicators')
    cols = imputedData.columns
    targets = importance.target.values
    



    for d in datasets.index:
        large_dict[d]=dict()
        subPred = imputedData[list(set(cols) & set(datasets[d]))+["Country Code","year"]]
        subLower = lower[list(set(cols) & set(datasets[d]))+["Country Code","year"]]
        subUpper = upper[list(set(cols) & set(datasets[d]))+["Country Code","year"]]
        for y in subPred.year.unique():
            large_dict[d][y] = dict()
            large_dict[d][y]["prediction"] = subPred[subPred.year == y]
            large_dict[d][y]["lower"] = subLower[subLower.year == y]
            large_dict[d][y]["upper"] = subUpper[subUpper.year == y]
            large_dict[d][y]["importance"] = importance[importance.target.isin(datasets[d])][["names","values","target"]].rename(columns={"names":"feature indicator","values":"feature importance","target":"predicted indicator"})
            
            # Save these intermediate results
            if not os.path.exists(mlResults+ model_code +"/predictions"):
                os.makedirs(mlResults+ model_code +"/predictions")
            if not os.path.exists(mlResults+ model_code +"/prediction intervals/"):
                os.makedirs(mlResults+ model_code +"/prediction intervals/")
            if not os.path.exists(mlResults+ model_code +"/feature importances"):
                os.makedirs(mlResults+ model_code +"/feature importances")
            large_dict[d][y]["prediction"].to_csv(mlResults+ model_code +"/predictions"+"/"+d+"_predictions_"+str(y)+".csv")
            large_dict[d][y]["lower"].to_csv(mlResults+ model_code +"/prediction intervals/"+"lower"+"/"+d+"_lower_"+str(y)+".csv")
            large_dict[d][y]["upper"].to_csv(mlResults+ model_code +"/prediction intervals/"+"upper"+"/"+d+"_upper_"+str(y)+".csv")
            large_dict[d][y]["importance"].to_csv(mlResults+ model_code +"/feature importances"+"/"+d+"_feature_importance_"+str(y)+".csv")
    
    # Convert to API format
    processMLData(large_dict)

    #Update Metadata
    with open(mlMetadata, "w") as write_file:
        json.dump(mlMetajson, write_file, indent=4)
    # Push to git
    COMMIT_MESSAGE = ' '.join(['add:',model_code,"from",start_year,'to',end_year, "(",response,")"])  


    git_push(COMMIT_MESSAGE)

    