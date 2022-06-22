import pandas as pd
import numpy as np
import json
import ast
import time


import os
from git import Repo
from tqdm import tqdm


from enums import Model, Interval
from constants import mlMetadata, DATASETS_PATH, SIDS, savepath, mlResults, PATH_OF_GIT_REPO
from utils import data_importer, model_trainer, get_inputs, folderChecker, metaUpdater


n= 10 #indicator cannot ignore more than this many sids countries
m = 5 #indicator must have measurements for atleast 1/mth of the total number of years in the indicatorData

seed = 100

with open(mlMetadata) as json_file:
    mlMetajson = json.load(json_file)

repo = Repo(PATH_OF_GIT_REPO)
repo.remotes.origin.pull()


#Inputs to guide modelling
model =get_inputs("Select model name",['rfr','gbr','etr'])

start_year = get_inputs("year to start from? e.g. 2010")
end_year = get_inputs("year to end at? e.g. 2019")

supported_years = list(range(int(start_year), int(end_year)))

model_code, response =  folderChecker()
if (response in  ['replace','new']):
    mlMetajson = metaUpdater(mlMetajson, model_code,"timeseries")

# Preprocess
def series_extractor(indicator, ind_data,method,direction='both',d=1):
    idx=pd.IndexSlice
    data = ind_data.loc[idx[indicator,SIDS],].copy()
    #missing = data.columns[data.isnull().any()].tolist()
    if method in ["spline","polynomial"]:
 
        imp_data = data.copy().interpolate(method=method,order=d,axis=1,limit_direction=direction).dropna(axis=1, how='all')
    else:
        imp_data = data.copy().interpolate(method=method,axis=1,limit_direction=direction).dropna(axis=1, how='all').dropna(axis=1, how='all')


    #missing_years = list(set(missing) & set(imp_data.columns))
    return imp_data




def missing_sids(indicator):
    """SIDS that are never measured for this indicator"""
    return len(list(set(SIDS)-set(indicatorData.loc(axis=0)[pd.IndexSlice[indicator,]].index)))
    
def missing_years(indicator):
    sumb = indicatorData.loc(axis=0)[pd.IndexSlice[indicator,]]
    sumb=sumb.isna().sum(axis=0).sort_values()/sumb.shape[0]
    return sumb[sumb>0.5].index.shape[0],sorted(sumb[sumb>0.5].index.tolist()), sorted(sumb[sumb<0.5].index.tolist())



def validity_check(ind_data,sids_count,years_count):
    #ind_data = ind_data.set_index(["Indicator Code","Country Code"]).sort_index(axis=1).dropna(axis=0,how='all')#.interpolate('linear')

    indicators = []
    missing_sids_list = []
    missing_years_list = []
    missing_years_count_list = []
    actual_years_list = []
    for i in indicatorData.index.levels[0]:
        indicators.append(i)

        try:
            missing_sids_list.append(missing_sids(i))
            c,my,y = missing_years(i)
            missing_years_count_list.append(c)
            missing_years_list.append(my)
            actual_years_list.append(y)
        except:
            missing_sids_list.append(50)
            missing_years_count_list.append(50)
            actual_years_list.append([])
            missing_years_list.append([])
    validity = pd.DataFrame(data={"Indicator":indicators,"missing_sids":missing_sids_list,"missing_years_count":missing_years_count_list,"missing_years":missing_years_list, "available_years":actual_years_list})
    return validity[(validity.missing_sids<sids_count) &(validity.missing_years_count<(len(indicatorData.columns)/years_count))]

def preprocessing(ind_data, predictors,target_year,target):
    dataCopy = ind_data.copy()
    #dataCopy = dataCopy.set_index(["Indicator Code","Country Code"]).sort_index(axis=1).dropna(axis=0,how='all')#.interpolate('linear')

    #rename_names = dict()
    #for i in dataCopy.columns:
    #    rename_names[i] = int(i)
        
    #dataCopy.rename(columns=rename_names,inplace=True)

    data = series_extractor(indicator=predictors,ind_data=dataCopy,method='linear',direction="both")

    restructured_data = pd.DataFrame()

    sample_weight = pd.DataFrame()

    restructured_target = pd.DataFrame()

    window_counter =1
    year = target_year
    while (year-3) > min(2000,target_year-15): # hard coded

        sub = data.loc(axis=1)[range(year-3,year)]

        sub = pd.DataFrame(data=sub.to_numpy(), index=sub.index,columns=["lag 3","lag 2","lag 1"]).unstack("Indicator Code").swaplevel(axis=1).sort_index(axis=1)
        sub["window"] = year
        sub.set_index('window',append=True,inplace=True)


        restructured_data = pd.concat([restructured_data,sub])
        weight= 1/window_counter
        sample_weight = pd.concat([sample_weight,pd.DataFrame(data=[weight]*sub.shape[0],index=sub.index,columns=["weight"])])
        window_counter = window_counter+1
        idx=pd.IndexSlice
        target_data = dataCopy.loc[idx[target,SIDS],].copy()
        
        target_sub = target_data.loc(axis=1)[year]
        target_sub = pd.DataFrame(data=target_sub.to_numpy(), index=target_sub.index,columns=["target"]).unstack("Indicator Code").swaplevel(axis=1).sort_index(axis=1)
    #     for i in sub.index:
    #         if i not in target_sub.index:
    #             target_sub.loc[i,target_sub.columns[0]] = [np.nan]
        target_sub["window"] = year
        target_sub.set_index('window',append=True,inplace=True)
        restructured_target = pd.concat([restructured_target,target_sub])
        
        year = year-1
    
    restructured_data.dropna(axis=0,inplace=True)

    training_data = restructured_data.merge(restructured_target,how='left',left_index=True,right_index=True)
    training_data = training_data.merge(sample_weight,how='left',left_index=True,right_index=True)
    X_train= training_data[training_data[(target,"target")].notna()]
    X_test = training_data[training_data[(target,"target")].isna()]
    X_test.pop("weight")
    sample_weight = X_train.pop("weight")
    X_test.pop((target,"target"))
    y_train = X_train.pop((target,"target"))
    X_test = X_test.loc(axis=0)[pd.IndexSlice[:,target_year]]

    return X_train,X_test,y_train,sample_weight




prediction, rmse, gs, best_model = model_trainer(X_train, X_test, y_train, seed, n_estimators, model_type, interval,sample_weight)
prediction.droplevel(1)

######################################################################################################

def query_and_train(model,supported_years,SIDS =SIDS,seed=seed):
    predictions = pd.DataFrame()
    indicator_importance = pd.DataFrame()
    performance = pd.DataFrame()
    valid_predictors = validity_check(indicatorData,n,m).sort_values(by=["missing_years_count","missing_sids"]).Indicator.values.tolist()[:10]
    valid_targets = validity_check(indicatorData, 50,2).sort_values(by=["missing_years_count","missing_sids"],ascending=False).Indicator.values.tolist()
    k= 0
    for i in supported_years:
        
        for j in valid_targets:
            k=k+1
            print(k)
    #for j in valid_targets:
    #    for i in valid_target[valid_target.Indicator==j].available_years.values[0]
            
            t0 = time.time()
            predictors = valid_predictors
            if j in valid_predictors:
                predictors = list(set(valid_predictors)-set([j]))
            estimators=100
            interval="quantile"
            if model in [Model.esvr.name,Model.sdg.name,Model.nusvr, Model.lsvr.name, Model.xgbr.name, Model.lgbmr.name]:
                interval = "bootstrap"
            SIDS=SIDS
            seed=seed
            try: 
                X_train,X_test,y_train,sample_weight = preprocessing(indicatorData, predictors,i,j)
            except:
                continue
            prediction, rmse, gs, best_model = model_trainer(X_train, X_test, y_train, seed, estimators, model, interval,sample_weight)
            prediction = prediction.droplevel(1)
            #print(prediction)
            t1 = time.time()
            #timer
            train_time = t1 - t0  
            #print("feature_importance_bar")
            if model in [Model.esvr.name,Model.sdg.name,Model.nusvr, Model.lsvr.name]:
                feature_importances = best_model.coef_.tolist()
            else:
                feature_importances = best_model.feature_importances_.tolist()
            try: 
                feature_names = best_model.feature_names_in_.tolist()
            except:
                #feature_names = best_model.feature_name_
                feature_names = X_train.columns.tolist()

            feature_importance_bar = pd.DataFrame()
            feature_importance_bar["names"] = feature_names
            feature_importance_bar["values"] = feature_importances#best_model.feature_importances_.tolist()
            feature_importance_bar["year"] = i
            feature_importance_bar["target"] = j
            feature_importance_bar["model"] = model
            feature_importance_bar.set_index(["model","year","target"],inplace=True)
            indicator_importance=pd.concat([indicator_importance,feature_importance_bar])
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
            perfor["model"] = [model]

            perfor.set_index(["model","year","target"])

            performance = pd.concat([performance,perfor])

            prediction["year"] = i
            prediction["target"] = j
            prediction["model"] = model
            prediction.set_index(["model","year","target"])
            predictions=pd.concat([predictions,prediction])
    return predictions,indicator_importance.reset_index(),performance,gs

def replacement(dataset,year, save_path, ind_data = indicatorData, ind_meta=indicatorMeta, sids=SIDS, pred=prediction):
    idx= pd.IndexSlice
    dataset_codes = indicatorMeta[indicatorMeta.Dataset==dataset]["Indicator Code"].values.tolist()
    subset_data = ind_data[ind_data["Indicator Code"].isin(dataset_codes)][["Country Code","Indicator Code",str(year)]].set_index(["Country Code","Indicator Code"]).stack(dropna=False).unstack("Indicator Code")
    subset_data = subset_data.loc[idx[SIDS,:],:]
    sub_pred = pred[(pred.year == year)&(pred.dataset==dataset)]#[["Country Code","prediction","target","year","dataset"]]
    #sub_pred = sub_pred.drop(columns="dataset").set_index(["target","Country Code","year"]).stack().unstack("target")#.index.droplevel(2)
    columns = np.unique(sub_pred.target).tolist()
    if not all(elem in subset_data.columns.tolist()  for elem in columns):
        print(columns)
        print(subset_data.columns)
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
                    #print(value)
                    if np.isnan(value):
                        #n=n+1
                        #if n in [1,10,50,100,200,1000]:
                            #print(value)
                        #print(i)
                        #print(j)
                        try:
                            results.loc[i,j] = sub_pred[(sub_pred["Country Code"]==i[0])&(sub_pred["target"]==j)].prediction.values[0]#sub_data.loc[i,j]
                            lower.loc[i,j] = sub_pred[(sub_pred["Country Code"]==i[0])&(sub_pred["target"]==j)].lower.values[0]
                            upper.loc[i,j] = sub_pred[(sub_pred["Country Code"]==i[0])&(sub_pred["target"]==j)].upper.values[0]
                        except:
                            print(f"cannot find "+ i[0] + " for indicator " + j + " and year " + i[1])
                    else:
                        lower.loc[i,j] = np.nan
                        upper.loc[i,j] = np.nan
    for i in np.unique(sub_pred["Country Code"]).tolist():
        try:
            assert i in results.index.levels[0], f"cannot find "+ i + " for dataset " + dataset + " and year " + str(year)
        except:
            print(f"cannot find "+ i + " for dataset " + dataset + " and year " + str(year))
            missed = prediction[(prediction.year == year)&(prediction.dataset==dataset)&(prediction["Country Code"]==i)]
            p = pd.DataFrame(data = [missed.prediction.values], columns=missed.target.values, index=[(i,year)])
            l = pd.DataFrame(data = [missed.lower.values], columns=missed.target.values, index=[(i,year)])
            u = pd.DataFrame(data = [missed.upper.values], columns=missed.target.values, index=[(i,year)])
            results=pd.concat([results,p])
            lower=pd.concat([lower,l])
            upper=pd.concat([upper,u])
    return results,lower,upper

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

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
# Import data
wb_data,indicatorMeta, datasetMeta, indicatorData = data_importer()
indicatorData = indicatorData[indicatorData["Country Code"].isin(SIDS)]
indicatorData = indicatorData.set_index(["Indicator Code","Country Code"]).sort_index(axis=1).dropna(axis=0,how='all')#.interpolate('linear')

rename_names = dict()
for i in indicatorData.columns:
    rename_names[i] = int(i)
    
indicatorData.rename(columns=rename_names,inplace=True)

predictions,indicator_importance,performance,gs = query_and_train(model,supported_years)

indicator_importance['predictor'] = indicator_importance.names.apply(lambda x: ast.literal_eval(x)[0])

indicator_importance.sort_values(["year","target","values"],inplace=True, ascending=False)
importanceSummed = indicator_importance.groupby(['model', 'year', 'target','predictor']).sum()
importanceSorted = importanceSummed.reset_index().sort_values('values',ascending = False).groupby(['year', 'target']).head(10)
indicator_importance = importanceSorted.sort_values(["year","target","values"], ascending=False)

# Merge with original
large_dict = dict()
targets = np.unique(predictions.target.values)
predictions["dataset"] = predictions.target.apply(lambda x: indicatorMeta[indicatorMeta["Indicator Code"]==x].Dataset.values[0])
print(indicator_importance)
indicator_importance["dataset"] = indicator_importance.target.apply(lambda x: indicatorMeta[indicatorMeta["Indicator Code"]==x].Dataset.values[0])

datasets = np.unique(indicatorMeta[indicatorMeta["Indicator Code"].isin(targets)].Dataset.values)
indicator_importance.rename(columns={"target":"predicted indicator","predictor":"feature indicator","values":"feature importance"},inplace=True)


for d in datasets:
    large_dict[d]=dict()
    print(d)
    for y in np.unique(predictions[predictions.dataset == d].year.values):
        large_dict[d][y] = dict()
        results,lower,upper = replacement(d,y, ind_data = indicatorData, ind_meta=indicatorMeta, sids=SIDS, pred=predictions)
        large_dict[d][y]["prediction"] = results
        large_dict[d][y]["lower"] = lower
        large_dict[d][y]["upper"] = upper
        large_dict[d][y]["importance"] = indicator_importance[((indicator_importance.year == y)&(indicator_importance.dataset==d))][["predicted indicator","feature indicator","feature importance"]]

        
# Convert to API format
processMLData(large_dict)

#Update Metadata
#metadata.to_excel(mlMetadata)
with open(mlMetadata, "w") as write_file:
    json.dump(mlMetajson, write_file, indent=4)
# Push to git
COMMIT_MESSAGE = ' '.join(['add:',model_code,"from",start_year,'to',end_year, "(",response,")"])  


git_push(COMMIT_MESSAGE)