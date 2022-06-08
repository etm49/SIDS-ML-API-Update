import pandas as pd
import os
import json
import numpy as np
from git import Repo

from constants import mlMetadata, DATASETS_PATH, SIDS, savepath, mlResults, PATH_OF_GIT_REPO
from utils import data_importer, model_trainer, get_inputs, folderChecker, metaUpdater,git_push

with open(mlMetadata) as json_file:
    mlMetajson = json.load(json_file)




repo = Repo(PATH_OF_GIT_REPO)
repo.remotes.origin.pull()



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

model_approach = get_inputs("Please enter modelling approach", ["year-by-year","timeseries",'iterative'])
#model_raw_data = get_inputs()
start_year = get_inputs("year to start from? e.g. 2010")
end_year = get_inputs("year to end at? e.g. 2019")


output_type = get_inputs("Are model outputs in a country by indicator matrix format", ['y','n'])


model_code,response = folderChecker()
if (response in  ['replace','new']):
    mlMetajson = metaUpdater(mlMetajson, model_code)
    


#folder="../mlResults"
allMeta=pd.read_csv(DATASETS_PATH+"indicatorMeta.csv")


def processMLData():
    print(model_code)
    #modelCodes = [model_code.split(" ")[1]]#[f.path.split("Model ")[1] for f in os.scandir(folder) if f.is_dir() ]
    modelCodes = model_code[-1]
    for modelCode in modelCodes:
        print(modelCode)
        fileNames=[ f.path for f in os.scandir(mlResults+"model"+modelCode+"/predictions") if not f.is_dir() ]
        
        datasetCodes=[]
        years=[]
        
        for fileName in fileNames:
            assert len(fileName.split("_")) > 1, fileName
            datasetCode=fileName.split("_")[0].split("/model"+modelCode+"/predictions/")[1]
            year=fileName.split("_")[-1][:-4]
            if datasetCode not in datasetCodes:
                datasetCodes.append(datasetCode)
            if year not in years:
                years.append(year)
                
        for datasetCode in datasetCodes:
            print(datasetCode)
            p = mlResults+"model"+modelCode+"/predictions/"+datasetCode+"_predictions_"+years[0]+".csv"
            if os.path.exists(p):
                indicatorCodes=pd.read_csv(p).columns.drop(['Unnamed: 1','Country Code'],errors='ignore').tolist()
                for indicator in indicatorCodes[:100]:
                    if indicator =="year":
                        continue
                    print(indicator)
                    if (os.path.exists(savepath+'model'+str(modelCode)+'/'+datasetCode+'/'+indicator+'.json') & (response == 'update')):
                        with open(savepath+'model'+str(modelCode)+'/'+datasetCode+'/'+indicator+'.json') as json_file:
                                indicatorJson = json.load(json_file)

                    else: 
                        indicatorJson={"data":{},"upperIntervals":{},"lowerIntervals":{},"categoryImportances":{},"featureImportances":{}}
                    for year in years:
                        if os.path.exists(mlResults+"model"+modelCode+"/predictions/"+datasetCode+"_predictions_"+year+".csv"):
                            predictionsDf=pd.read_csv(mlResults+"model"+modelCode+"/predictions/"+datasetCode+"_predictions_"+year+".csv")
                            lowerIntervalsDf=pd.read_csv(mlResults+"model"+modelCode+"/prediction intervals/lower/"+datasetCode+"_lower_"+year+".csv")
                            upperIntervalsDf=pd.read_csv(mlResults+"model"+modelCode+"/prediction intervals/upper/"+datasetCode+"_upper_"+year+".csv")
                            featureImportancesDf=pd.read_csv(mlResults+"model"+modelCode+"/feature importances/"+datasetCode+"_feature_importance_"+year+".csv")
                            
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
                                    
                                featuresMeta=allMeta[allMeta["Indicator Code"].isin(features)]
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


if output_type == 'y':

    if model_approach == "iterative":
        # need different script setup
        print("automation currently not implemented")
        #p = folder+"/Model "+modelCode+"/predictions/"+datasetCode+"_predictions"+".csv"
        #temp_p = folder+"/Model "+modelCode+"/predictions/"+datasetCode+"_test"+"_predictions"+".csv"
    else: 
        processMLData()
        with open(mlMetadata, "w") as write_file:
            json.dump(mlMetajson, write_file, indent=4)
        COMMIT_MESSAGE = ' '.join(['add:',model_code,"from",start_year,'to',end_year, "(",response,")"])  
        print(COMMIT_MESSAGE)
        git_push(COMMIT_MESSAGE)

else: 
    print("convert output into a country by indicator format")





