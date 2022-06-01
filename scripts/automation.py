import pandas as pd
import os
import json
import numpy as np
from git import Repo

def get_inputs(text: [str], expect: [str] = None, default=None):
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
model_code = get_inputs("Please enter model code in mlResults (format of Model 1, Model 2,...)")

output_type = get_inputs("Are model outputs in a country by indicator matrix format", ['y','n'])

if model_code.replace(" ", "").lower() in os.listdir('../api/data/ml'):
    response = get_inputs("model code already present in the API. Would you like to continue with this code and replace existing folder in API?", ['y','n'])
    if response == 'n':
        model_code = get_inputs("Please enter different model code (format of Model 1, Model 2,...)")

folder="../mlResults"
allMeta=pd.read_csv("/Volumes/My Passport for Mac/jobs/UNDP/ML-IndicatorData/indicatorMeta.csv")


def processMLData():
    print(model_code)
    modelCodes = [model_code.split(" ")[1]]#[f.path.split("Model ")[1] for f in os.scandir(folder) if f.is_dir() ]
    for modelCode in modelCodes:
        print(modelCode)
        fileNames=[ f.path for f in os.scandir(folder+"/Model "+modelCode+"/predictions") if not f.is_dir() ]
        
        datasetCodes=[]
        years=[]
        
        for fileName in fileNames:
            assert len(fileName.split("_")) > 1, fileName
            datasetCode=fileName.split("_")[0].split("/Model "+modelCode+"/predictions/")[1]
            year=fileName.split("_")[-1][:-4]
            if datasetCode not in datasetCodes:
                datasetCodes.append(datasetCode)
            if year not in years:
                years.append(year)
                
        for datasetCode in datasetCodes:
            print(datasetCode)
            p = folder+"/Model "+modelCode+"/predictions/"+datasetCode+"_predictions_"+years[0]+".csv"
            if os.path.exists(p):
                indicatorCodes=pd.read_csv(p).columns.drop(['Unnamed: 1','Country Code'],errors='ignore').tolist()
                for indicator in indicatorCodes[:100]:

                    print(indicator)
                    indicatorJson={"data":{},"upperIntervals":{},"lowerIntervals":{},"categoryImportances":{},"featureImportances":{}}
                    for year in years:
                        if os.path.exists("../mlResults/Model "+modelCode+"/predictions/"+datasetCode+"_predictions_"+year+".csv"):
                            predictionsDf=pd.read_csv("../mlResults/Model "+modelCode+"/predictions/"+datasetCode+"_predictions_"+year+".csv")
                            lowerIntervalsDf=pd.read_csv("../mlResults/Model "+modelCode+"/prediction intervals/lower/"+datasetCode+"_lower_"+year+".csv")
                            upperIntervalsDf=pd.read_csv("../mlResults/Model "+modelCode+"/prediction intervals/upper/"+datasetCode+"_upper_"+year+".csv")
                            featureImportancesDf=pd.read_csv("../mlResults/Model "+modelCode+"/feature importances/"+datasetCode+"_feature_importance_"+year+".csv")
                            
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
                        print(indicatorJson)

                        with open('../api/data/ml/model'+str(modelCode)+'/'+datasetCode+'/'+indicator+'.json', 'w') as outfile:
                            json.dump(indicatorJson, outfile,cls=NpEncoder)

if output_type == 'y':
    name = get_inputs("Model name for excel sheet")
    description = get_inputs("Model description for excel sheet")
    metadata = pd.read_excel(folder + "/ML Model Metadata.xlsx")
    layer = dict()
    for i in metadata.columns:
        if i == "Model":
            layer[i] = model_code
        elif i == "Model name":
            layer[i] = name
        elif i == "Model Description":
            layer[i] = description
        else:
            layer[i] = np.nan
    metadata.append(layer, ignore_index = True)
    metadata.to_excel(folder + "/ML Model Metadata.xlsx")
    if model_approach == "iterative":
        # need different script setup
        pass
        #p = folder+"/Model "+modelCode+"/predictions/"+datasetCode+"_predictions"+".csv"
        #temp_p = folder+"/Model "+modelCode+"/predictions/"+datasetCode+"_test"+"_predictions"+".csv"
    else: 
        processMLData()


else: 
    print("convert output into a country by indicator format")



PATH_OF_GIT_REPO = "../api" # make sure .git folder is properly configured
COMMIT_MESSAGE = 'add ML results for ' + model_code

def git_push():
    try:
        repo = Repo(PATH_OF_GIT_REPO)
        repo.git.add(update=True)
        repo.index.commit(COMMIT_MESSAGE)
        origin = repo.remote(name='origin')
        origin.push()
    except:
        print('Some error occured while pushing the code')    

git_push()

