import pandas as pd
import os
import json
import numpy as np
from git import Repo

mlMetadata = "/Volumes/My Passport for Mac/jobs/UNDP/ML Interface/automate test/Automate/mlResults/ML Model Metadata.json"
with open(mlMetadata) as json_file:
    mlMetajson = json.load(json_file)

DATASETS_PATH = "/Volumes/My Passport for Mac/jobs/UNDP/ML-IndicatorData/"

savepath = "/Volumes/My Passport for Mac/jobs/UNDP/ML Interface/automate test/Automate/data/ml/"
mlResults = "/Volumes/My Passport for Mac/jobs/UNDP/ML Interface/automate test/Automate/mlResults/"



PATH_OF_GIT_REPO ="/Volumes/My Passport for Mac/jobs/UNDP/ML Interface/automate test/Automate"# "../api" # make sure .git folder is properly configured
repo = Repo(PATH_OF_GIT_REPO)
repo.remotes.origin.pull()


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
start_year = get_inputs("year to start from? e.g. 2010")
end_year = get_inputs("year to end at? e.g. 2019")


output_type = get_inputs("Are model outputs in a country by indicator matrix format", ['y','n'])


def folderChecker():
    model_code = get_inputs("Input model code in format 'model1', 'model2',...")
    if model_code in os.listdir(savepath):
        response = get_inputs("model code already present in the API. Would you like to update without removing content (including model Metadata), replace existing folder (including model Metadata) or neither?", ['update','replace', 'neither'])
        if response == 'neither':
            model_code, response = folderChecker()
        return model_code, response
    else:
        response='new'
        return model_code, response

model_code,response = folderChecker()
if (response in  ['replace','new']):
    name = get_inputs("Model name for excel sheet")
    description = get_inputs("Model description for excel sheet")
    modellingApproach = "year-by-year"
    parameters = get_inputs("Some of the parameters used or searched for excel sheet (alternatively type unknown)")
    advantage = get_inputs("What are the advantages of this model (alternatively type unknown)")
    drawback = get_inputs("what are the drawbacks of this model (alternatively type unknown)")
    #layer = dict()
    #for i in metadata.columns:
    #    if i == "Model":
    #        layer[i] = "Model " + model_code[-1]
    #    elif i == "Model name":
    #        layer[i] = name
    #    elif i == "Model Description":
    #        layer[i] = description
    #    else:
    #        layer[i] = np.nan
    #metadata = metadata.append(layer, ignore_index = True)
    
    mlMetajson["Model " + model_code[-1]] = dict()
    mlMetajson["Model " + model_code[-1]]["Modelling Approach"] = modellingApproach
    mlMetajson["Model " + model_code[-1]]["Model Name"] = name
    mlMetajson["Model " + model_code[-1]]["Parameters"] = parameters
    mlMetajson["Model " + model_code[-1]]["Model Description"] = description
    mlMetajson["Model " + model_code[-1]]["Model Advantage"] = advantage
    mlMetajson["Model " + model_code[-1]]["Model Drawback"] = drawback
    


#folder="../mlResults"
allMeta=pd.read_csv(DATASETS_PATH+"indicatorMeta.csv")


def processMLData():
    print(model_code)
    #modelCodes = [model_code.split(" ")[1]]#[f.path.split("Model ")[1] for f in os.scandir(folder) if f.is_dir() ]
    modelCodes = model_code[-1]
    for modelCode in modelCodes:
        print(modelCode)
        fileNames=[ f.path for f in os.scandir(mlResults+"/model"+modelCode+"/predictions") if not f.is_dir() ]
        
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
            p = mlResults+"/model"+modelCode+"/predictions/"+datasetCode+"_predictions_"+years[0]+".csv"
            if os.path.exists(p):
                indicatorCodes=pd.read_csv(p).columns.drop(['Unnamed: 1','Country Code'],errors='ignore').tolist()
                for indicator in indicatorCodes[:100]:

                    print(indicator)
                    if (os.path.exists(savepath+'model'+str(modelCode)+'/'+datasetCode+'/'+indicator+'.json') & (response == 'update')):
                        with open(savepath+'model'+str(modelCode)+'/'+datasetCode+'/'+indicator+'.json') as json_file:
                                indicatorJson = json.load(json_file)

                    else: 
                        indicatorJson={"data":{},"upperIntervals":{},"lowerIntervals":{},"categoryImportances":{},"featureImportances":{}}
                    for year in years:
                        if os.path.exists(mlResults+"/model"+modelCode+"/predictions/"+datasetCode+"_predictions_"+year+".csv"):
                            predictionsDf=pd.read_csv(mlResults+"/model"+modelCode+"/predictions/"+datasetCode+"_predictions_"+year+".csv")
                            lowerIntervalsDf=pd.read_csv(mlResults+"/model"+modelCode+"/prediction intervals/lower/"+datasetCode+"_lower_"+year+".csv")
                            upperIntervalsDf=pd.read_csv(mlResults+"/model"+modelCode+"/prediction intervals/upper/"+datasetCode+"_upper_"+year+".csv")
                            featureImportancesDf=pd.read_csv(mlResults+"/model"+modelCode+"/feature importances/"+datasetCode+"_feature_importance_"+year+".csv")
                            
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
    #name = get_inputs("Model name for excel sheet")
    #description = get_inputs("Model description for excel sheet")
    #metadata = pd.read_excel(mlResults + "/ML Model Metadata.xlsx")
    #layer = dict()
    #for i in metadata.columns:
    #    if i == "Model":
    #        layer[i] = model_code
    #    elif i == "Model name":
    #        layer[i] = name
    #    elif i == "Model Description":
    #        layer[i] = description
    #    else:
    #        layer[i] = np.nan
    #metadata.append(layer, ignore_index = True)
    #metadata.to_excel(mlResults + "/ML Model Metadata.xlsx")
    if model_approach == "iterative":
        # need different script setup
        print("automation currently not implemented")
        #p = folder+"/Model "+modelCode+"/predictions/"+datasetCode+"_predictions"+".csv"
        #temp_p = folder+"/Model "+modelCode+"/predictions/"+datasetCode+"_test"+"_predictions"+".csv"
    else: 
        processMLData()
        with open(mlMetadata, "w") as write_file:
            json.dump(mlMetajson, write_file, indent=4)
        COMMIT_MESSAGE = ' '.join(['test:','add',model_code,"from",start_year,'to',end_year])  

        def git_push():
            try:
                repo = Repo(PATH_OF_GIT_REPO)
                repo.git.add(all=True)
                repo.index.commit(COMMIT_MESSAGE)
                origin = repo.remote(name='origin')
                origin.push()
            except:
                print('Some error occured while pushing the code')    

        git_push()

else: 
    print("convert output into a country by indicator format")



#PATH_OF_GIT_REPO = "../api" # make sure .git folder is properly configured



