import pandas as pd
import numpy as np

import os
from git import Repo

from constants import DATASETS_PATH, SIDS, savepath,PATH_OF_GIT_REPO
from enums import Model, Interval, Interpolator, Schema


from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,cross_val_predict

from sklearn.svm import SVR, NuSVR
from sklearn.linear_model import SGDRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import Pool, CatBoostRegressor


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

def metaUpdater(mlMetajson, model_code):
    name = get_inputs("Model name for excel sheet")
    description = get_inputs("Model description for excel sheet")
    modellingApproach = "year-by-year"
    parameters = get_inputs("Some of the parameters used or searched for excel sheet (alternatively type unknown)")
    advantage = get_inputs("What are the advantages of this model (alternatively type unknown)")
    drawback = get_inputs("what are the drawbacks of this model (alternatively type unknown)")
    mlMetajson["Model " + model_code[-1]] = dict()
    mlMetajson["Model " + model_code[-1]]["Modelling Approach"] = modellingApproach
    mlMetajson["Model " + model_code[-1]]["Model Name"] = name
    mlMetajson["Model " + model_code[-1]]["Parameters"] = parameters
    mlMetajson["Model " + model_code[-1]]["Model Description"] = description
    mlMetajson["Model " + model_code[-1]]["Model Advantage"] = advantage
    mlMetajson["Model " + model_code[-1]]["Model Drawback"] = drawback

    return mlMetajson

def model_trainer(X_train, X_test, y_train, seed, n_estimators, model_type, interval,sample_weight = None):
    """
    Train the selected model, cross validate, score and generate a 90% prediction interval based on bootstrapped residuals.
    Args:
        X_train: training data
        X_test: prediction data
        y_train: training target array
        seed: random state setter
        n_estimators: number of trees for tree based models
        model: type of model to be trained
    Returns:
        prediction: dataframe with prediction values and confidence interval
        rmse: root mean squared error of the model
        gs: trained GridSearchCV model
        best_model: the best model
    """
    #X_train["sample_weight"]= X_train.reset_index()['Country Code'].apply(lambda x: 100 if x in SIDS else 1)
    if sample_weight is None:
        sample_weight = []#X_train.pop("sample_weight")
        sids_weights = (X_train.index.isin(SIDS)).sum()
        total = X_train.shape[0]

        # Inverse class weighting for SIDS and non-SIDS
        for i in X_train.index:
            if i in SIDS:
                sample_weight.append(1/sids_weights)
            else:
                sample_weight.append(1/(total-sids_weights))

    model_list = None
    if model_type == Model.all.name:
        model_list = [e.name for e in Model if e != Model.all]
    else:
        model_list = [model_type]

    print(model_list)
    model_instances = []
    params = []

    num_folds = min(5,X_train.shape[0])  # Hard coded
    scoring = 'neg_mean_squared_error'
    if Model.rfr.name in model_list:
        clf1 = RandomForestRegressor(random_state=seed)
        param1 = {}
        param1['regressor__n_estimators'] = [n_estimators]
        param1['regressor__max_depth'] = [5, 10, 20, 100, None]  # Hard coded
        param1['regressor'] = [clf1]
        model_instances.append(clf1)
        params.append(param1)
    if Model.etr.name in model_list:
        clf2 = ExtraTreesRegressor(random_state=seed)
        param2 = {}
        param2['regressor__n_estimators'] = [n_estimators]
        param2['regressor__max_depth'] = [5, 10, 20, 100, None]  # Hard coded
        param2['regressor'] = [clf2]
        model_instances.append(clf2)
        params.append(param2)

    if Model.gbr.name in model_list:
        clf3 = GradientBoostingRegressor(random_state=seed)
        param3 = {}
        if interval == Interval.quantile:
            param3['regressor__loss'] = ['quantile']
            param3['regressor__alpha'] = [0.5]  # hard coded
        param3['regressor__n_estimators'] = [n_estimators]
        param3['regressor__max_depth'] = [3, 5, 10, 20, None]  # Hard coded
        param3['regressor'] = [clf3]
        model_instances.append(clf3)
        params.append(param3)

    if Model.esvr.name in model_list:
        clf4 = SVR(kernel='linear')
        param4 = {}
        param4['regressor__degree'] = [2,3,4]
        param4['regressor__C'] = [1, 2, 3]  # Hard coded
        param4['regressor'] = [clf4]
        model_instances.append(clf4)
        params.append(param4)
    
    if Model.nusvr.name in model_list:
        clf5 = NuSVR(kernel='linear')
        param5 = {}
        param5['regressor__degree'] = [2,3,4]
        param5['regressor__C'] = [1, 2, 3]  # Hard coded
        param5['regressor'] = [clf5]
        model_instances.append(clf5)
        params.append(param5)
    
    if Model.sdg.name in model_list:
        clf6 = SGDRegressor()
        param6 = {}
        param6['regressor__penalty'] =['l2', 'l1', 'elasticnet']
        param6['regressor__alpha'] = [0.0001,0.001,0.01,0.1]
        param6['regressor'] = [clf6]
        model_instances.append(clf6)
        params.append(param6)

    if Model.xgbr.name in model_list:
        clf7 = XGBRegressor(random_state=seed,importance_type='weight')
        param7 = {}
        param7['regressor__n_estimators'] = [n_estimators]
        param7['regressor__max_depth'] = [5, 10, 20, 100, None]  # Hard coded

        param7['regressor'] = [clf7]
        model_instances.append(clf7)
        params.append(param7)
    if Model.lgbmr.name in model_list:
        clf8 = LGBMRegressor(random_state=seed)
        param8 = {}
        param8['regressor__n_estimators'] = [n_estimators]
        param8['regressor__max_depth'] = [5, 10, 20, 100, None]  # Hard coded

        param8['regressor'] = [clf8]
        model_instances.append(clf8)
        params.append(param8)
    if Model.cat.name in model_list:
        clf9 = CatBoostRegressor(random_state=seed)
        param9 = {}
        #param9['regressor__n_estimators'] = [n_estimators]
        param9['regressor__max_depth'] = [5, 10]  # Hard coded

        param9['regressor'] = [clf9]
        model_instances.append(clf9)
        params.append(param9)


    pipeline = Pipeline([('regressor', model_instances[0])])

    n_jobs = 1
    if os.getenv("SEARCH_JOBS") is not None:
        n_jobs = int(os.getenv("SEARCH_JOBS"))
    

    print("Perform grid search using %d jobs", n_jobs)

    kwargs = {pipeline.steps[-1][0] + '__sample_weight': sample_weight}
    gs = GridSearchCV(pipeline, params, cv=num_folds, n_jobs=n_jobs, scoring=scoring, refit=True).fit(X_train, y_train,**kwargs)
    rmse = np.sqrt(-gs.best_score_)

    best_model = gs.best_estimator_["regressor"]

    prediction = pd.DataFrame(gs.predict(X_test), columns=["prediction"], index=X_test.index)

    if interval == Interval.bootstrap.name:

        # Residual Bootsrapping  on validation data
        pred_train = cross_val_predict(best_model, X_train, y_train, cv=3)

        res = y_train - pred_train

        ### BOOTSTRAPPED INTERVALS ###

        alpha = 0.1  # (90% prediction interval) #Hard Coded

        bootstrap = np.asarray([np.random.choice(res, size=res.shape) for _ in range(100)])
        q_bootstrap = np.quantile(bootstrap, q=[alpha / 2, 1 - alpha / 2], axis=0)

        # prediction = pd.DataFrame(gs.predict(X_test), columns=["prediction"], index=X_test.index)
        prediction["upper"] = prediction["prediction"] + q_bootstrap[1].mean()
        prediction["lower"] = prediction["prediction"] + q_bootstrap[0].mean()

    else:
        if str(type(best_model)) == "<class 'sklearn.ensemble._gb.GradientBoostingRegressor'>":
            all_models = {}
            for alpha in [0.05, 0.95]:  # Hard Coded
                gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha,
                                                max_depth=gs.best_params_['regressor__max_depth'],
                                                n_estimators=gs.best_params_['regressor__n_estimators'])
                all_models["q %1.2f" % alpha] = gbr.fit(X_train, y_train)
                # For prediction

            prediction["lower"] = all_models["q 0.05"].predict(X_test)
            prediction["upper"] = all_models["q 0.95"].predict(X_test)
        else:
            pred_Q = pd.DataFrame()
            for pred in best_model.estimators_:
                temp = pd.Series(pred.predict(X_test))
                pred_Q = pd.concat([pred_Q, temp], axis=1)
            quantiles = [0.05, 0.95]  # Hard Coded

            for q in quantiles:
                s = pred_Q.quantile(q=q, axis=1)
                prediction[str(q)] = s.values
            prediction.rename(columns={"0.05": "lower", "0.95": "upper"}, inplace=True)  # Column names are hard coded

    # Predict for SIDS countries with missing values
    #prediction = prediction[prediction.index.isin(SIDS)]
    #prediction = prediction.reset_index().rename(columns={"index": "country"}).to_dict(orient='list')
    #################### Prediction dataframe and best_model instance are the final results of the ML################

    return prediction, rmse, gs, best_model


def git_push(COMMIT_MESSAGE):
    try:
        repo = Repo(PATH_OF_GIT_REPO)
        repo.git.add(all=True)
        repo.index.commit(COMMIT_MESSAGE)
        origin = repo.remote(name='refactor')
        origin.push()
    except:
        print('Some error occured while pushing the code')   