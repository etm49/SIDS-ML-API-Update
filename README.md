# Automate
To update  ML API first update the paths in the **constants.py** script
### Folder Structure
- **data** - API folder strucutre where ML results are stored
- **mlResults** - Folder where intermediate ML results format are stored. Folder exists if training scripts temporarily save data in different formats
- **Scripts** - Folder with automation and training scripts
	- year-by-year-automate.py script for training year-by-year two level imputation models, restructuring to API format and pushing to the required repo.
	- automation.py script for restructuring output in **mlResults** and saving in the **data** folder and pushing to the required repo.
	- utils.py script with all common functions used in year-by-year.py, automation.py as well as other scripts to be added
	- enums.py script holding symbolic names (members) bound to unique, constant values shared across training and automation scripts
	- constants.py script holding fixed constants such as data storage path and API path on local machine


### 3 modelling approach structure

<img src="./docs/images/modelling approaches.png?raw=true" height="500px">

The explanation for each section are as follows:

0.	Original Dataset

The development indicator dataset is a combination of 18 publicly available datasets. This includes data from the World Bank, WHO, UNICEF, UNDESA, IRENA, ITU, IGRAC, IHME, FAO, UNCTAD, UNDESA, Yale, CIESIN, OECD, UNWTO, ILO, and many additional sources. The dataset provides measurements for 2729 development indicators for 299 countries spanning 5 decades, from 1970 to 2021. Of the 2729 indicators, 590 were measures for all years and 315 were measured only for a single year. The dataset comes in a large 468827 by 54 matrix format where the columns represent the years measured and rows represent unique country-indicator pairs.

**Year-by-year approach (left most)**

In this approach only a single year of data is taken into account for imputation. After which a target column is selected and imputed according to the following methodology.

1.	Year-by-year pre-processing

In the pre-processing stage, the column of the target year is subset sliced from the original dataset. Then using pandas’ stack function the sliced data frame is reshaped into a country by indicator format.

2.	Data split

Here the new reshaped data frame is split into training and testing data based on the target column (indicator to be imputed). A column can be considered a target column only if it has at least 20% of values as non-missing. Additionally, all columns, except the target column, are ranked by the amount of missing values they contain in the training data frame. For a given column, if more than 40% of the values are missing (i.e., measure = 40) the column will not be considered as a possible predictor and are removed from both training and testing data frames.

3.	First level imputation for predictors

As stated previously, predictors can have up to 40% of their values missing. Hence, we fit a first level standard imputer for the predictors on the training data frame. The current implementation uses the K-NN imputer where Each sample’s missing values are imputed using the mean value from n_neighbors nearest neighbors found in the training data frame. Both the training and testing predictors are transformed by the fitted imputer. However, other implemented options include simpleimputer and iterativeimputer.

4.	Feature selection

In this stage, dimensionality reduction is performed on the imputed predictors of the training data frame, reducing their number to 10. Available options include Recursive feature selection, principal component analysis and manual selection of features based on prior knowledge. PCA dimensionality reduction is the least computational intensive method and hence, used for generating current results. The selected features are also used to subset the test data frame.

5.	Model training

This stage involves train the selected model, use cross-validation to explore the parameter space via gridsearch, measure generalizability via normalized root mean squared error and generate a 90% prediction interval using based on bootstrapped residuals or quantiles. The models available include, 'Random Forest Regressor', 'Gradient Boost Regressor','Extra tree Regressor', "Epsilon-Support Vector Regressor",  "Nu Support Vector Regressor", "Linear Support Vector Regressor", "SGD Regressor", "XGBoost Regressor" and "LGBM Regressor".

In the year-by-year approach, due to the small size of data, both SIDS and non-SIDS are used to train the selected model. In order the accommodate for the small sample size of SIDS, SIDS observation are given higher weight during training using Inverse class weighting.

**Timeseries approach (center)**

the timeseries approach, takes a more longitudinal look at the data by focusing on the relationship between dependent and independent variables not just for the given year but for a handful of the preceding year. It behaves similar to the year-by-year approach in that it imputes missing values for one year but considers relation between indicators in previous years.

6.	Validity check

The first step in this approach is to check which indicators can be a valid target for a model. The validity check function is used to  select indicators observed for at least one SIDS country and for at least 5 out of 50 possible years as the pool of possible target indicators. The corresponding predictors need to have been observed for at least 10 SIDS countries for at least 10 years. Note that while this thresholds significantly reduce the possible predictors, they can be modified at any time to expand viable indicators. The original data frame is sliced according to the target indicator and valid predictor indicators.

7.	First level linear interpolation of predictors

In this stage, gaps in the predictors across time are filled using pandas interpolate function via linear method. The linear method was selected because exploratory work showed that most indicators tend to show linear trends for small slices of time.

8.	Time-series preprocessing

Once the predictors are imputed, the sliced data frame is reshaped into an multi-index data frame where each row represents a country and target history (also called window) pair. Each predictor column, on the other hand, represents the historical information (also called lag) of the indicator. <INSERT PICTURE>

In addition, sample weight is generated such that target history (windows) further away from the target year are given small weights to force the model to focus on the relationship between predictors and indicators close in time to the target year.

9.	Data split

Once set to this new format, the data frame is split into training and testing based on missing values in the target indicators. Test data frame includes observations that have missing values for the target indicator in the target year. All observations that are non-missing for the target are added to the training data frame. 


**Iterative approach (right most)**

Unlike the previous two methodologies, in this approach the entire original dataset is imputed using sklearn’s iterative imputer

10.	Preprocessing

Using pandas’ stack functionality, the original data frame is reshaped into a multi-index data frame where the rows represent country-year pairs and columns represent indicators.

11.	Model training

The iterative imputer is an experimental estimator which models each feature with missing values as a function of other features, and uses that estimate for imputation. It does so in an iterated round-robin fashion: at each step, a feature column is designated as output y and the other feature columns are treated as inputs X. A regressor is fit on (X, y) for known y. Then, the regressor is used to predict the missing values of y. This is done for each feature in an iterative fashion, and then is repeated for “max_iter” imputation rounds. The results of the final imputation round are returned.

The base estimator can be any sklearn regressor but currently there is no framework for hyperparameter tuning.


### Running Scripts
When one of the modelling scripts (time-series-automate.py, year-by-year-automate.py or iterative-automate.py) is run:
- Any updates from the API remote repo is pulled automatically
- The user is asked to select the model alogrithm according to their code in enums.py. The [Metadata](/data/api/data/ml/ML Model Metadata.json) information for the models currently in the API repo should be consulted before hand
- Then the user inputs the range of years to train and predict on the model
- The user then inputs the mode folder code according to the api (model1, model2, etc.)
	- if the model folder code already exists in the API, the user is asked if they want to *"update"* the folder contents without replacing the associated metadata or *"replace"* the contents or the metadata together.
	- if user is training the model on previously untrained year/indicators, "update" should be selected.
	- if user wants to replace folder contents with new model output, "replace" should be selected. If replace is selected:
		- user will be asked to provide new metadata associated with this model code. Information to be provided include
			- model name
			- model parameters tuned
			- model description
			- model advantages
			- model drawback
- The model will then train and save raw model data in **mlResults/modelcode/raw data from model** folder with the associated model folder code
- After the training and prediction is done for all selected years, the script will start to combine predictions with observed values and generate intermediate data including prediction/values, prediction intervals and feature importance tables in **mlResults/modelcode/**. 
- Then the script will further process these tables into json files that match the API format and save them in the location provided by the user (this is savepath variable in the constants.py file). In the current setup, the [SIDS organization API repo](https://github.com/SIDS-Dashboard/api) is cloned in the **data** folder
- The script's run will conclude after committing and pushing new json files (and metadata if needed) into the remote API repo.


### Next steps

- Merge scripts with SIDS ML Backend to allow simultaneous model improvements
- Setup automatic monitoring of script runs
