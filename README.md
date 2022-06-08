# Automate
To update  ML API first update the paths in the **constants.py** script
### Folder Structure
- **data** - API folder strucutre where ML results are stored
- **mlResults** - intermediate ML results format. Folder exists if training scripts temporarily save data in different formats
- **Scripts** - Folder with automation and training scripts
	- year-by-year-automate.py script for training year-by-year two level imputation models, restructuring to API format and pushing to the required repo.
	- automation.py script for restructuring output in **mlResults** and saving in the **data** folder and pushing to the required repo.
	- utils.py script with all common functions used in year-by-year.py, automation.py as well as other scripts to be added
	- enums.py script holding symbolic names (members) bound to unique, constant values shared across training and automation scripts
	- constants.py script holding fixed constants such as data storage path and API path on local machine


### Next steps
- Improve parameter search space for gradient boost regressor and XGBoost regressor to reduce lack of meaningful learning (i.e. all feature importances being zero)
- Start pushing to SIDS API repo 
- Merge scripts with SIDS ML Backend to allow simultaneous model improvements
- Setup automatic monitoring of script runs

