# Automate

### Folder Structure
- **data** - API folder strucutre where ML results are stored
- **mlResults** - intermediate ML results format. Folder exists if training scripts output different formats
- **Scripts** - Folder with automation and training scripts
	- year-by-year-automate.py script for training year-by-year two level imputation models, restructuring to API format and pushing to the required repo.
	- automation.py script for restructuring output in **mlResults** and saving in the **data** folder and pushing to the required repo.


### Next steps
- Refactor code(keep in mind that other modelling apporaches can be added)
- Setup automatic monitoring of script runs
