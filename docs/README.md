# AI-QSAR
The project is a software for QSAR prediction and SMILES design by deep learning. 

### Environment
Tested on python3.6(conda virtual environment)

### Requirements
You have to install a valid version of **rdkit**. For python3.6 in conda virtual environment, you can install with `conda install conda-forge::rdkit`. To create a conda virtual enviroment for python3.6, run `conda create --name py36 python=3.6` and use `source activate py36` and `source deactivate py36` to activate and deactivate the virtual environment.

For other packages, see `requirements.txt` and run `pip install -r requirement.txt`

### Run
Enter folder `QSAR-GUI` and run `python main.py`

### GUI Description
The name of the GUI is **PyMolPredictor**. The window of the GUI has 3 parts: a menubar on the top, a toolbar below the menubar, and the main window under the memubar. The menubar consists of various kinds of operations. The toolbar is made up of shortcuts of operations in the menubar. The main window is the main part for loading data and model, making prediction and design, output and save model and so on. In the main window, there are 4 tabs: Data Processing, Model Training, Result Analysis, Activity Prediction, and Model Design.

#### Menubar
<!-- ![](2020-01-03-10-52-14.png) -->
<img src="./images/2020-01-03-10-52-14.png" width="300px"/>

+ Open Project: Browse and select the project's folder. Data folders in all main window's tabs will also be set to the project folder.

+ Exit: Exit the GUI.

<img src="./images/2020-01-03-12-57-56.png" width="300px"/>

+ Open Data: Browse and select the data folder of the current tab.

+ Select Data: Select the currently highlighted data of the current tab.

<img src="./images/2020-01-03-13-07-09.png" width="300px"/>

+ Load Model: Browse and load a model file in the current tab.

+ Select model: Select the currently highlighted model in the model list of the current tab.

<img src="./images/2020-01-03-13-13-37.png" width="300px"/>

+ Select History: Select a training history file from the training history folder in "Result Analysis" tab.

+ Analyze: Trigger the "Analyze" or "Design" pushButton to make some analysis in the current tab.

+ Train: Trigger the "Train" pushButton to start training models in the current tab.

<img src="./images/2020-01-03-13-14-01.png" width="300px"/>

+ Data Processing: Switch to "Data Processing" tab.
+ Model Training: Switch to "Model Training" tab.
+ Result Analysis: Switch to "Result Analysis" tab.
+ Molecule Design: Switch to "Molecule Design" tab.

<img src="./images/2020-01-03-13-14-15.png" width="300px"/>

+ About: Not Implemented yet.

#### Main Window: Data Processing
This tab is used for preposessing and output .csv data for input of other tabs.

#### Main Window: Model Training
This tab is used for training QSAR model.

Browse and Select Data: 
+ Browse: Set the current data folder.
+ Select(or double click): Select the highlighted data file in the current data folder.

<img src="./images/2020-01-03-14-06-13.png" width="300px"/>

Data Preprocessing:
+ Outlier Operation: Select a missing value operation. 
+ Feature Analysis: Select a dimension reduction method to reduce dimention of features in the data, and plot them in a 2-d plane.
+ Train : Test: The ratio of length of trainset to length of testset. Split the selected dataset into trainset and testset according to the given ratio.
+ Browse: Set the output folder of preprocessed and splitted trainset and testset.
+ Analyze: Analyzing the original and transformed dataset, and show on the right panel.
+ Output: Output the transformed trainset and testset to the output folder.

<img src="./images/2020-01-03-14-08-00.png" width="300px"/>


#### Main Window: Result Analysis
This tab is used for analysing trained QSAR model.

#### Main Window: Activity Prediction
This tab is used for predicting QSAR on test datasets, plotting predicted molecule structure, and plot fitting curve.

#### Main Window: Model Design
This tab is used for training and analyzing VAE model to design molecules.