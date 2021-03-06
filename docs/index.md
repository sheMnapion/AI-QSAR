# AI-QSAR
The project is a software for QSAR prediction and molecule design by deep learning. 

### Environment
Tested on python3.6(conda virtual environment)

### Requirements
See `requirements.txt` and run `pip install -r requirement.txt`.

You have to install a valid version of **rdkit**. For python3.6 in conda virtual environment, you can install with `conda install conda-forge::rdkit`. To create a conda virtual enviroment for python3.6, run `conda create --name py36 python=3.6` and use `source activate py36` and `source deactivate py36` to activate and deactivate the virtual environment.

### Run
Enter folder `QSAR-GUI` and run `python main.py`

### Software Description

<img src="./images/2020-01-09-16-42-25.png" width="500px"/>

The name of the Software is **PyMolPredictor**. The window of the Software has 3 parts: a menubar on the top, a toolbar below the menubar, and the main window under the memubar. The menubar consists of various kinds of operations. The toolbar is made up of shortcuts of operations in the menubar. The main window is the main part for loading data and model, making prediction and design, output and save model and so on. In the main window, there are 5 tabs: Data Processing, Model Training, Result Analysis, Activity Prediction, and Molecule Design.

#### Menubar
<img src="./images/2020-01-03-10-52-14.png" width="300px"/>

+ Open Project: Browse and select the project's folder. Data folders in all main window's tabs will also be set to the project folder.

+ Exit: Exit the Software.

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

<img src="./images/2020-01-03-16-17-44.png" width="800px"/>

This tab is used for preposessing and output .csv data for input of other tabs.

<img src="./images/2020-01-03-14-06-13.png" width="300px"/>

Browse and Select Data: 
+ Browse: Click to set the current data folder.
+ Select(or double click): Click to select the highlighted data file in the current data folder.
+ Header: If "Header" is checked, then the first row of the csv file is used as column names; otherwise, the column name is automatically generated.

<img src="./images/2020-01-03-14-08-00.png" width="300px"/>

Data Preprocessing:
+ Outlier Operation: Select a missing value operation. 
+ Feature Analysis: Select a dimension reduction method to reduce dimention of features in the data, and plot them in a 2-d plane.
+ Train : Test: The ratio of length of trainset to length of testset. Split the selected dataset into trainset and testset according to the given ratio.
+ Browse: Click to set the output folder of preprocessed and splitted trainset and testset.
+ Analyze: Click to analyze the original and transformed dataset, and show on the right panel.
+ Output: Click to output the transformed trainset and testset to the output folder.

<img src="./images/2020-01-03-15-35-42.png" width="300px"/>

Analysis Result:
+ Activity: The upper panel is used to plot the distribution of the selected column. The lower panel shows the first 100 rows of Original Data and Transformed Data.
    + Select: Select a feature column and plot its distribution.
    + Origin Data: Switch to the Original Data's table
    + Transformed Data: Switch to the Transformed Data's table
+ Feature: This panel shows the 2-D plot of features after dimension reduction by PCA or SVD.

#### Main Window: Model Training

<img src="./images/2020-01-03-16-19-04.png" width="800px"/>

This tab is used for training QSAR model.

<img src="./images/2020-01-03-15-45-13.png" width="300px"/>

Browse and Select Data: 
+ Browse: Click to set the current data folder.
+ Select(or double click): Click to select the highlighted data file in the current data folder.
+ Header: If "Header" is checked, then the first row of the csv file is used as column names; otherwise, the column name is automatically generated.

<img src="./images/2020-01-03-15-45-46.png" width="300px"/>

Browse and Select Model: 
+ Browse: Click to browse and add a model file to the model list.
+ Select(or double click): Click to select the highlighted model file in the current model list.

<img src="./images/2020-01-03-15-46-17.png" width="300px"/>

Training Parameters:
+ From Loaded Model: If it is checked, training will start from the loaded model. 
+ ModelType: Whether to train DNN model or RNN model.
+ TargetType: Whether to train regression model or classification model. Only support regression currently.
+ Select Target: Select the target feature of training.
+ Learning Rate: Step size of Gradient Descent in the training process.
+ earlyStop: If it is checked, training will stop when the loss doesn't go down in **earlyStopEpochs** epochs. 
+ earlyStopEpochs: Set the epoch limit of earlyStop.
+ Batch Size: Size of training batch of Stochastic Gradient Descent in the training process.
+ Epochs: Maximum epochs of training.
+ Train: Click to start training model with all training parameters above. When training is finished, you can press **Save** to save the model.
+ Save: Click to browse and save the trained model.

#### Main Window: Result Analysis

<img src="./images/2020-01-03-16-19-30.png" width="800px"/>

This tab is used for asnalysing trained QSAR model.

<img src="./images/2020-01-03-15-47-00.png" width="300px"/>

Training History: After each training process, the training history is saved to folder `QSAR-DNN/__trainingcache__` automatically. All training history files in the folder are displayed here.
+ Select(or double click): Click to select the highlighted training history.

<img src="./images/2020-01-03-15-47-49.png" width="300px"/>

Plot Settings:
+ Loss Curve: Whether to plot the Loss Curve.
+ Model Structure: Whether to plot the Model Structure. 
+ Precision Curve: Whether to plot the Precision Curve.
+ Analyze: Click to draw the analysis plots of the selected training history.

#### Main Window: Activity Prediction

<img src="./images/2020-01-03-16-19-57.png" width="800px"/>

This tab is used for predicting QSAR on test datasets, plotting predicted molecule structure, and plot fitting curve.

<img src="./images/2020-01-03-15-48-31.png" width="300px"/>

Browse and Select Data: 
+ Browse: Click to set the current data folder.
+ Select(or double click): Click to select the highlighted data file in the current data folder.
+ Header: If "Header" is checked, then the first row of the csv file is used as column names; otherwise, the column name is automatically generated.

<img src="./images/2020-01-03-15-50-05.png" width="300px"/>

Browse and Select Model: 
+ Select Target: Select the target feature of the test dataset.
+ Select Smiles: Select the SMILES feature of the the test dataset. This feature is used for plotting molecule structures.
+ Browse: Click to browse and add a model file to the model list.
+ Select(or double click): Click to select the highlighted model file in the current model list.
+ Analyze: Start calculating prediction informations, plotting molecules with top5 target value, and the fitting plot.

#### Main Window: Molecule Design

<img src="./images/2020-01-03-16-27-33.png" width="500px"/>

This tab is used for training and analyzing VAE model to design molecules.

<img src="./images/2020-01-03-15-52-04.png" width="300px"/>

Browse and Select Data: 
+ Browse: Click to set the current data folder.
+ Select(or double click): Click to select the highlighted data file in the current data folder.
+ Header: If "Header" is checked, then the first row of the csv file is used as column names; otherwise, the column name is automatically generated.

<img src="./images/2020-01-03-15-53-38.png" width="300px"/>

Browse and Select Model: 
+ Select Target: Select the target feature of the train dataset.
+ Select Smiles: Select the SMILES feature of the the train dataset. This feature is used for plotting molecule structures.
+ Train: Click to start training the VAE model with selected trainset. When training is finished, you can press **Save** to save the model.
+ Browse: Click to browse and add a model file to the model list.
+ Select(or double click): Click to select the highlighted model file in the current model list.
+ Save: Click to browse and save the trained model.
+ Design: Start designing and plot molecules with top target value according to the loaded VAE model. 