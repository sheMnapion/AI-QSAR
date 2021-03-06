# AI-QSAR
This is a software for QSAR prediction and molecule design by deep learning. 

### Environment
Tested on python3.6(conda virtual environment)

### Requirements
See `requirements.txt` and run `pip install -r requirement.txt`.

You have to install a valid version of **rdkit**. For python3.6 in conda virtual environment, you can install with `conda install conda-forge::rdkit`. To create a conda virtual enviroment for python3.6, run `conda create --name py36 python=3.6` and use `source activate py36` and `source deactivate py36` to activate and deactivate the virtual environment.


### Run
Enter folder `QSAR-GUI` and run `python main.py`

### Software Description
<img src="./docs/images/2020-01-09-16-42-25.png" width="500px"/>

The name of the Software is **PyMolPredictor**. The window of the Software has 3 parts: a menubar on the top, a toolbar below the menubar, and the main window under the memubar. The menubar consists of various kinds of operations. The toolbar is made up of shortcuts of operations in the menubar. The main window is the main part for loading data and model, training models, making predictions, designing molecules, saving and outputing models and so on. In the main window, there are 4 tabs: Data Processing, Model Training, Result Analysis, Activity Prediction, and Molecule Design.

See docs for more information.