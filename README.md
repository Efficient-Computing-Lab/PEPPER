## Logistic Regression

For our case to train our logistic regression model we have to provide every metric collected from profiling to the model in the x parameter. 
In the y parameters we provide a value that characterizes a slow inference time.
We expect from the model to find which models are slow and which are fast.
In addition, it should provide an estimation output on which features have a major role on the inference time.

### Code

The code is based on Python 3.13.2. Make sure to us conda and install the package dependencies from the requirements.txt
```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate logistric_regression
pip install -r requirements.txt
```
to train the model run the following:
```bash
python3 main.py
```