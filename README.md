## Logistic Regression

For our case we train our XGBoost model to consume every metric collected from profiling and predict the inference time of the provided model.


### Code

The code is based on Python 3.13.2. Make sure to us conda and install the package dependencies from the requirements.txt
```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate logistics-regression
pip install -r requirements.txt
```
to train the model run the following:
```bash
python3 main.py
```