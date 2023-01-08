# Above-Ground Carbon Density Estimation

### Setup
1. Using a non-cambridge account and signup for earth engine: https://signup.earthengine.google.com/
2. Create a Google Earth Engine enabled Google Cloud Project
3. Install the Earth Engine library `pip install earthengine-api`
4. Authenticate yourself. Either the code itself will have `ee.Authenticate()` which will open a browser window to authenticate yourself. Or you can use the command line to authenticate yourself, using the gcloud CLI. See https://developers.google.com/earth-engine/guides/python_install-conda#authenticate
5. Emukit appears to use concatenate with empty arrays which is no longer possible in current versions of numpy. To fix this, insert `outputs = list(filter(lambda x: x.size > 0, outputs))` in `emukit.core.loop.user_function.py:176` above `outputs = np.concatenate(outputs, axis=0)` 
        outputs = np.concatenate(outputs, axis=0)

### Running the code
Install the dependencies using `pip install -r requirements.txt`.

Run the default bayesian optimization with `python src/main.py`. This will run the function on default, small subset of data and save a visualization of the estimation vs the ground truth data. To use the local data, run the script from the home directory of the project. The data is stored in `data/`.

Tinker with the parameters in `main.py`, or you can modify the bayesian optimization procedure in `bayes_opt.py`.

Running the basic examples will look like:
**NDVI**
![NDVI](results/sample_NDVI_bayes_opt_vis.png)
**EVI**
![EVI](results/sample_EVI_bayes_opt_vis.png)

### Known Issues
Our current usage of GPy ends up throwing several warnings. When running NDVI, we get
```
 .../python3.8/site-packages/GPy/kern/src/rbf.py:52: RuntimeWarning:overflow encountered in square
 .../python3.8/site-packages/paramz/transformations.py:111: RuntimeWarning:overflow encountered in expm1
```
When running EVI, we get
```
 .../python3.8/site-packages/GPy/kern/src/stationary.py:168: RuntimeWarning:overflow encountered in divide
 .../python3.8/site-packages/GPy/kern/src/rbf.py:52: RuntimeWarning:overflow encountered in square
 .../python3.8/site-packages/GPy/kern/src/rbf.py:76: RuntimeWarning:invalid value encountered in multiply
 .../python3.8/site-packages/paramz/transformations.py:111: RuntimeWarning:overflow encountered in expm1
```