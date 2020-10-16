# patectsdgym
A refactor and extension of the PATECTGAN model with an evaluation suite based on DPSDGym (by OpenDP)

This repo implements the PATECTGAN model with the Wasserstein Loss function and the DRAGAN regularization (WPATECTDRAGAN). The repo is based on the whitenoise system from OpenDP. 


All the package requirements needed for the notebooks, the benchmarker and the models are listed in the `requirements.txt` file.

To compile the models as a dynamic package:
```
python -m pip install -e sdk/
```

To compile testing suite and run a benchmark of the models:
```
python -m pip install -e evaluation/
```
Run a benchmark of all datasets and all models with:
```
python ./evaluation/dpbench/main.py
```
You can see the available tests that you can run from
```
python ./evaluation/dpbench/main.py --help
```
You can add new datasets from `./evaluation/dpbench/datasets.json` and add new models to the benchmarking from `/evaluation/dpbench/conf.py`.

Try experimenting with the models!

ToDo:
- Add unit tests
- Compare / Test PATECTGAN against other losses
- Parse data from MLFLOW to Jupyter notebook to run data analysis of results
- Add data visualization notebooks

All the code in this repo is released under the MIT License.