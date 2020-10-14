# patectsdgym
A refactor and extension of the PATECTGAN model with an evaluation suite based on DPSDGym (by OpenDP)




To compile as a dynamic package:
python -m pip install -e sdk/

To compile testing suite:
python -m pip install -e evaluation/

Run a benchmark of all datasets and all models with:
```
cd ./evaluation/dpbench/
python main.py
```
ToDo:
- Add unit tests
- Compare / Test PATECTGAN against other losses
- Fix "need 3" attributes for DPCTGAN and DPGAN
- Add Jupyter notebook interfaces
- Log correct values in MLFlow
- Save models for the various epsilons
