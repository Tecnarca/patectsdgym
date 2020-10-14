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
- Compare / Test against other losses
- Add sra
- Add Jupyter notebook interfaces
- Log correct values in MLFlow
- Save models for different (selected) epsilons
