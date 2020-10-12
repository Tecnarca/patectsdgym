# patectsdgym
A refactor and extension of the PATECTGAN model with an evaluation suite based on OpenDP

To compile as a dynamic package:
python -m pip install -e sdk/

To compile testing suite:
python -m pip install -e evaluation

ToDo:
- Add DPSDGym
- Add saving of models
- Add unit tests
- Support Incremental Training for DPGAN and PATEGAN (by saving the Preprocessor)
- Compare / Test against other losses