from patectgan.pytorch_synthesizer import PytorchDPSynthesizer
from patectgan.architectures import DPGAN, PATEGAN, DPCTGAN, PATECTGAN 
from patectgan.preprocessing import GeneralTransformer

import subprocess
import os
import string
import pandas as pd

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

csv_path = os.path.join(git_root_dir, os.path.join("datasets/PUMS.csv"))

df = pd.read_csv(csv_path)

print("Starting PATEGAN...")
pategan = PytorchDPSynthesizer(PATEGAN(), GeneralTransformer(), epsilon=1)
pategan.fit(df, categorical_columns=['sex','educ','race','married'])
synth_data = pategan.sample(df.size)
s = synth_data.corr()
d = df.corr()

print("Starting DPGAN...")
dpgan = PytorchDPSynthesizer(DPGAN(), GeneralTransformer(), epsilon=1)
dpgan.fit(df, categorical_columns=['sex','educ','race','married'])
synth_data = dpgan.sample(df.size)
s = synth_data.corr()
d = df.corr()

print("Starting PATECTGAN...")
patectgan = PytorchDPSynthesizer(PATECTGAN(loss="wasserstein", regularization="dragan", verbose=False), None, epsilon=1)
patectgan.fit(df, categorical_columns=['sex','educ','race','married'])
synth_data = patectgan.sample(df.size)
s = synth_data.corr()
d = df.corr()
#print(d.subtract(s))

print("Starting CTGAN...")
from ctgan import CTGANSynthesizer
ctgan = CTGANSynthesizer()
ctgan.fit(df, ['sex','educ','race','married'], epochs=10)
synth_data = ctgan.sample(df.size)
s = synth_data.corr()
d = df.corr()
#print(d.subtract(s))

print("Starting DPCTGAN...")
dpctgan = PytorchDPSynthesizer(DPCTGAN(verbose=False), epsilon=1)
dpctgan.fit(df, categorical_columns=['sex','educ','race','married'])
synth_data = dpctgan.sample(df.size)
s = synth_data.corr()
d = df.corr()


