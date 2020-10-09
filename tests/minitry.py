from patectgan.preprocessing import GeneralTransformer
from patectgan.pytorch_synthesizer import PytorchDPSynthesizer
from patectgan.architectures.patectgan import PATECTGAN

import subprocess
import os
import string
import pandas as pd

git_root_dir = "C:\\Users\\Tecnarca\\Desktop\\dev-env\\service\\datasets"

csv_path = os.path.join(git_root_dir, os.path.join("PUMS.csv"))

df = pd.read_csv(csv_path)

patectgan = PytorchDPSynthesizer(PATECTGAN(loss="wasserstein", regularization="dragan", verbose=True), None, epsilon=2)
patectgan.fit(df, categorical_columns=['sex','educ','race','married'])
synth_data = patectgan.sample(df.size)
s = synth_data.corr()
d = df.corr()
print(d.subtract(s))


#from ctgan import CTGANSynthesizer
#ctgan = CTGANSynthesizer()
#ctgan.fit(df, ['sex','educ','race','married'], epochs=2000)
#synth_data = ctgan.sample(df.size)
#s = synth_data.corr()
#d = df.corr()
#print(d.subtract(s))
