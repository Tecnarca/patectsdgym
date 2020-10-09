import subprocess
import os

import pytest
import string
import pandas as pd

from opendp.whitenoise.metadata import CollectionMetadata

try:
    from opendp.whitenoise.synthesizers.preprocessors.preprocessing import GeneralTransformer
    from opendp.whitenoise.synthesizers.pytorch.pytorch_synthesizer import PytorchDPSynthesizer
    from opendp.whitenoise.synthesizers.pytorch.nn import DPGAN, DPCTGAN, PATECTGAN

except:
    import logging
    test_logger = logging.getLogger(__name__)
    test_logger.warning("Requires torch and torchdp")


git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.csv"))

schema = CollectionMetadata.from_file(meta_path)
df = pd.read_csv(csv_path)

@pytest.mark.torch
class TestPytorchDPSynthesizer_DPGAN:
    def setup(self):
        self.dpgan = PytorchDPSynthesizer(DPGAN(), GeneralTransformer())

    def test_fit(self):
        self.dpgan.fit(df)
        assert self.dpgan.gan.generator
    
    def test_sample(self):
        self.dpgan.fit(df)
        sample_size = len(df)
        synth_data = self.dpgan.sample(sample_size)
        assert synth_data.shape == df.shape

class TestPytorchDPSynthesizer_DPCTGAN:
    def setup(self):
        self.dpctgan = PytorchDPSynthesizer(DPCTGAN(), None)

    def test_fit(self):
        self.dpctgan.fit(df, categorical_columns=['sex','educ','race','married'])
        assert self.dpctgan.gan.generator
    
    def test_sample(self):
        self.dpctgan.fit(df, categorical_columns=['sex','educ','race','married'])
        sample_size = len(df)
        synth_data = self.dpctgan.sample(sample_size)
        assert synth_data.shape == df.shape

class TestPytorchDPSynthesizer_PATECTGAN:
    def setup(self):
        self.patectgan = PytorchDPSynthesizer(PATECTGAN(), None)

    def test_fit(self):
        self.patectgan.fit(df, categorical_columns=['sex','educ','race','married'])
        assert self.patectgan.gan.generator
    
    def test_sample(self):
        self.patectgan.fit(df, categorical_columns=['sex','educ','race','married'])
        sample_size = len(df)
        synth_data = self.patectgan.sample(sample_size)
        assert synth_data.shape == df.shape