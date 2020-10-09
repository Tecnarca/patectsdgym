import numpy as np
import pandas as pd

from patectgan.preprocessing import GeneralTransformer

class SDGYMBaseSynthesizer():
    """
    Base for Whitenoise Synthesizers, based off of SDGymBaseSynthesizer
    (to allow for benchmarking)
    """

    def fit(self, data, categorical_columns=None, ordinal_columns=None):
        """
        Fits some data to synthetic data model.
        """
        pass

    def sample(self, samples, categorical_columns=None, ordinal_columns=None):
        """
        Produces n (samples) using the fitted synthetic data model.
        """
        pass

    def fit_sample(self, data, categorical_columns=None, ordinal_columns=None):
        """
        Common use case. Fits a synthetic data model to data, and returns
        # of samples equal to size of original dataset.
        Note data must be numpy array.
        """
        self.fit(data, categorical_columns, ordinal_columns)
        return self.sample(data.shape[0])


class PytorchDPSynthesizer(SDGYMBaseSynthesizer):
    def __init__(self, gan, preprocessor=None, epsilon=None):
        self.preprocessor = preprocessor
        self.gan = gan
        
        self.epsilon = epsilon

        self.categorical_columns = None
        self.ordinal_columns = None
        self.dtypes = None

        self.data_columns = None
    
    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        if isinstance(data, pd.DataFrame):
            self.data_columns = data.columns

        self.categorical_columns = categorical_columns
        self.ordinal_columns = ordinal_columns
        self.dtypes = data.dtypes

        if not self.epsilon:
            self.epsilon = 1.0

        if self.preprocessor:
            self.preprocessor.fit(data, categorical_columns, ordinal_columns)
            preprocessed_data = self.preprocessor.transform(data)
            self.gan.train(preprocessed_data, categorical_columns=categorical_columns, ordinal_columns=ordinal_columns, update_epsilon=self.epsilon)
        else:
            self.gan.train(data, categorical_columns=categorical_columns, ordinal_columns=ordinal_columns, update_epsilon=self.epsilon)
    
    def sample(self, n):
        synth_data = self.gan.generate(n)
        
        if self.preprocessor is not None:
            if isinstance(self.preprocessor, GeneralTransformer):
                synth_data = self.preprocessor.inverse_transform(synth_data, None)
            else:
                synth_data = self.preprocessor.inverse_transform(synth_data)

        if isinstance(synth_data, np.ndarray):
            synth_data = pd.DataFrame(synth_data,columns=self.data_columns)
        elif isinstance(synth_data, pd.DataFrame):
            synth_data.columns = self.data_columns
        else:
            raise ValueError("Generated data is neither numpy array nor dataframe!")

        return synth_data