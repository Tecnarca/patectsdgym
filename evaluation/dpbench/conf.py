  
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from patectgan.pytorch_synthesizer import PytorchDPSynthesizer
from patectgan.preprocessing import GeneralTransformer
from patectgan.architectures import DPGAN, PATEGAN, DPCTGAN, PATECTGAN, CTGAN

# Keep seed consistent for reproducibility 
SEED = 42

# Turn on/off balancing imbalanced data with SMOTE
BALANCE = True

# Turn on/off the synthesizers you want to use in eval here
SYNTHESIZERS = [
    ('ctgan', PytorchDPSynthesizer), #to be implemented (by a wrapper)
    ('dpgan', PytorchDPSynthesizer),
    ('dpctgan', PytorchDPSynthesizer),
    ('patectgan', PytorchDPSynthesizer),
    ('wpatectgan', PytorchDPSynthesizer), # with no regularization, gradients explode or vanish
    ('patectdragan', PytorchDPSynthesizer),
    ('wpatectdragan', PytorchDPSynthesizer),
    ('pategan', PytorchDPSynthesizer),
]

# Define the defaults epsilons you want to use in eval
EPSILONS = [0.01, 0.05, 0.1, 0.5, 1, 3, 6, 10]

# Add datasets on which to evaluate synthesis
KNOWN_DATASETS =  ['bank','adult','mushroom','shopping','car']

# Default metrics used to evaluate differential privacy 
KNOWN_METRICS = ['wasserstein', 'ml_eval', 'pmse', 'sra']

# Add ML models on which to evaluate utility
KNOWN_MODELS = [AdaBoostClassifier, BaggingClassifier,
               LogisticRegression, MLPClassifier,
               RandomForestClassifier] 

# Mirror strings for ML models, to log
KNOWN_MODELS_STR = ['AdaBoostClassifier', 'BaggingClassifier',
               'LogisticRegression', 'MLPClassifier',
               'GaussianNB', 'RandomForestClassifier']

VERBOSE = True

SYNTH_SETTINGS = {
    'ctgan': {
        'default': {
            
            'gan': CTGAN(epochs=100)
        }
    },
    'dpgan': {
        'default': {
            'preprocessor': GeneralTransformer(),
            'gan': DPGAN(batch_size=1280)
        }
    },
    'dpctgan': {
        'default': {
            
            'gan': DPCTGAN(epochs=100)
        }
    },
    'patectgan': {
        'default': {
            
            'gan': PATECTGAN(epochs=100)
        },
    },
    'wpatectgan': {
        'default': {
            
            'gan': PATECTGAN(epochs=100, loss="wasserstein")
        },
    },
    'patectdragan': {
        'default': {
            
            'gan': PATECTGAN(epochs=100, regularization="dragan")
        },
    },
    'wpatectdragan': {
        'default': {
            
            'gan': PATECTGAN(epochs=100, loss="wasserstein", regularization="dragan")
        },
    },
    'pategan': {
        'default': {
            'preprocessor': GeneralTransformer(),
            'gan': PATEGAN(batch_size=1280)
        }
    }
}

MODEL_ARGS = {
    'AdaBoostClassifier': {
        'random_state': SEED,
        'n_estimators': 100
    },
    'BaggingClassifier': {
        'random_state': SEED
    },
    'LogisticRegression': {
        'random_state': SEED,
        'max_iter': 5000,
        'multi_class': 'auto',
        'solver': 'lbfgs'
    },
    'MLPClassifier': {
        'random_state': SEED,
        'max_iter': 10000,
        'early_stopping': True,
        'n_iter_no_change': 20
    },
    'DecisionTreeClassifier': {
        'random_state': SEED,
        'class_weight': 'balanced'
    },
    'RandomForestClassifier': {
        'random_state': SEED,
        'class_weight': 'balanced',
        'n_estimators': 200
    },
    'ExtraTreesClassifier': {
        'random_state': SEED,
        'class_weight': 'balanced',
        'n_estimators': 200
    }
}