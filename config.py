from os.path import join

# Paths
DATA_PATH = join('data', 'motor_data14-2018.csv')

# Features and target values
class Features:
    CATEGORICAL_OHE = ['INSR_TYPE', 'TYPE_VEHICLE', 'USAGE']
    REAL            = ['PREMIUM']
    ALL             = CATEGORICAL_OHE + REAL

TARGET = ['INSURED_VALUE']

# Random generation
RANDOM_SEED = 42
