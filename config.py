from os.path import join

DATA_PATH = join('data', 'motor_data14-2018.csv')

class Features:
    CATEGORICAL_OHE = ['INSR_TYPE', 'TYPE_VEHICLE', 'USAGE']
    CATEGORICAL_ORD = []
    REAL            = ['PREMIUM']
    ALL             = CATEGORICAL_OHE + CATEGORICAL_ORD + REAL

TARGET = ['INSURED_VALUE']
