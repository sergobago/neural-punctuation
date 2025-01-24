from os.path import abspath, dirname, join

BASE_DIR = dirname(dirname(abspath(__file__)))
ENCODING = 'utf-8'
MEGABYTE = 1024 * 1024
DATASET_DIR = join(BASE_DIR, 'datasets')
NMODELS_DIR = join(BASE_DIR, 'nmodels')
TEMP_DIR = join(BASE_DIR, 'temps')
