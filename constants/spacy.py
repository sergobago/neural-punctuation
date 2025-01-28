from os.path import join

from constants.project_params import NMODELS_DIR

SPACY_DIR_NAME = 'spacy'
SPACY_CPU_CORES = 1
MAX_SPACY_CHUNK_SIZE = int(1e7)
SPACY_DIR = join(NMODELS_DIR, SPACY_DIR_NAME)

SPACY_ENTITY_IOBS = ['I', 'O', 'B']
