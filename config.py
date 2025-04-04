from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    BASE_DATASET_NAME = 'Chicago_Hackathon_base_datasets'
    DATASET_DIR = str(os.getenv('DATASETS_DIR'))
    BASE_DATSET_DIR = os.path.join(DATASET_DIR, BASE_DATASET_NAME)
