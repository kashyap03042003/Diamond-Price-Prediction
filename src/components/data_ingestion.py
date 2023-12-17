import os
import sys
import pickle
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Initialize the Data Ingestion configuration
@dataclass
class DataIngestionconfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    config_file_path: str = os.path.join('artifacts', 'data_ingestion_config.pkl')

# Create the data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:
            df = pd.read_csv(os.path.join('notebooks/data', 'gemstone.csv'))
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info('Raw data is created')

            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of Data is completed')

            # Save DataIngestionconfig as a pickle file
            with open(self.ingestion_config.config_file_path, 'wb') as config_file:
                pickle.dump(self.ingestion_config, config_file)

            logging.info('Data Ingestion configuration saved as a pickle file')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.config_file_path
            )

        except Exception as e:
            logging.info('Exception occurred at Data Ingestion Stage')
            raise CustomException(e, sys)

# Example usage:
data_ingestion_instance = DataIngestion()
train_data_path, test_data_path, config_file_path = data_ingestion_instance.initiate_data_ingestion()
print(f'Train data path: {train_data_path}')
print(f'Test data path: {test_data_path}')
print(f'Config file path: {config_file_path}')
