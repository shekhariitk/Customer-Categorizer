import sys
from datetime import datetime
import numpy as np
import os
import pandas as pd
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.components.data_ingestion import DataIngestion
from src.components.data_clustering import CreateClusters
from src.constant.training_pipeline import TARGET_COLUMN
from src.entity.config_entity import SimpleImputerConfig
from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils

class DataTransformation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 data_tranasformation_config: DataTransformationConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_tranasformation_config
        self.data_ingestion = DataIngestion()
        self.imputer_config = SimpleImputerConfig()
        self.utils = MainUtils()
        logging.info("DataTransformation initialized successfully")

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from: {file_path}")
            df = pd.read_csv(file_path)
            logging.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error reading data from {file_path}: {str(e)}")
            raise CustomerException(e, sys)

    def get_new_features(self, train_set: DataFrame, test_set: DataFrame) -> tuple:
        """Create new features for both train and test sets"""
        try:
            logging.info("Starting feature engineering process")
            logging.info(f"Input - Train shape: {train_set.shape}, Test shape: {test_set.shape}")
            
            train_set_with_new_features = pd.DataFrame()
            test_set_with_new_features = pd.DataFrame()
            datasets = {"train_set": train_set, "test_set": test_set}

            for key in datasets:
                logging.info(f"Processing {key}")
                dataset = datasets[key].copy()
                original_shape = dataset.shape
                
                # Log initial data quality
                logging.info(f"{key} - Original shape: {original_shape}")
                logging.info(f"{key} - Missing values: {dataset.isnull().sum().sum()}")
                
                # Creating Age feature
                dataset['Age'] = 2022 - dataset['Year_Birth']
                logging.info(f"{key} - Age feature created. Range: {dataset['Age'].min()} to {dataset['Age'].max()}")

                # Recoding education level
                education_mapping = {"Basic": 0, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 4}
                dataset["Education"] = dataset["Education"].replace(education_mapping).astype('int64')
                logging.info(f"{key} - Education recoded: {dataset['Education'].value_counts().to_dict()}")

                # Recoding marital status
                marital_mapping = {"Married": 1, "Together": 1, "Absurd": 0, "Widow": 0, 
                                 "YOLO": 0, "Divorced": 0, "Single": 0, "Alone": 0}
                dataset['Marital_Status'] = dataset['Marital_Status'].replace(marital_mapping).astype('int64')
                logging.info(f"{key} - Marital status recoded: {dataset['Marital_Status'].value_counts().to_dict()}")

                # Creating children count
                dataset['Children'] = dataset['Kidhome'] + dataset['Teenhome']
                logging.info(f"{key} - Children feature created. Range: {dataset['Children'].min()} to {dataset['Children'].max()}")

                # Creating Family_Size
                dataset['Family_Size'] = dataset['Marital_Status'] + dataset['Children'] + 1
                logging.info(f"{key} - Family size created. Range: {dataset['Family_Size'].min()} to {dataset['Family_Size'].max()}")

                # Creating total spending
                spending_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
                dataset['Total_Spending'] = dataset[spending_cols].sum(axis=1)
                logging.info(f"{key} - Total spending created. Range: {dataset['Total_Spending'].min()} to {dataset['Total_Spending'].max()}")

                # Total promotions
                promo_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
                dataset["Total Promo"] = dataset[promo_cols].sum(axis=1)
                logging.info(f"{key} - Total promo created. Range: {dataset['Total Promo'].min()} to {dataset['Total Promo'].max()}")

                # Date processing
                dataset['Dt_Customer'] = pd.to_datetime(dataset['Dt_Customer'], format='%d-%m-%Y')
                today = datetime.today()
                dataset['Days_as_Customer'] = (today - dataset['Dt_Customer']).dt.days
                logging.info(f"{key} - Days as customer created. Range: {dataset['Days_as_Customer'].min()} to {dataset['Days_as_Customer'].max()}")

                # Offers responded to
                response_cols = promo_cols + ['Response']
                dataset['Offers_Responded_To'] = dataset[response_cols].sum(axis=1)
                logging.info(f"{key} - Offers responded created. Range: {dataset['Offers_Responded_To'].min()} to {dataset['Offers_Responded_To'].max()}")

                # Parental status
                dataset["Parental Status"] = np.where(dataset["Children"] > 0, 1, 0)
                logging.info(f"{key} - Parental status created: {dataset['Parental Status'].value_counts().to_dict()}")

                # Drop columns used to create new features
                columns_to_drop = ['Year_Birth', 'Kidhome', 'Teenhome', 'Dt_Customer']
                dataset.drop(columns=columns_to_drop, axis=1, inplace=True)
                logging.info(f"{key} - Dropped columns: {columns_to_drop}")

                # Rename columns
                rename_mapping = {
                    "Marital_Status": "Marital Status", "MntWines": "Wines", "MntFruits": "Fruits",
                    "MntMeatProducts": "Meat", "MntFishProducts": "Fish", "MntSweetProducts": "Sweets",
                    "MntGoldProds": "Gold", "NumWebPurchases": "Web", "NumCatalogPurchases": "Catalog",
                    "NumStorePurchases": "Store", "NumDealsPurchases": "Discount Purchases"
                }
                dataset.rename(columns=rename_mapping, inplace=True)
                logging.info(f"{key} - Columns renamed")

                # Select final columns
                final_columns = [
                    "Age", "Education", "Marital Status", "Parental Status",
                    "Children", "Income", "Total_Spending", "Days_as_Customer",
                    "Recency", "Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold",
                    "Web", "Catalog", "Store", "Discount Purchases", "Total Promo",
                    "NumWebVisitsMonth"
                ]
                
                # Check if all columns exist
                missing_cols = [col for col in final_columns if col not in dataset.columns]
                if missing_cols:
                    logging.error(f"{key} - Missing columns: {missing_cols}")
                    logging.info(f"{key} - Available columns: {list(dataset.columns)}")
                
                dataset = dataset[final_columns]
                
                # Final data quality check
                logging.info(f"{key} - Final shape: {dataset.shape}")
                logging.info(f"{key} - Final missing values: {dataset.isnull().sum().sum()}")
                logging.info(f"{key} - Data types: {dataset.dtypes.value_counts().to_dict()}")

                if key == 'train_set':
                    train_set_with_new_features = dataset
                else:
                    test_set_with_new_features = dataset

            logging.info("Feature engineering completed successfully")
            return train_set_with_new_features, test_set_with_new_features

        except Exception as e:
            logging.error(f"Error in feature engineering: {str(e)}")
            raise CustomerException(e, sys)

    def transform_data(self, train_set: DataFrame, test_set: DataFrame) -> tuple:
        """Apply preprocessing transformations"""
        try:
            logging.info("Starting data transformation process")
            logging.info(f"Input shapes - Train: {train_set.shape}, Test: {test_set.shape}")
            
            # Define feature groups
            numeric_features = [feature for feature in train_set.columns if train_set[feature].dtype != 'O']
            outlier_features = ["Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold", "Age", "Total_Spending"]
            numeric_features = [x for x in numeric_features if x not in outlier_features]
            
            logging.info(f"Numeric features ({len(numeric_features)}): {numeric_features}")
            logging.info(f"Outlier features ({len(outlier_features)}): {outlier_features}")

            # Create pipelines
            numeric_pipeline = Pipeline(steps=[
                ("Imputer", SimpleImputer(**self.imputer_config.__dict__)),
                ("StandardScaler", StandardScaler())
            ])

            outlier_features_pipeline = Pipeline(steps=[
                ("Imputer", SimpleImputer(**self.imputer_config.__dict__)),
                ("transformer", PowerTransformer(standardize=True))
            ])

            # Combine pipelines
            preprocessor = ColumnTransformer([
                ("numeric pipeline", numeric_pipeline, numeric_features),
                ("Outliers Features Pipeline", outlier_features_pipeline, outlier_features)
            ])

            logging.info("Fitting preprocessor on training data")
            # Fit and transform
            preprocessed_train_set = preprocessor.fit_transform(train_set)
            logging.info(f"Training data transformed. Shape: {preprocessed_train_set.shape}")
            
            preprocessed_test_set = preprocessor.transform(test_set)
            logging.info(f"Test data transformed. Shape: {preprocessed_test_set.shape}")

            # Convert back to DataFrame
            columns = train_set.columns
            preprocessed_train_set = pd.DataFrame(preprocessed_train_set, columns=columns)
            preprocessed_test_set = pd.DataFrame(preprocessed_test_set, columns=columns)
            
            logging.info("Data converted back to DataFrames")
            
            # Check for any issues in transformed data
            train_nan_count = preprocessed_train_set.isnull().sum().sum()
            test_nan_count = preprocessed_test_set.isnull().sum().sum()
            train_inf_count = np.isinf(preprocessed_train_set.values).sum()
            test_inf_count = np.isinf(preprocessed_test_set.values).sum()
            
            logging.info(f"Post-transformation data quality:")
            logging.info(f"Train - NaN: {train_nan_count}, Inf: {train_inf_count}")
            logging.info(f"Test - NaN: {test_nan_count}, Inf: {test_inf_count}")
            
            if train_nan_count > 0 or test_nan_count > 0:
                logging.warning("NaN values detected after transformation!")
            if train_inf_count > 0 or test_inf_count > 0:
                logging.warning("Infinite values detected after transformation!")

            # Save preprocessor
            preprocessor_obj_dir = os.path.dirname(self.data_transformation_config.transformed_object_file_path)
            os.makedirs(preprocessor_obj_dir, exist_ok=True)
            self.utils.save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            logging.info(f"Preprocessor saved to: {preprocessor_obj_dir}")

            return preprocessed_train_set, preprocessed_test_set

        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise CustomerException(e, sys) from e

    def initiate_data_transformation(self):
        """Initiate the complete data transformation process"""
        try:
            logging.info("=== STARTING DATA TRANSFORMATION PIPELINE ===")
            
            if not self.data_validation_artifact.validation_status:
                raise Exception("Data Validation Failed.")
            
            logging.info("Data validation passed. Starting transformation...")

            # Read data
            logging.info("=== STEP 1: READING DATA ===")
            train_set = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_set = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            # Create new features
            logging.info("=== STEP 2: FEATURE ENGINEERING ===")
            train_set, test_set = self.get_new_features(train_set, test_set)

            # Apply transformations
            logging.info("=== STEP 3: DATA PREPROCESSING ===")
            preprocessed_train_set, preprocessed_test_set = self.transform_data(train_set, test_set)

            # Apply clustering
            logging.info("=== STEP 4: CLUSTERING ===")
            cluster_creator = CreateClusters()
            
            # Train clustering
            logging.info("--- Training Clustering ---")
            labelled_train_set = cluster_creator.initialize_train_clustering(preprocessed_data=preprocessed_train_set)
            train_cluster_labels = labelled_train_set[TARGET_COLUMN].values
            logging.info(f"Train clustering completed. Labels shape: {train_cluster_labels.shape}")
            
            # Test clustering - FIXED: Don't pass train labels for alignment
            logging.info("--- Test Clustering ---")
            logging.info("NOT using train labels for test clustering alignment to avoid size mismatch")
            labelled_test_set = cluster_creator.initialize_test_clustering(
                preprocessed_data=preprocessed_test_set,
                y_true_train_labels=None  # FIXED: Removed problematic alignment
            )
            test_cluster_labels = labelled_test_set[TARGET_COLUMN].values
            logging.info(f"Test clustering completed. Labels shape: {test_cluster_labels.shape}")

            # Final data preparation
            logging.info("=== STEP 5: FINAL DATA PREPARATION ===")
            X_train = labelled_train_set.drop(columns=[TARGET_COLUMN], axis=1)
            y_train = labelled_train_set[TARGET_COLUMN]
            X_test = labelled_test_set.drop(columns=[TARGET_COLUMN], axis=1)
            y_test = labelled_test_set[TARGET_COLUMN]

            logging.info(f"Final dataset shapes:")
            logging.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
            logging.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
            logging.info(f"Train label distribution: {np.unique(y_train, return_counts=True)}")
            logging.info(f"Test label distribution: {np.unique(y_test, return_counts=True)}")

            # Create arrays
            train_arr = np.c_[np.array(X_train), np.array(y_train)]
            test_arr = np.c_[np.array(X_test), np.array(y_test)]
            
            logging.info(f"Final arrays - Train: {train_arr.shape}, Test: {test_arr.shape}")

            # Save arrays
            logging.info("=== STEP 6: SAVING TRANSFORMED DATA ===")
            self.utils.save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            self.utils.save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            
            logging.info(f"Train array saved to: {self.data_transformation_config.transformed_train_file_path}")
            logging.info(f"Test array saved to: {self.data_transformation_config.transformed_test_file_path}")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logging.info("=== DATA TRANSFORMATION PIPELINE COMPLETED SUCCESSFULLY ===")
            return data_transformation_artifact

        except Exception as e:
            logging.error(f"Error in data transformation pipeline: {str(e)}")
            raise CustomerException(e, sys) from e
