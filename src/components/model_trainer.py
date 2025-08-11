import sys
from typing import List, Tuple
import os
from pandas import DataFrame
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils, load_numpy_array_data
from neuro_mf import ModelFactory

class CustomerSegmentationModel:
    def __init__(self, preprocessing_object: object, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
        logging.info("CustomerSegmentationModel initialized")

    def predict(self, X: DataFrame) -> np.ndarray:
        logging.info("=== CustomerSegmentationModel PREDICT METHOD ===")
        try:
            logging.info(f"Input shape for prediction: {X.shape}")
            logging.info(f"Input data types: {X.dtypes.value_counts().to_dict()}")
            
            # Check for any data quality issues
            nan_count = X.isnull().sum().sum()
            inf_count = np.isinf(X.values).sum()
            logging.info(f"Input data quality - NaN: {nan_count}, Inf: {inf_count}")
            
            # CRITICAL FIX: Check if data is already preprocessed
            # During evaluation, the data comes already preprocessed (all float64)
            # During training, the data comes raw and needs preprocessing
            if all(X.dtypes == 'float64'):
                logging.info("Data appears to be already preprocessed, using directly")
                transformed_feature = X.values  # Convert to numpy array
            else:
                logging.info("Data needs preprocessing, applying transformation")
                transformed_feature = self.preprocessing_object.transform(X)
            
            logging.info(f"Final feature shape for model: {transformed_feature.shape}")
            
            predictions = self.trained_model_object.predict(transformed_feature)
            logging.info(f"Predictions shape: {predictions.shape}")
            logging.info(f"Unique predictions: {np.unique(predictions, return_counts=True)}")
            
            return predictions
        except Exception as e:
            logging.error(f"Error in CustomerSegmentationModel.predict: {str(e)}")
            raise CustomerException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"

class ModelTrainer:
    def __init__(self,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.utils = MainUtils()
        logging.info("ModelTrainer initialized successfully")

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("=== STARTING MODEL TRAINING PIPELINE ===")
        try:
            # Load data
            logging.info("=== STEP 1: LOADING TRAINING DATA ===")
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            
            logging.info(f"Loaded arrays - Train: {train_arr.shape}, Test: {test_arr.shape}")

            # Split features and labels
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info(f"Data split completed:")
            logging.info(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
            logging.info(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

            # Detailed label analysis
            train_unique, train_counts = np.unique(y_train, return_counts=True)
            test_unique, test_counts = np.unique(y_test, return_counts=True)
            
            logging.info(f"TRAINING labels distribution: {dict(zip(train_unique, train_counts))}")
            logging.info(f"TEST labels distribution: {dict(zip(test_unique, test_counts))}")
            
            # Check for data quality issues
            train_nan = np.isnan(x_train).sum()
            test_nan = np.isnan(x_test).sum()
            train_inf = np.isinf(x_train).sum()
            test_inf = np.isinf(x_test).sum()
            
            logging.info(f"Data quality check:")
            logging.info(f"Train - NaN: {train_nan}, Inf: {train_inf}")
            logging.info(f"Test - NaN: {test_nan}, Inf: {test_inf}")

            # Train model
            logging.info("=== STEP 2: MODEL TRAINING ===")
            logging.info(f"Using model config: {self.model_trainer_config.model_config_file_path}")
            logging.info(f"Expected accuracy threshold: {self.model_trainer_config.expected_accuracy}")
            
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            
            best_model_detail = model_factory.get_best_model(
                X=x_train,
                y=y_train,
                base_accuracy=self.model_trainer_config.expected_accuracy
            )
            
            logging.info(f"Best model found: {best_model_detail.best_model}")
            logging.info(f"Best model score: {best_model_detail.best_score}")
            logging.info(f"Best model parameters: {best_model_detail.best_parameters}")

            # Load preprocessor
            logging.info("=== STEP 3: LOADING PREPROCESSOR ===")
            preprocessing_obj = self.utils.load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )
            logging.info("Preprocessor loaded successfully")

            # Check if model meets expectations
            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                error_msg = f"No best model found with score ({best_model_detail.best_score}) more than base score ({self.model_trainer_config.expected_accuracy})"
                logging.error(error_msg)
                raise Exception(error_msg)

            # Create final model
            logging.info("=== STEP 4: CREATING FINAL MODEL ===")
            customer_segmentation_model = CustomerSegmentationModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model_detail.best_model
            )
            logging.info("CustomerSegmentationModel created successfully")

            # Save model
            logging.info("=== STEP 5: SAVING MODEL ===")
            trained_model_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(trained_model_path, exist_ok=True)
            
            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=customer_segmentation_model
            )
            logging.info(f"Model saved to: {self.model_trainer_config.trained_model_file_path}")

            # Calculate metrics on test data using the raw model (not the wrapped one)
            logging.info("=== STEP 6: MODEL EVALUATION ===")
            y_test_pred = best_model_detail.best_model.predict(x_test)
            
            # Detailed evaluation
            logging.info("--- Detailed Model Evaluation ---")
            logging.info(f"Test predictions shape: {y_test_pred.shape}")
            pred_unique, pred_counts = np.unique(y_test_pred, return_counts=True)
            logging.info(f"Test PREDICTIONS distribution: {dict(zip(pred_unique, pred_counts))}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_test_pred)
            logging.info(f"Confusion Matrix:\n{cm}")
            print("Confusion Matrix:\n", cm)
            
            # Classification Report
            try:
                class_report = classification_report(y_test, y_test_pred, output_dict=True)
                logging.info(f"Classification Report: {class_report}")
            except Exception as e:
                logging.warning(f"Could not generate classification report: {e}")
            
            # Calculate metrics
            f1 = f1_score(y_test, y_test_pred, average='weighted')
            precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_test_pred, average='weighted')
            accuracy = accuracy_score(y_test, y_test_pred)

            logging.info(f"=== FINAL MODEL METRICS ===")
            logging.info(f"F1 Score: {f1:.4f}")
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall: {recall:.4f}")
            logging.info(f"Accuracy: {accuracy:.4f}")
            
            # Check for prediction collapse (all same class)
            if len(pred_unique) == 1:
                logging.error(f"MODEL PREDICTION COLLAPSE! All predictions are class {pred_unique[0]}")
                logging.error("This indicates a serious problem with the model or data pipeline")
            elif len(pred_unique) < len(train_unique):
                logging.warning(f"MODEL UNDER-PREDICTING! Only predicting {len(pred_unique)} out of {len(train_unique)} classes")

            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1,
                precision_score=precision,
                recall_score=recall
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )

            logging.info("=== MODEL TRAINING PIPELINE COMPLETED ===")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            
            return model_trainer_artifact

        except Exception as e:
            logging.error(f"Error in model training pipeline: {str(e)}")
            raise CustomerException(e, sys) from e
