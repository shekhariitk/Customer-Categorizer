from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact, DataTransformationArtifact
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, classification_report
from src.exception import CustomerException
from src.pipeline.prediction_pipeline import CustomerData
from src.constant.training_pipeline import TARGET_COLUMN
from src.logger import logging
import sys
import pandas as pd
import numpy as np
from src.ml.model.s3_estimator import CustomerClusterEstimator
from dataclasses import dataclass
from typing import Optional
from src.entity.config_entity import Prediction_config
from src.utils.main_utils import MainUtils, load_numpy_array_data
from src.ml.metric import calculate_metric
from src.entity.artifact_entity import ClassificationMetricArtifact

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    changed_accuracy: float
    best_model_metric_artifact: Optional[ClassificationMetricArtifact]

def convert_test_numpy_array_to_dataframe(array: np.ndarray):
    """Converts numpy array to dataframe"""
    try:
        logging.info(f"Converting array to dataframe. Shape: {array.shape}")
        prediction_config = Prediction_config().__dict__
        columns = list(prediction_config['prediction_schema']['columns'].keys())
        dataframe = pd.DataFrame(array, columns=columns)
        logging.info(f"Successfully converted to dataframe with columns: {len(columns)}")
        return dataframe
    except Exception as e:
        logging.warning(f"Error using prediction config columns: {e}. Using generic column names.")
        # Fallback: create generic column names
        dataframe = pd.DataFrame(array, columns=[f'feature_{i}' for i in range(array.shape[1])])
        logging.info(f"Created dataframe with generic column names: {dataframe.shape}")
        return dataframe

class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.utils = MainUtils()
            logging.info("ModelEvaluation initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing ModelEvaluation: {str(e)}")
            raise CustomerException(e, sys) from e

    def get_best_model(self) -> Optional[CustomerClusterEstimator]:
        try:
            logging.info("Checking for existing best model in S3")
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            
            logging.info(f"S3 bucket: {bucket_name}, Model path: {model_path}")
            
            customer_cluster_estimator = CustomerClusterEstimator(bucket_name=bucket_name,
                                                                model_path=model_path)
            
            if customer_cluster_estimator.is_model_present(model_path=model_path):
                logging.info("Existing model found in S3")
                return customer_cluster_estimator
            else:
                logging.info("No existing model found in S3")
                return None
        except Exception as e:
            logging.error(f"Error checking for best model: {str(e)}")
            raise CustomerException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        logging.info("=== STARTING MODEL EVALUATION ===")
        try:
            # Load TEST data (FIXED: was loading train data before)
            logging.info("=== STEP 1: LOADING TEST DATA ===")
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info(f"Test array loaded. Shape: {test_arr.shape}")
            
            # Split features and labels
            x_test = convert_test_numpy_array_to_dataframe(array=test_arr[:, :-1])
            y_test = test_arr[:, -1]
            
            logging.info(f"Test data split - X: {x_test.shape}, y: {y_test.shape}")
            
            # Analyze test data
            test_unique, test_counts = np.unique(y_test, return_counts=True)
            logging.info(f"TEST TRUE labels distribution: {dict(zip(test_unique, test_counts))}")
            
            # Check data quality
            x_test_nan = x_test.isnull().sum().sum()
            y_test_nan = np.isnan(y_test).sum()
            logging.info(f"Test data quality - X NaN: {x_test_nan}, y NaN: {y_test_nan}")

            # Load trained model
            logging.info("=== STEP 2: LOADING TRAINED MODEL ===")
            trained_model = self.utils.load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info(f"Trained model loaded: {type(trained_model)}")

            # Make predictions with detailed logging
            logging.info("=== STEP 3: MAKING PREDICTIONS ===")
            logging.info("Calling trained_model.predict()...")
            
            try:
                y_hat_trained_model = trained_model.predict(x_test)
                logging.info(f"Predictions successful. Shape: {y_hat_trained_model.shape}")
            except Exception as pred_error:
                logging.error(f"Error during prediction: {str(pred_error)}")
                logging.info(f"Model type: {type(trained_model)}")
                logging.info(f"Model attributes: {dir(trained_model)}")
                raise pred_error
            
            # Analyze predictions
            pred_unique, pred_counts = np.unique(y_hat_trained_model, return_counts=True)
            logging.info(f"PREDICTED labels distribution: {dict(zip(pred_unique, pred_counts))}")
            
            # Check for prediction issues
            if len(pred_unique) == 1:
                logging.error(f"CRITICAL: MODEL PREDICTS ONLY ONE CLASS: {pred_unique[0]}")
                logging.error("This indicates a serious model or data pipeline problem!")
            elif len(pred_unique) < len(test_unique):
                logging.warning(f"WARNING: Model predicts {len(pred_unique)} classes but test has {len(test_unique)} classes")

            # Confusion matrix analysis
            logging.info("=== STEP 4: CONFUSION MATRIX ANALYSIS ===")
            cm = confusion_matrix(y_test, y_hat_trained_model)
            logging.info(f"Confusion Matrix:\n{cm}")
            
            # Analyze confusion matrix
            cm_sum = cm.sum()
            diagonal_sum = np.trace(cm)
            logging.info(f"Confusion matrix - Total predictions: {cm_sum}, Correct predictions: {diagonal_sum}")
            
            # Row and column analysis
            for i, (true_count, pred_count) in enumerate(zip(test_counts, cm.sum(axis=1))):
                logging.info(f"Class {test_unique[i]}: True count = {true_count}, Predicted count = {pred_count}")

            # Calculate metrics
            logging.info("=== STEP 5: CALCULATING METRICS ===")
            
            try:
                trained_model_f1_score = f1_score(y_test, y_hat_trained_model, average='weighted')
                precision = precision_score(y_test, y_hat_trained_model, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_hat_trained_model, average='weighted')
                
                logging.info(f"Trained model metrics:")
                logging.info(f"  F1 Score: {trained_model_f1_score:.4f}")
                logging.info(f"  Precision: {precision:.4f}")
                logging.info(f"  Recall: {recall:.4f}")
                
                # Detailed classification report
                try:
                    class_report = classification_report(y_test, y_hat_trained_model, output_dict=True, zero_division=0)
                    logging.info("Classification Report (per class):")
                    for class_label, metrics in class_report.items():
                        if isinstance(metrics, dict):
                            logging.info(f"  Class {class_label}: Precision={metrics.get('precision', 0):.3f}, Recall={metrics.get('recall', 0):.3f}, F1={metrics.get('f1-score', 0):.3f}")
                except Exception as report_error:
                    logging.warning(f"Could not generate detailed classification report: {report_error}")
                
            except Exception as metric_error:
                logging.error(f"Error calculating metrics: {metric_error}")
                trained_model_f1_score = 0.0
                precision = 0.0
                recall = 0.0

            # Check for existing best model
            logging.info("=== STEP 6: CHECKING EXISTING BEST MODEL ===")
            best_model_f1_score = None
            best_model_metric_artifact = None

            best_model = self.get_best_model()
            if best_model is not None:
                try:
                    logging.info("Evaluating existing best model...")
                    y_hat_best_model = best_model.predict(x_test)
                    best_model_f1_score = f1_score(y_test, y_hat_best_model, average='weighted')
                    
                    best_pred_unique, best_pred_counts = np.unique(y_hat_best_model, return_counts=True)
                    logging.info(f"Best model predictions distribution: {dict(zip(best_pred_unique, best_pred_counts))}")
                    logging.info(f"Best model F1 score: {best_model_f1_score:.4f}")
                    
                    best_model_metric_artifact = calculate_metric(best_model, x_test, y_test)
                    logging.info(f"Best model metric artifact: {best_model_metric_artifact}")
                    
                except Exception as best_model_error:
                    logging.error(f"Error evaluating best model: {best_model_error}")
                    best_model_f1_score = 0.0
            else:
                logging.info("No existing best model to compare against")

            # Calculate improvement
            logging.info("=== STEP 7: FINAL COMPARISON ===")
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            improvement = trained_model_f1_score - tmp_best_model_score
            is_accepted = trained_model_f1_score > tmp_best_model_score
            
            logging.info(f"Model comparison:")
            logging.info(f"  Trained model F1: {trained_model_f1_score:.4f}")
            logging.info(f"  Previous best F1: {tmp_best_model_score:.4f}")
            logging.info(f"  Improvement: {improvement:.4f}")
            logging.info(f"  Model accepted: {is_accepted}")

            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=is_accepted,
                changed_accuracy=improvement,
                best_model_metric_artifact=best_model_metric_artifact
            )

            logging.info(f"Final evaluation result: {result}")
            logging.info("=== MODEL EVALUATION COMPLETED ===")
            
            return result

        except Exception as e:
            logging.error(f"Error in model evaluation: {str(e)}")
            raise CustomerException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            evaluate_model_response = self.evaluate_model()
            
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                best_model_path=self.model_trainer_artifact.trained_model_file_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.changed_accuracy,
                best_model_metric_artifact=evaluate_model_response.best_model_metric_artifact
            )

            logging.info(f"Model evaluation artifact created: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            logging.error(f"Error in initiate_model_evaluation: {str(e)}")
            raise CustomerException(e, sys) from e
