import sys
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from src.constant.training_pipeline import TARGET_COLUMN
from src.entity.config_entity import PCAConfig
from src.exception import CustomerException
from src.logger import logging

class CreateClusters:
    def __init__(self):
        self.pca_config = PCAConfig()
        self.pca_object = None
        self.kmeans_model = None
        self.label_mapping = None
        logging.info("CreateClusters initialized successfully")

    def get_train_dataset_using_pca(self, preprocessed_data: DataFrame):
        """Fit PCA on train data and return reduced dataset"""
        try:
            logging.info(f"Starting PCA transformation on TRAIN data with shape: {preprocessed_data.shape}")
            
            # Check for any NaN or infinite values
            if preprocessed_data.isnull().sum().sum() > 0:
                logging.warning(f"Found {preprocessed_data.isnull().sum().sum()} NaN values in training data")
            
            if np.isinf(preprocessed_data.values).sum() > 0:
                logging.warning(f"Found {np.isinf(preprocessed_data.values).sum()} infinite values in training data")
            
            self.pca_object = PCA(**self.pca_config.__dict__).fit(preprocessed_data)
            reduced_dataset = self.pca_object.transform(preprocessed_data)
            
            logging.info(f"PCA transformation on TRAIN data completed. Reduced shape: {reduced_dataset.shape}")
            logging.info(f"PCA explained variance ratio: {self.pca_object.explained_variance_ratio_}")
            
            return reduced_dataset
        except Exception as e:
            logging.error(f"Error in get_train_dataset_using_pca: {str(e)}")
            raise CustomerException(e, sys)

    def initialize_train_clustering(self, preprocessed_data: DataFrame) -> DataFrame:
        """Fit KMeans on train data and assign clusters"""
        try:
            logging.info(f"Initializing clustering for TRAIN data with shape: {preprocessed_data.shape}")
            
            # Apply PCA transformation
            reduced_dataset = self.get_train_dataset_using_pca(preprocessed_data)
            
            # Fit KMeans
            logging.info("Fitting KMeans model on reduced training data")
            self.kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(reduced_dataset)
            
            # Assign cluster labels
            cluster_labels = self.kmeans_model.labels_.astype(int)
            preprocessed_data = preprocessed_data.copy()  # Avoid modifying original
            preprocessed_data[TARGET_COLUMN] = cluster_labels
            
            # Log cluster statistics
            cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
            logging.info(f"TRAIN cluster distribution: {cluster_counts.to_dict()}")
            logging.info(f"KMeans inertia: {self.kmeans_model.inertia_}")
            logging.info(f"KMeans cluster centers shape: {self.kmeans_model.cluster_centers_.shape}")
            
            return preprocessed_data
        except Exception as e:
            logging.error(f"Error in initialize_train_clustering: {str(e)}")
            raise CustomerException(e, sys)

    def get_test_dataset_using_pca(self, preprocessed_data: DataFrame):
        """Transform test data using already fitted PCA"""
        try:
            if self.pca_object is None:
                raise ValueError("PCA model not fitted. Train clustering first.")
            
            logging.info(f"Transforming TEST data using fitted PCA. Input shape: {preprocessed_data.shape}")
            
            # Check for any NaN or infinite values
            if preprocessed_data.isnull().sum().sum() > 0:
                logging.warning(f"Found {preprocessed_data.isnull().sum().sum()} NaN values in test data")
            
            if np.isinf(preprocessed_data.values).sum() > 0:
                logging.warning(f"Found {np.isinf(preprocessed_data.values).sum()} infinite values in test data")
            
            pca_transformed = self.pca_object.transform(preprocessed_data)
            
            logging.info(f"PCA transformation on TEST data completed. Output shape: {pca_transformed.shape}")
            
            return pca_transformed
        except Exception as e:
            logging.error(f"Error in get_test_dataset_using_pca: {str(e)}")
            raise CustomerException(e, sys)

    def align_cluster_labels(self, y_true, y_pred):
        """Align test cluster labels with train cluster labels using Hungarian algorithm"""
        try:
            logging.info(f"Attempting cluster label alignment. y_true shape: {np.array(y_true).shape}, y_pred shape: {np.array(y_pred).shape}")
            
            # Ensure both arrays have the same length
            if len(y_true) != len(y_pred):
                logging.error(f"Size mismatch: y_true has {len(y_true)} samples, y_pred has {len(y_pred)} samples")
                raise ValueError(f"Input arrays must have the same length. Got {len(y_true)} and {len(y_pred)}")
            
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            logging.info(f"Confusion matrix for alignment:\n{cm}")
            
            # Use Hungarian algorithm to find optimal mapping
            row_ind, col_ind = linear_sum_assignment(-cm)
            mapping = {col: row for row, col in zip(row_ind, col_ind)}
            
            # Apply mapping
            aligned_labels = np.array([mapping.get(label, label) for label in y_pred])
            
            logging.info(f"Cluster label mapping created: {mapping}")
            logging.info(f"Original test clusters: {np.unique(y_pred, return_counts=True)}")
            logging.info(f"Aligned test clusters: {np.unique(aligned_labels, return_counts=True)}")
            
            return aligned_labels, mapping
        except Exception as e:
            logging.error(f"Label alignment failed: {str(e)}. Using original labels.")
            # Return original predictions with identity mapping
            unique_labels = np.unique(y_pred)
            identity_mapping = {label: label for label in unique_labels}
            return y_pred, identity_mapping

    def initialize_test_clustering(self, preprocessed_data: DataFrame, y_true_train_labels=None) -> DataFrame:
        """Predict clusters on test data and optionally align labels"""
        try:
            logging.info(f"Initializing clustering for TEST data with shape: {preprocessed_data.shape}")
            
            if self.kmeans_model is None:
                raise ValueError("KMeans model not fitted. Train clustering first.")
            
            # Apply PCA transformation
            reduced_dataset = self.get_test_dataset_using_pca(preprocessed_data)
            
            # Predict clusters
            logging.info("Predicting clusters on test data using trained KMeans model")
            predicted_clusters = self.kmeans_model.predict(reduced_dataset).astype(int)
            
            # Log initial predictions
            initial_counts = pd.Series(predicted_clusters).value_counts().sort_index()
            logging.info(f"Initial TEST cluster predictions: {initial_counts.to_dict()}")
            
            # Apply label alignment if train labels are provided
            if y_true_train_labels is not None:
                logging.info("Attempting to align test cluster labels with training labels")
                
                # Check if we can use the train labels for alignment
                if len(y_true_train_labels) == len(predicted_clusters):
                    logging.info("Using provided train labels for cluster alignment")
                    predicted_clusters, self.label_mapping = self.align_cluster_labels(y_true_train_labels, predicted_clusters)
                else:
                    logging.warning(f"Cannot align: train labels length ({len(y_true_train_labels)}) != test predictions length ({len(predicted_clusters)})")
                    logging.warning("Skipping alignment and using original predictions")
                    self.label_mapping = {i: i for i in range(3)}
            else:
                logging.info("No training labels provided for alignment. Using original predictions.")
                self.label_mapping = {i: i for i in range(3)}
            
            # Assign final cluster labels
            preprocessed_data = preprocessed_data.copy()  # Avoid modifying original
            preprocessed_data[TARGET_COLUMN] = predicted_clusters
            
            # Log final cluster distribution
            final_counts = pd.Series(predicted_clusters).value_counts().sort_index()
            logging.info(f"Final TEST cluster distribution: {final_counts.to_dict()}")
            logging.info(f"Applied label mapping: {self.label_mapping}")
            logging.info("Clustering on TEST data completed successfully")
            
            return preprocessed_data
        except Exception as e:
            logging.error(f"Error in initialize_test_clustering: {str(e)}")
            raise CustomerException(e, sys)
