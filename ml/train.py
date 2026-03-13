"""
Model Training for Aftershock Probability Prediction
Trains ML models using PySpark MLlib and tracks with MLflow
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import mlflow
import mlflow.spark
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AftershockModelTrainer:
    """Train and evaluate models for aftershock prediction"""

    def __init__(self, spark=None):
        if spark is None:
            self.spark = SparkSession.builder \
                .appName("Aftershock Model Training") \
                .config("spark.sql.adaptive.enabled", "true") \
                .getOrCreate()
        else:
            self.spark = spark

        # MLflow setup
        mlflow.set_experiment("aftershock_prediction")

    def load_features(self, path):
        """
        Load engineered features from Parquet

        Args:
            path (str): Path to feature Parquet files

        Returns:
            pyspark.sql.DataFrame: Features DataFrame
        """
        logger.info(f"Loading features from {path}")
        df = self.spark.read.parquet(path)
        logger.info(f"Loaded {df.count()} feature records")
        return df

    def prepare_training_data(self, df, test_split=0.2):
        """
        Prepare data for training with train/test split

        Args:
            df (pyspark.sql.DataFrame): Features DataFrame
            test_split (float): Fraction of data for testing

        Returns:
            tuple: (train_df, test_df, feature_columns)
        """
        logger.info("Preparing training data")

        # Define feature columns (excluding ID, time, and label columns)
        feature_columns = [
            "magnitude",
            "depth_km",
            "latitude",
            "longitude",
            "significance",
            "num_stations",
            "gap",
            "rms",
            "hour",
            "day_of_week",
            "day_of_year",
            "time_since_prev_event",
            "spatial_density",
            "mag_mean_30d",
            "mag_max_30d",
            "mag_std_30d",
            "event_count_30d",
            "depth_mean_30d",
            "energy_release",
            "cumulative_energy_30d"
        ]

        # Check class balance
        class_counts = df.groupBy("has_aftershock").count()
        logger.info("Class distribution:")
        class_counts.show()

        # Time-based split to prevent data leakage
        # Use events before 2024-11-01 for training, after for testing
        train_df = df.filter(F.col("event_time") < "2024-11-01")
        test_df = df.filter(F.col("event_time") >= "2024-11-01")

        train_count = train_df.count()
        test_count = test_df.count()

        logger.info(f"Training set: {train_count} samples")
        logger.info(f"Test set: {test_count} samples")

        return train_df, test_df, feature_columns

    def create_pipeline(self, feature_columns, model_type="random_forest"):
        """
        Create ML pipeline with feature processing and model

        Args:
            feature_columns (list): List of feature column names
            model_type (str): Type of model (random_forest, gbt, logistic_regression)

        Returns:
            pyspark.ml.Pipeline: ML pipeline
        """
        logger.info(f"Creating pipeline with {model_type} model")

        # Assemble features into vector
        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="features_raw",
            handleInvalid="skip"
        )

        # Scale features
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )

        # Select model
        if model_type == "random_forest":
            model = RandomForestClassifier(
                labelCol="has_aftershock",
                featuresCol="features",
                numTrees=100,
                maxDepth=10,
                minInstancesPerNode=10,
                seed=42
            )
        elif model_type == "gbt":
            model = GBTClassifier(
                labelCol="has_aftershock",
                featuresCol="features",
                maxIter=50,
                maxDepth=8,
                seed=42
            )
        elif model_type == "logistic_regression":
            model = LogisticRegression(
                labelCol="has_aftershock",
                featuresCol="features",
                maxIter=100,
                regParam=0.01,
                elasticNetParam=0.5
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Create pipeline
        pipeline = Pipeline(stages=[assembler, scaler, model])

        return pipeline

    def evaluate_model(self, predictions):
        """
        Evaluate model performance

        Args:
            predictions (pyspark.sql.DataFrame): Predictions DataFrame

        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating model")

        # Binary classification metrics
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol="has_aftershock",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        auc_roc = binary_evaluator.evaluate(predictions)

        binary_evaluator.setMetricName("areaUnderPR")
        auc_pr = binary_evaluator.evaluate(predictions)

        # Multiclass metrics for additional insights
        mc_evaluator = MulticlassClassificationEvaluator(
            labelCol="has_aftershock",
            predictionCol="prediction"
        )

        mc_evaluator.setMetricName("accuracy")
        accuracy = mc_evaluator.evaluate(predictions)

        mc_evaluator.setMetricName("f1")
        f1 = mc_evaluator.evaluate(predictions)

        mc_evaluator.setMetricName("weightedPrecision")
        precision = mc_evaluator.evaluate(predictions)

        mc_evaluator.setMetricName("weightedRecall")
        recall = mc_evaluator.evaluate(predictions)

        metrics = {
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

        logger.info("Evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def train_and_evaluate(self, train_df, test_df, feature_columns, model_type="random_forest"):
        """
        Train model and evaluate on test set with MLflow tracking

        Args:
            train_df (pyspark.sql.DataFrame): Training data
            test_df (pyspark.sql.DataFrame): Test data
            feature_columns (list): Feature column names
            model_type (str): Model type

        Returns:
            tuple: (fitted_model, metrics)
        """
        with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("num_features", len(feature_columns))
            mlflow.log_param("train_size", train_df.count())
            mlflow.log_param("test_size", test_df.count())

            # Create and train pipeline
            pipeline = self.create_pipeline(feature_columns, model_type)

            logger.info("Training model...")
            fitted_model = pipeline.fit(train_df)

            # Make predictions on test set
            logger.info("Making predictions on test set...")
            predictions = fitted_model.transform(test_df)

            # Evaluate
            metrics = self.evaluate_model(predictions)

            # Log metrics to MLflow
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)

            # Log model
            mlflow.spark.log_model(fitted_model, "model")

            # Show sample predictions
            logger.info("Sample predictions:")
            predictions.select(
                "event_id",
                "magnitude",
                "has_aftershock",
                "prediction",
                "probability"
            ).show(10, truncate=False)

            return fitted_model, metrics

    def run_training_pipeline(self, features_path, model_output_path, model_types=None):
        """
        Run complete training pipeline for multiple models

        Args:
            features_path (str): Path to feature Parquet files
            model_output_path (str): Path to save best model
            model_types (list): List of model types to train

        Returns:
            dict: Results for all models
        """
        if model_types is None:
            model_types = ["random_forest", "gbt", "logistic_regression"]

        logger.info("Starting model training pipeline")

        # Load features
        df = self.load_features(features_path)

        # Prepare data
        train_df, test_df, feature_columns = self.prepare_training_data(df)

        # Train multiple models
        results = {}
        best_model = None
        best_auc = 0

        for model_type in model_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_type} model")
            logger.info(f"{'='*60}")

            model, metrics = self.train_and_evaluate(
                train_df, test_df, feature_columns, model_type
            )

            results[model_type] = metrics

            # Track best model by AUC-ROC
            if metrics["auc_roc"] > best_auc:
                best_auc = metrics["auc_roc"]
                best_model = model
                best_model_type = model_type

        # Save best model
        logger.info(f"\nBest model: {best_model_type} (AUC-ROC: {best_auc:.4f})")
        logger.info(f"Saving best model to {model_output_path}")
        best_model.write().overwrite().save(model_output_path)

        logger.info("\nTraining pipeline completed")
        logger.info("\nFinal Results Summary:")
        for model_type, metrics in results.items():
            logger.info(f"{model_type}:")
            logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
            logger.info(f"  AUC-PR: {metrics['auc_pr']:.4f}")
            logger.info(f"  F1: {metrics['f1']:.4f}")

        return results

    def stop(self):
        """Stop the Spark session"""
        self.spark.stop()


def main():
    """Main execution function"""
    trainer = AftershockModelTrainer()

    try:
        results = trainer.run_training_pipeline(
            features_path="data/processed/features",
            model_output_path="data/models/best_model"
        )
    finally:
        trainer.stop()


if __name__ == "__main__":
    main()
