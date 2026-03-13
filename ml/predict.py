"""
Aftershock Prediction Service
Loads trained model and generates predictions for new earthquake data
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AftershockPredictor:
    """Generate aftershock probability predictions using trained model"""

    def __init__(self, model_path, spark=None):
        """
        Initialize predictor with trained model

        Args:
            model_path (str): Path to saved PipelineModel
            spark (SparkSession, optional): Spark session
        """
        if spark is None:
            self.spark = SparkSession.builder \
                .appName("Aftershock Prediction") \
                .config("spark.sql.adaptive.enabled", "true") \
                .getOrCreate()
        else:
            self.spark = spark

        logger.info(f"Loading model from {model_path}")
        self.model = PipelineModel.load(model_path)
        logger.info("Model loaded successfully")

    def load_recent_events(self, earthquake_data_path, days_back=7):
        """
        Load recent earthquake events for prediction

        Args:
            earthquake_data_path (str): Path to earthquake Parquet data
            days_back (int): Number of days to look back

        Returns:
            pyspark.sql.DataFrame: Recent earthquake events
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")

        logger.info(f"Loading events since {cutoff_str}")
        df = self.spark.read.parquet(earthquake_data_path)
        df = df.filter(F.col("event_time") >= cutoff_str)

        logger.info(f"Loaded {df.count()} recent events")
        return df

    def prepare_prediction_features(self, df):
        """
        Prepare features for prediction (mirror feature engineering)

        Args:
            df (pyspark.sql.DataFrame): Raw earthquake data

        Returns:
            pyspark.sql.DataFrame: DataFrame with prediction features
        """
        from ml.features import SeismicFeatureEngine

        logger.info("Preparing prediction features")

        # Use feature engine to create features
        engine = SeismicFeatureEngine(self.spark)

        # Apply feature engineering steps
        df = engine.create_temporal_features(df)
        df = engine.create_spatial_features(df)
        df = engine.create_seismic_features(df)

        # Select relevant columns (no labels needed for prediction)
        feature_columns = [
            "event_id",
            "event_time",
            "magnitude",
            "depth_km",
            "latitude",
            "longitude",
            "place",
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

        df = df.select(*feature_columns)

        # Fill nulls
        df = df.fillna({
            "time_since_prev_event": 0,
            "mag_mean_30d": 0,
            "mag_max_30d": 0,
            "mag_std_30d": 0,
            "event_count_30d": 1,
            "depth_mean_30d": 0,
            "cumulative_energy_30d": 0,
            "num_stations": 0,
            "gap": 180,
            "rms": 0,
            "significance": 0
        })

        return df

    def predict(self, df):
        """
        Generate predictions for earthquake data

        Args:
            df (pyspark.sql.DataFrame): Features DataFrame

        Returns:
            pyspark.sql.DataFrame: Predictions with probabilities
        """
        logger.info("Generating predictions")

        predictions = self.model.transform(df)

        # Extract probability of aftershock (class 1)
        predictions = predictions.withColumn(
            "aftershock_probability",
            F.col("probability").getItem(1)
        )

        # Create risk level categories
        predictions = predictions.withColumn(
            "risk_level",
            F.when(F.col("aftershock_probability") >= 0.7, "HIGH")
            .when(F.col("aftershock_probability") >= 0.4, "MEDIUM")
            .otherwise("LOW")
        )

        return predictions

    def generate_predictions_report(self, predictions, top_n=20):
        """
        Generate summary report of predictions

        Args:
            predictions (pyspark.sql.DataFrame): Predictions DataFrame
            top_n (int): Number of top risk events to show

        Returns:
            pyspark.sql.DataFrame: Top risk events
        """
        logger.info("Generating predictions report")

        # Summary statistics
        total_events = predictions.count()
        high_risk = predictions.filter(F.col("risk_level") == "HIGH").count()
        medium_risk = predictions.filter(F.col("risk_level") == "MEDIUM").count()
        low_risk = predictions.filter(F.col("risk_level") == "LOW").count()

        logger.info(f"\nPrediction Summary:")
        logger.info(f"Total events analyzed: {total_events}")
        logger.info(f"High risk: {high_risk} ({100*high_risk/total_events:.1f}%)")
        logger.info(f"Medium risk: {medium_risk} ({100*medium_risk/total_events:.1f}%)")
        logger.info(f"Low risk: {low_risk} ({100*low_risk/total_events:.1f}%)")

        # Top risk events
        top_risk_events = predictions.select(
            "event_id",
            "event_time",
            "magnitude",
            "depth_km",
            "latitude",
            "longitude",
            "place",
            "aftershock_probability",
            "risk_level"
        ).orderBy(F.desc("aftershock_probability")).limit(top_n)

        logger.info(f"\nTop {top_n} High-Risk Events:")
        top_risk_events.show(top_n, truncate=False)

        return top_risk_events

    def save_predictions(self, predictions, output_path):
        """
        Save predictions to Parquet files

        Args:
            predictions (pyspark.sql.DataFrame): Predictions DataFrame
            output_path (str): Output path for predictions
        """
        logger.info(f"Saving predictions to {output_path}")

        predictions.select(
            "event_id",
            "event_time",
            "magnitude",
            "depth_km",
            "latitude",
            "longitude",
            "place",
            "aftershock_probability",
            "risk_level",
            "prediction"
        ).write.mode("overwrite").parquet(output_path)

        logger.info("Predictions saved successfully")

    def run_prediction_pipeline(self, earthquake_data_path, output_path, days_back=7):
        """
        Run complete prediction pipeline

        Args:
            earthquake_data_path (str): Path to raw earthquake data
            output_path (str): Path to save predictions
            days_back (int): Days to look back for events

        Returns:
            pyspark.sql.DataFrame: Top risk events
        """
        logger.info("Starting prediction pipeline")

        # Load recent events
        df = self.load_recent_events(earthquake_data_path, days_back)

        if df.count() == 0:
            logger.warning("No recent events found")
            return None

        # Prepare features
        df_features = self.prepare_prediction_features(df)

        # Generate predictions
        predictions = self.predict(df_features)

        # Generate report
        top_risk = self.generate_predictions_report(predictions)

        # Save predictions
        self.save_predictions(predictions, output_path)

        logger.info("Prediction pipeline completed")

        return top_risk

    def predict_single_event(self, magnitude, depth_km, latitude, longitude,
                            recent_magnitude_mean=None, recent_event_count=None):
        """
        Predict aftershock probability for a single event (simplified)

        Args:
            magnitude (float): Earthquake magnitude
            depth_km (float): Depth in kilometers
            latitude (float): Latitude
            longitude (float): Longitude
            recent_magnitude_mean (float): Mean magnitude in region (optional)
            recent_event_count (int): Recent event count in region (optional)

        Returns:
            dict: Prediction result
        """
        from pyspark.sql import Row

        # Create minimal feature set
        event_data = Row(
            magnitude=magnitude,
            depth_km=depth_km,
            latitude=latitude,
            longitude=longitude,
            significance=magnitude * 100,  # Rough estimate
            num_stations=10,
            gap=100.0,
            rms=0.5,
            hour=12,
            day_of_week=3,
            day_of_year=180,
            time_since_prev_event=3600,
            spatial_density=5,
            mag_mean_30d=recent_magnitude_mean or magnitude,
            mag_max_30d=magnitude * 1.2,
            mag_std_30d=0.5,
            event_count_30d=recent_event_count or 10,
            depth_mean_30d=depth_km,
            energy_release=10 ** (1.5 * magnitude + 4.8),
            cumulative_energy_30d=10 ** (1.5 * magnitude + 5.5)
        )

        df = self.spark.createDataFrame([event_data])
        prediction = self.model.transform(df)

        result = prediction.select("prediction", "probability").collect()[0]
        prob = float(result["probability"][1])

        return {
            "aftershock_probability": prob,
            "risk_level": "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW",
            "prediction": int(result["prediction"])
        }

    def stop(self):
        """Stop the Spark session"""
        self.spark.stop()


def main():
    """Main execution function"""
    predictor = AftershockPredictor(model_path="data/models/best_model")

    try:
        # Run prediction pipeline
        top_risk = predictor.run_prediction_pipeline(
            earthquake_data_path="data/raw/earthquakes",
            output_path="data/predictions/latest",
            days_back=7
        )

        # Example: Predict for a single hypothetical event
        logger.info("\nExample single event prediction:")
        result = predictor.predict_single_event(
            magnitude=5.5,
            depth_km=10.0,
            latitude=34.05,
            longitude=-118.25,
            recent_magnitude_mean=3.2,
            recent_event_count=45
        )
        logger.info(f"Magnitude 5.5 earthquake prediction:")
        logger.info(f"  Aftershock probability: {result['aftershock_probability']:.3f}")
        logger.info(f"  Risk level: {result['risk_level']}")

    finally:
        predictor.stop()


if __name__ == "__main__":
    main()
