"""
Feature Engineering for Aftershock Prediction
Generates features from earthquake data to predict aftershock probability
"""
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeismicFeatureEngine:
    """Generate features for aftershock prediction from earthquake data"""

    def __init__(self, spark=None):
        if spark is None:
            self.spark = SparkSession.builder \
                .appName("Seismic Feature Engineering") \
                .config("spark.sql.adaptive.enabled", "true") \
                .getOrCreate()
        else:
            self.spark = spark

    def load_earthquake_data(self, path):
        """
        Load earthquake data from Parquet files

        Args:
            path (str): Path to Parquet files

        Returns:
            pyspark.sql.DataFrame: Loaded earthquake data
        """
        logger.info(f"Loading earthquake data from {path}")
        df = self.spark.read.parquet(path)
        logger.info(f"Loaded {df.count()} earthquake records")
        return df

    def create_temporal_features(self, df):
        """
        Create time-based features

        Args:
            df (pyspark.sql.DataFrame): Input earthquake data

        Returns:
            pyspark.sql.DataFrame: DataFrame with temporal features
        """
        logger.info("Creating temporal features")

        df = df.withColumn("hour", F.hour("event_time")) \
            .withColumn("day_of_week", F.dayofweek("event_time")) \
            .withColumn("day_of_year", F.dayofyear("event_time"))

        # Time since last earthquake (seconds)
        window_spec = Window.partitionBy("network").orderBy("event_time")
        df = df.withColumn(
            "time_since_prev_event",
            F.col("event_time").cast("long") - F.lag("event_time").over(window_spec).cast("long")
        )

        return df

    def create_spatial_features(self, df):
        """
        Create location-based features including spatial clustering

        Args:
            df (pyspark.sql.DataFrame): Input earthquake data

        Returns:
            pyspark.sql.DataFrame: DataFrame with spatial features
        """
        logger.info("Creating spatial features")

        # Grid cell assignment (0.5 degree cells)
        df = df.withColumn("lat_grid", F.floor(F.col("latitude") / 0.5)) \
            .withColumn("lon_grid", F.floor(F.col("longitude") / 0.5))

        # Count earthquakes in same grid cell (spatial density)
        window_spatial = Window.partitionBy("lat_grid", "lon_grid")
        df = df.withColumn("spatial_density", F.count("event_id").over(window_spatial))

        return df

    def create_seismic_features(self, df):
        """
        Create earthquake-specific features

        Args:
            df (pyspark.sql.DataFrame): Input earthquake data

        Returns:
            pyspark.sql.DataFrame: DataFrame with seismic features
        """
        logger.info("Creating seismic features")

        # Rolling statistics within time and space windows
        # Define window: 30 days before, same grid cell
        days_30_seconds = 30 * 24 * 60 * 60
        window_spec = Window.partitionBy("lat_grid", "lon_grid") \
            .orderBy(F.col("event_time").cast("long")) \
            .rangeBetween(-days_30_seconds, 0)

        df = df.withColumn(
            "mag_mean_30d",
            F.avg("magnitude").over(window_spec)
        ).withColumn(
            "mag_max_30d",
            F.max("magnitude").over(window_spec)
        ).withColumn(
            "mag_std_30d",
            F.stddev("magnitude").over(window_spec)
        ).withColumn(
            "event_count_30d",
            F.count("event_id").over(window_spec)
        ).withColumn(
            "depth_mean_30d",
            F.avg("depth_km").over(window_spec)
        )

        # Energy release (simplified)
        df = df.withColumn(
            "energy_release",
            F.pow(10, 1.5 * F.col("magnitude") + 4.8)
        )

        # Cumulative energy in past 30 days
        df = df.withColumn(
            "cumulative_energy_30d",
            F.sum("energy_release").over(window_spec)
        )

        return df

    def create_aftershock_labels(self, df, threshold_magnitude=4.0, time_window_days=30):
        """
        Create labels for aftershock prediction
        Label = 1 if there's a significant earthquake (>threshold) within time_window after this event

        Args:
            df (pyspark.sql.DataFrame): Input earthquake data
            threshold_magnitude (float): Minimum magnitude to consider as significant aftershock
            time_window_days (int): Days to look ahead for aftershocks

        Returns:
            pyspark.sql.DataFrame: DataFrame with aftershock labels
        """
        logger.info(f"Creating aftershock labels (mag>{threshold_magnitude}, window={time_window_days}d)")

        time_window_seconds = time_window_days * 24 * 60 * 60

        # Window looking forward in time within same spatial region
        window_spec = Window.partitionBy("lat_grid", "lon_grid") \
            .orderBy(F.col("event_time").cast("long")) \
            .rangeBetween(1, time_window_seconds)

        # Check if there's any earthquake above threshold magnitude in the future window
        df = df.withColumn(
            "future_max_mag",
            F.max("magnitude").over(window_spec)
        )

        df = df.withColumn(
            "has_aftershock",
            F.when(F.col("future_max_mag") >= threshold_magnitude, 1).otherwise(0)
        )

        # Also create probability score based on max magnitude in window
        df = df.withColumn(
            "aftershock_magnitude",
            F.coalesce(F.col("future_max_mag"), F.lit(0.0))
        )

        return df

    def select_features_for_training(self, df):
        """
        Select and clean features for model training

        Args:
            df (pyspark.sql.DataFrame): Input DataFrame with all features

        Returns:
            pyspark.sql.DataFrame: Clean DataFrame ready for ML
        """
        logger.info("Selecting features for training")

        feature_columns = [
            "event_id",
            "event_time",
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
            "cumulative_energy_30d",
            "has_aftershock",
            "aftershock_magnitude"
        ]

        # Select features and handle nulls
        df_features = df.select(*feature_columns)

        # Fill nulls with appropriate values
        df_features = df_features.fillna({
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

        return df_features

    def run_feature_pipeline(self, input_path, output_path):
        """
        Run complete feature engineering pipeline

        Args:
            input_path (str): Path to raw earthquake Parquet data
            output_path (str): Path to save processed features
        """
        logger.info("Starting feature engineering pipeline")

        # Load data
        df = self.load_earthquake_data(input_path)

        # Filter for quality (remove very small earthquakes and nulls)
        df = df.filter(
            (F.col("magnitude").isNotNull()) &
            (F.col("latitude").isNotNull()) &
            (F.col("longitude").isNotNull()) &
            (F.col("magnitude") >= 0)
        )

        # Create features
        df = self.create_temporal_features(df)
        df = self.create_spatial_features(df)
        df = self.create_seismic_features(df)
        df = self.create_aftershock_labels(df)

        # Select and clean
        df_final = self.select_features_for_training(df)

        # Show sample
        logger.info("Sample of engineered features:")
        df_final.show(5)

        logger.info("Feature schema:")
        df_final.printSchema()

        # Save to Parquet
        logger.info(f"Saving features to {output_path}")
        df_final.write.mode("overwrite").parquet(output_path)

        # Statistics
        total_events = df_final.count()
        aftershock_events = df_final.filter(F.col("has_aftershock") == 1).count()
        logger.info(f"Total events: {total_events}")
        logger.info(f"Events with aftershocks: {aftershock_events} ({100*aftershock_events/total_events:.2f}%)")

        logger.info("Feature engineering completed")

        return df_final

    def stop(self):
        """Stop the Spark session"""
        self.spark.stop()


def main():
    """Main execution function"""
    engine = SeismicFeatureEngine()

    try:
        engine.run_feature_pipeline(
            input_path="data/raw/earthquakes",
            output_path="data/processed/features"
        )
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
