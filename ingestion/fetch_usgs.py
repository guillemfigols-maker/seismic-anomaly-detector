"""
USGS Earthquake Data Extraction using PySpark
Fetches earthquake data from USGS API and saves as partitioned Parquet files
"""
import requests
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, ArrayType
from pyspark.sql import functions as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class USGSDataFetcher:
    """Fetches earthquake data from USGS and processes with PySpark"""

    def __init__(self, output_path="data/raw/earthquakes"):
        self.base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        self.output_path = output_path
        self.spark = SparkSession.builder \
            .appName("USGS Earthquake Data Extraction") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()

    def fetch_data_chunk(self, start_date, end_date):
        """
        Fetch earthquake data for a specific date range

        Args:
            start_date (datetime): Start date for data fetch
            end_date (datetime): End date for data fetch

        Returns:
            list: List of earthquake features
        """
        params = {
            "format": "geojson",
            "starttime": start_date.strftime("%Y-%m-%d"),
            "endtime": end_date.strftime("%Y-%m-%d")
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Fetched {len(data['features'])} earthquakes for {start_date.date()} to {end_date.date()}")
            return data["features"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data for {start_date} to {end_date}: {e}")
            return []

    def fetch_all_data(self, start_date_str, end_date_str, chunk_days=30):
        """
        Fetch all earthquake data in chunks

        Args:
            start_date_str (str): Start date in YYYY-MM-DD format
            end_date_str (str): End date in YYYY-MM-DD format
            chunk_days (int): Number of days per API request chunk

        Returns:
            list: All earthquake features
        """
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        all_data = []
        current = start_date

        while current < end_date:
            chunk_end = min(current + timedelta(days=chunk_days), end_date)
            chunk_data = self.fetch_data_chunk(current, chunk_end)
            all_data.extend(chunk_data)
            current = chunk_end

        logger.info(f"Total earthquakes fetched: {len(all_data)}")
        return all_data

    def process_to_dataframe(self, raw_data):
        """
        Convert raw GeoJSON features to structured PySpark DataFrame

        Args:
            raw_data (list): List of earthquake features from USGS API

        Returns:
            pyspark.sql.DataFrame: Processed earthquake data
        """
        if not raw_data:
            logger.warning("No data to process")
            return None

        # Create DataFrame from raw JSON
        df = self.spark.createDataFrame(raw_data)

        # Extract and flatten nested properties and geometry
        df_processed = df.select(
            F.col("id").alias("event_id"),
            F.col("properties.mag").alias("magnitude"),
            F.col("properties.place").alias("place"),
            F.col("properties.time").alias("time_ms"),
            F.col("properties.updated").alias("updated_ms"),
            F.col("properties.tz").alias("timezone"),
            F.col("properties.url").alias("url"),
            F.col("properties.detail").alias("detail"),
            F.col("properties.felt").alias("felt"),
            F.col("properties.cdi").alias("cdi"),
            F.col("properties.mmi").alias("mmi"),
            F.col("properties.alert").alias("alert"),
            F.col("properties.status").alias("status"),
            F.col("properties.tsunami").alias("tsunami"),
            F.col("properties.sig").alias("significance"),
            F.col("properties.net").alias("network"),
            F.col("properties.code").alias("code"),
            F.col("properties.ids").alias("ids"),
            F.col("properties.sources").alias("sources"),
            F.col("properties.types").alias("types"),
            F.col("properties.nst").alias("num_stations"),
            F.col("properties.dmin").alias("min_distance"),
            F.col("properties.rms").alias("rms"),
            F.col("properties.gap").alias("gap"),
            F.col("properties.magType").alias("magnitude_type"),
            F.col("geometry.coordinates").alias("coordinates")
        )

        # Extract longitude, latitude, depth from coordinates array
        df_processed = df_processed.withColumn("longitude", F.col("coordinates")[0]) \
            .withColumn("latitude", F.col("coordinates")[1]) \
            .withColumn("depth_km", F.col("coordinates")[2]) \
            .drop("coordinates")

        # Convert timestamps to proper datetime
        df_processed = df_processed.withColumn(
            "event_time",
            F.from_unixtime(F.col("time_ms") / 1000).cast("timestamp")
        ).withColumn(
            "updated_time",
            F.from_unixtime(F.col("updated_ms") / 1000).cast("timestamp")
        )

        # Add partition columns for efficient storage
        df_processed = df_processed.withColumn("year", F.year("event_time")) \
            .withColumn("month", F.month("event_time")) \
            .withColumn("day", F.dayofmonth("event_time"))

        return df_processed

    def save_to_parquet(self, df, mode="overwrite"):
        """
        Save DataFrame to partitioned Parquet files

        Args:
            df (pyspark.sql.DataFrame): DataFrame to save
            mode (str): Write mode (overwrite, append)
        """
        if df is None:
            logger.warning("No DataFrame to save")
            return

        df.write \
            .partitionBy("year", "month") \
            .mode(mode) \
            .parquet(self.output_path)

        logger.info(f"Data saved to {self.output_path}")

    def run_pipeline(self, start_date, end_date, chunk_days=30):
        """
        Run the complete data extraction pipeline

        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            chunk_days (int): Number of days per API request chunk
        """
        logger.info("Starting USGS data extraction pipeline")

        # Fetch raw data
        raw_data = self.fetch_all_data(start_date, end_date, chunk_days)

        # Process to DataFrame
        df = self.process_to_dataframe(raw_data)

        if df is not None:
            # Show sample
            logger.info("Sample of processed data:")
            df.show(5, truncate=False)

            # Print schema
            logger.info("DataFrame schema:")
            df.printSchema()

            # Save to Parquet
            self.save_to_parquet(df)

            # Print statistics
            total_count = df.count()
            logger.info(f"Total records processed: {total_count}")

            logger.info("Summary statistics:")
            df.select("magnitude", "depth_km", "significance").summary().show()

        logger.info("Pipeline completed")

    def stop(self):
        """Stop the Spark session"""
        self.spark.stop()


def main():
    """Main execution function"""
    fetcher = USGSDataFetcher(output_path="data/raw/earthquakes")

    try:
        # Fetch data for 2024
        fetcher.run_pipeline(
            start_date="2024-01-01",
            end_date="2025-01-01",
            chunk_days=30
        )
    finally:
        fetcher.stop()


if __name__ == "__main__":
    main()
