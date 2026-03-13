#!/usr/bin/env python3
"""
End-to-end pipeline runner for seismic anomaly detection
Orchestrates data extraction, feature engineering, training, and prediction
"""
import argparse
import logging
from datetime import datetime
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_data_extraction(start_date, end_date, output_path="data/raw/earthquakes"):
    """
    Run data extraction from USGS

    Args:
        start_date (str): Start date YYYY-MM-DD
        end_date (str): End date YYYY-MM-DD
        output_path (str): Output path for raw data
    """
    from ingestion.fetch_usgs import USGSDataFetcher

    logger.info("=" * 60)
    logger.info("STEP 1: DATA EXTRACTION")
    logger.info("=" * 60)

    fetcher = USGSDataFetcher(output_path=output_path)
    try:
        fetcher.run_pipeline(start_date, end_date)
        logger.info("✓ Data extraction completed")
    finally:
        fetcher.stop()


def run_feature_engineering(input_path="data/raw/earthquakes",
                           output_path="data/processed/features"):
    """
    Run feature engineering

    Args:
        input_path (str): Path to raw earthquake data
        output_path (str): Path to save features
    """
    from ml.features import SeismicFeatureEngine

    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("=" * 60)

    engine = SeismicFeatureEngine()
    try:
        engine.run_feature_pipeline(input_path, output_path)
        logger.info("✓ Feature engineering completed")
    finally:
        engine.stop()


def run_model_training(features_path="data/processed/features",
                      model_output_path="data/models/best_model"):
    """
    Run model training

    Args:
        features_path (str): Path to features
        model_output_path (str): Path to save model
    """
    from ml.train import AftershockModelTrainer

    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: MODEL TRAINING")
    logger.info("=" * 60)

    trainer = AftershockModelTrainer()
    try:
        results = trainer.run_training_pipeline(features_path, model_output_path)
        logger.info("✓ Model training completed")
        return results
    finally:
        trainer.stop()


def run_predictions(model_path="data/models/best_model",
                   earthquake_data_path="data/raw/earthquakes",
                   output_path="data/predictions/latest",
                   days_back=7):
    """
    Run predictions on recent data

    Args:
        model_path (str): Path to trained model
        earthquake_data_path (str): Path to earthquake data
        output_path (str): Path to save predictions
        days_back (int): Days to look back for predictions
    """
    from ml.predict import AftershockPredictor

    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: GENERATING PREDICTIONS")
    logger.info("=" * 60)

    predictor = AftershockPredictor(model_path)
    try:
        top_risk = predictor.run_prediction_pipeline(
            earthquake_data_path,
            output_path,
            days_back
        )
        logger.info("✓ Predictions completed")
        return top_risk
    finally:
        predictor.stop()


def main():
    """Main pipeline orchestrator"""
    parser = argparse.ArgumentParser(
        description="Seismic Anomaly Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_pipeline.py --full

  # Run only data extraction
  python run_pipeline.py --extract --start-date 2024-01-01 --end-date 2025-01-01

  # Run only feature engineering
  python run_pipeline.py --features

  # Run only training
  python run_pipeline.py --train

  # Run only predictions
  python run_pipeline.py --predict --days-back 14
        """
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline (extract, features, train, predict)"
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Run data extraction only"
    )
    parser.add_argument(
        "--features",
        action="store_true",
        help="Run feature engineering only"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run model training only"
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run predictions only"
    )
    parser.add_argument(
        "--start-date",
        default="2024-01-01",
        help="Start date for data extraction (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        default="2025-01-01",
        help="End date for data extraction (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="Number of days to look back for predictions"
    )

    args = parser.parse_args()

    # Validate arguments
    if not any([args.full, args.extract, args.features, args.train, args.predict]):
        parser.print_help()
        sys.exit(1)

    start_time = datetime.now()
    logger.info("Starting Seismic Anomaly Detection Pipeline")
    logger.info(f"Start time: {start_time}")

    try:
        if args.full or args.extract:
            run_data_extraction(args.start_date, args.end_date)

        if args.full or args.features:
            run_feature_engineering()

        if args.full or args.train:
            run_model_training()

        if args.full or args.predict:
            run_predictions(days_back=args.days_back)

        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Total duration: {duration}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
