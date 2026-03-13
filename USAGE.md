# Seismic Anomaly Detector - Usage Guide

This guide explains how to use the PySpark-based seismic data extraction and aftershock prediction system.

## Overview

The pipeline consists of four main stages:

1. **Data Extraction**: Fetch earthquake data from USGS API using PySpark
2. **Feature Engineering**: Generate features for machine learning from raw data
3. **Model Training**: Train aftershock probability prediction models with MLflow tracking
4. **Predictions**: Generate aftershock probability predictions for recent events

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Full Pipeline

```bash
# Run all stages: extract, features, train, predict
python run_pipeline.py --full
```

### Run Individual Stages

```bash
# 1. Extract earthquake data from USGS
python run_pipeline.py --extract --start-date 2024-01-01 --end-date 2025-01-01

# 2. Generate features for ML
python run_pipeline.py --features

# 3. Train models
python run_pipeline.py --train

# 4. Generate predictions for recent events (last 7 days)
python run_pipeline.py --predict --days-back 7
```

## Detailed Usage

### 1. Data Extraction (`ingestion/fetch_usgs.py`)

Fetches earthquake data from USGS API and saves as partitioned Parquet files.

**Features:**
- Chunks requests into 30-day periods to avoid API limits
- Processes GeoJSON responses into structured PySpark DataFrames
- Saves data partitioned by year/month for efficient querying
- Extracts 25+ earthquake attributes including magnitude, location, depth, timing

**Usage:**

```python
from ingestion.fetch_usgs import USGSDataFetcher

fetcher = USGSDataFetcher(output_path="data/raw/earthquakes")
fetcher.run_pipeline(
    start_date="2024-01-01",
    end_date="2025-01-01",
    chunk_days=30
)
fetcher.stop()
```

**Output:** Partitioned Parquet files in `data/raw/earthquakes/`

### 2. Feature Engineering (`ml/features.py`)

Generates features for aftershock prediction using temporal and spatial analysis.

**Features Generated:**
- **Temporal**: Hour, day of week, day of year, time since previous event
- **Spatial**: Grid cell assignment, spatial density
- **Seismic**:
  - Rolling 30-day statistics (mean/max/std magnitude, event count)
  - Energy release calculations
  - Cumulative energy over time windows
- **Labels**: Binary aftershock indicator (1 if significant earthquake follows within 30 days)

**Usage:**

```python
from ml.features import SeismicFeatureEngine

engine = SeismicFeatureEngine()
engine.run_feature_pipeline(
    input_path="data/raw/earthquakes",
    output_path="data/processed/features"
)
engine.stop()
```

**Output:** Parquet files in `data/processed/features/`

### 3. Model Training (`ml/train.py`)

Trains multiple classification models with MLflow experiment tracking.

**Models Trained:**
- Random Forest Classifier
- Gradient Boosted Trees (GBT)
- Logistic Regression

**Features:**
- Time-based train/test split (prevents data leakage)
- Automated feature scaling
- Comprehensive metrics: AUC-ROC, AUC-PR, F1, precision, recall
- MLflow tracking of all experiments
- Saves best model based on AUC-ROC

**Usage:**

```python
from ml.train import AftershockModelTrainer

trainer = AftershockModelTrainer()
results = trainer.run_training_pipeline(
    features_path="data/processed/features",
    model_output_path="data/models/best_model"
)
trainer.stop()
```

**Output:**
- Trained model saved to `data/models/best_model/`
- MLflow experiment logs in `mlruns/`

### 4. Predictions (`ml/predict.py`)

Generates aftershock probability predictions for recent earthquakes.

**Features:**
- Loads recent earthquake events (configurable time window)
- Applies same feature engineering pipeline
- Generates probability scores and risk levels (HIGH/MEDIUM/LOW)
- Produces detailed prediction reports
- Can predict for single events or batches

**Usage:**

```python
from ml.predict import AftershockPredictor

predictor = AftershockPredictor(model_path="data/models/best_model")

# Batch predictions
top_risk = predictor.run_prediction_pipeline(
    earthquake_data_path="data/raw/earthquakes",
    output_path="data/predictions/latest",
    days_back=7
)

# Single event prediction
result = predictor.predict_single_event(
    magnitude=5.5,
    depth_km=10.0,
    latitude=34.05,
    longitude=-118.25,
    recent_magnitude_mean=3.2,
    recent_event_count=45
)
print(f"Aftershock probability: {result['aftershock_probability']:.3f}")
print(f"Risk level: {result['risk_level']}")

predictor.stop()
```

**Output:**
- Predictions saved to `data/predictions/latest/`
- Console report showing top risk events

## Data Directory Structure

After running the pipeline, your data directory will look like:

```
data/
├── raw/
│   └── earthquakes/          # Raw USGS data (partitioned by year/month)
│       ├── year=2024/
│       │   ├── month=1/
│       │   ├── month=2/
│       │   └── ...
│       └── ...
├── processed/
│   └── features/             # Engineered features for ML
├── models/
│   └── best_model/           # Trained PySpark pipeline model
└── predictions/
    └── latest/               # Latest prediction results
```

## Understanding the Predictions

### Risk Levels

- **HIGH** (≥70% probability): Significant aftershock very likely
- **MEDIUM** (40-70% probability): Moderate aftershock risk
- **LOW** (<40% probability): Low aftershock risk

### Prediction Output

Each prediction includes:
- `event_id`: Earthquake event identifier
- `event_time`: When the earthquake occurred
- `magnitude`: Earthquake magnitude
- `depth_km`: Depth in kilometers
- `latitude`, `longitude`: Location
- `place`: Text description of location
- `aftershock_probability`: Probability score (0-1)
- `risk_level`: HIGH/MEDIUM/LOW category
- `prediction`: Binary prediction (1=aftershock expected, 0=no aftershock)

## MLflow Tracking

View experiment results:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser to explore:
- Model parameters
- Performance metrics across runs
- Trained models
- Comparison between model types

## Performance Considerations

### PySpark Configuration

For larger datasets, adjust Spark configuration:

```python
spark = SparkSession.builder \
    .appName("SeismicAnalysis") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()
```

### Data Volume Estimates

- 1 year of global earthquakes: ~50,000-100,000 events
- Raw data size: ~50-100 MB per year
- Feature data size: ~100-200 MB per year
- Model size: ~10-50 MB

## Customization

### Modify Prediction Window

Edit the `time_window_days` parameter in feature engineering:

```python
df = engine.create_aftershock_labels(df, threshold_magnitude=4.0, time_window_days=30)
```

### Change Magnitude Threshold

Adjust what counts as a "significant" aftershock:

```python
# Only consider magnitude 5.0+ as significant aftershocks
df = engine.create_aftershock_labels(df, threshold_magnitude=5.0, time_window_days=30)
```

### Adjust Spatial Grid Size

Modify grid cell size for spatial features:

```python
# Larger cells (1.0 degree instead of 0.5)
df = df.withColumn("lat_grid", F.floor(F.col("latitude") / 1.0)) \
      .withColumn("lon_grid", F.floor(F.col("longitude") / 1.0))
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Increase Spark executor memory
2. **API Rate Limiting**: Increase chunk_days or add delays between requests
3. **Empty Predictions**: Check if recent earthquake data exists
4. **Model Not Found**: Run training stage before prediction

### Logging

All modules use Python logging. Adjust level in code:

```python
logging.basicConfig(level=logging.DEBUG)  # More verbose
logging.basicConfig(level=logging.WARNING)  # Less verbose
```

## Next Steps

After running predictions:

1. **Integrate with dbt**: Use predictions in dbt models for further analysis
2. **Airflow DAG**: Schedule pipeline runs with the dag in `dags/seismic_pipeline.py`
3. **Streamlit Dashboard**: Visualize predictions in `streamlit_app/app.py`
4. **API Endpoint**: Expose predictions via REST API for real-time queries

## References

- [USGS Earthquake API Documentation](https://earthquake.usgs.gov/fdsnws/event/1/)
- [PySpark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
