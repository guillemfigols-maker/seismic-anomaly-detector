# seismic-anomaly-detector
End-to-end earthquake data pipeline with dbt, Airflow &amp; MLflow.
## project structure
```
seismic-anomaly-detector/
├── dbt_project/
│   ├── dbt_project.yml
│   ├── models/
│   │   ├── staging/
│   │   │   ├── stg_earthquakes.sql
│   │   │   ├── stg_weather.sql
│   │   │   └── schema.yml
│   │   ├── intermediate/
│   │   │   ├── int_earthquake_sequences.sql
│   │   │   └── int_earthquakes_enriched.sql
│   │   └── marts/
│   │       ├── fct_anomalies.sql
│   │       ├── fct_daily_seismicity.sql
│   │       └── dim_regions.sql
│   └── tests/
│       └── assert_magnitude_range.sql
├── ingestion/
│   ├── fetch_usgs.py
│   ├── fetch_weather.py
│   └── load_to_db.py
├── ml/
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── dags/
│   └── seismic_pipeline.py
├── streamlit_app/
│   └── app.py
├── docker-compose.yml
├── Dockerfile
├── .github/workflows/ci.yml
└── README.md
```
