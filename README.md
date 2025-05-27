# 🌌POLARIS

**Predictive Observation Lab for AI-Resilient Infrastructure & Sustainability**

POLARIS is a cutting-edge AI-powered geospatial intelligence platform that forecasts climate risks, assesses infrastructure resilience, and generates high-resolution vulnerability maps using satellite imagery and machine learning.

Built for researchers, policymakers, and cities navigating the age of climate change, POLARIS fuses Earth observation data, climate models, and advanced neural networks into a powerful real-world impact tool.


## Why POLARIS?
Natural disasters are accelerating. Infrastructure is aging. Communities are vulnerable.

**POLARIS addresses this crisis** by combining:

- **Satellite Imagery Analysis** (Sentinel-2, xView2)
- **Climate Risk Forecasting**
- **Infrastructure Damage Detection**
- **Interactive Geospatial Dashboards**
- **AI Models for Image Segmentation & Prediction**

## Features

| Capability | Description |
|-----------|-------------|
| **Disaster Damage Detection** | Uses Vision Transformers (ViT) and U-Net models to segment satellite images and identify flood, fire, and damage zones. |
| **Infrastructure Risk Modeling** | Employs Graph Neural Networks (GNNs) to analyze critical infrastructure vulnerabilities. |
| **Climate Forecasting Engine** | Predicts long-term risk based on historical climate and MODIS satellite data. |
| **Equity-Aware Insights** | Overlays socioeconomic data to assess which communities are most at risk. |
| **Interactive Map Dashboard** | Real-time frontend built in React + Mapbox for exploring live and historical risk layers. |
| **FastAPI Backend** | RESTful API serving predictions, maps, and geospatial intelligence to third-party systems. |


## Tech Stack

| Layer        | Stack                                                  |
|-------------|--------------------------------------------------------|
| **Frontend** | React, Mapbox GL, Tailwind CSS                        |
| **Backend**  | FastAPI, Python, PostgreSQL + PostGIS                |
| **ML Models**| PyTorch, Hugging Face Transformers, TensorFlow        |
| **GIS Tools**| Rasterio, GDAL, GeoPandas, shapely                    |
| **Deployment** | Docker, Heroku / AWS / Render                        |


## Project Structure

