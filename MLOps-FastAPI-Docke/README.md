# Boston Housing Regression — Deployment

## 0. Train the Model (one-time, before building the image)
```bash
python app/train.py
```
This fits the champion model (tuned XGBoost) on the Boston Housing dataset and
saves `model/model.joblib`, `model/feature_order.json`, and `model/metrics.json`.

---
# Docker Setup
## 1. Build the Docker Image
```bash
docker build -t <your-dockerhub-username>/ml_101_regression:latest .
```
---
## 2. Run the Docker Container
```bash
docker run -p 8000:8000 <your-dockerhub-username>/ml_101_regression:latest
```
The API will now be available at
```
http://localhost:8000
```
---
## 3. Push the Image to Docker Hub
Login to Docker Hub
```bash
docker login
```
Push the image
```bash
docker push <your-dockerhub-username>/ml_101_regression:latest
```
---
## 4. Pull the Image
Anyone can pull the image using
```bash
docker pull <your-dockerhub-username>/ml_101_regression:latest
```
---
## 5. Run the Pulled Image
```bash
docker run -p 8000:8000 <your-dockerhub-username>/ml_101_regression:latest
```
---
# API Endpoints
## Home
**GET**
```
/
```
Response
```json
{
    "message": "Boston Housing Price Prediction API",
    "expected_features": ["crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat"],
    "docs": "/docs"
}
```
---
## Health Check
**GET**
```
/health
```
Response
```json
{
    "status": "ok"
}
```
---
## Prediction
**POST**
```
/predict
```
Request
```json
{
    "features": [0.00632, 18, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9, 4.98]
}
```
Example Response
```json
{
    "predicted_value": 24.3,
    "unit": "$1000s"
}
```
---
# Docker Hub Repository
```
https://hub.docker.com/r/<your-dockerhub-username>/ml_101_regression
```
---
