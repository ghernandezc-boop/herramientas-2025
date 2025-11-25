# analytics.py  (Evidently 0.4.x  –  nombres nuevos)
# analytics.py
import numpy as np
np.float_ = np.float64          # ← compat NumPy 2.0
# analytics.py  (Evidently 0.4.x  –  nombres nuevos)
import os
import time
import logging
import requests
import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# --------- nuevos nombres de métricas ----------
from evidently.metrics import RegressionRMSEMetric, RegressionMAEMetric, RegressionR2ScoreMetric
from evidently.metric_preset import MetricPreset
from evidently.metrics.base_report import Report
from evidently import ColumnMapping

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("bmw_analytics")

# 1. Métricas Prometheus
PRED_COUNT = Counter("bmw_predictions_total", "Total de predicciones")
PRED_ERROR = Counter("bmw_prediction_errors", "Errores 500")
LATENCY    = Histogram("bmw_prediction_latency_seconds", "Latencia")
RMSE_GAUGE = Gauge("bmw_rmse_last", "RMSE último lote")

# 2. GA4
GA4_ID  = os.getenv("GA4_ID", "G-XXXXXXXXXX")
GA4_URL = f"https://www.google-analytics.com/mp/collect?measurement_id={GA4_ID}&api_secret=YOUR_SECRET"

def send_ga4(event_name: str, price: float, rmse: float):
    payload = {"client_id": "bmw_backend", "events": [{"name": event_name, "params": {"price": float(price), "rmse": float(rmse)}}]}
    try:
        requests.post(GA4_URL, json=payload, timeout=2)
    except Exception as e:
        logger.warning("GA4 error: %s", e)

# 3. Column mapping
column_mapping = ColumnMapping(
    target="Price_USD",
    prediction="pred_price",
    numerical_features=["Year", "Engine_Size_L", "Mileage_KM"]
)

# 4. Cálculo de métricas con Evidently 0.4
def compute_metrics(reference: pd.DataFrame, current: pd.DataFrame):
    report = Report(metrics=[
        RegressionRMSEMetric(),
        RegressionMAEMetric(),
        RegressionR2ScoreMetric()
    ])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
    result = report.as_dict()
    rmse = result["metrics"][0]["result"]["current"]["value"]
    RMSE_GAUGE.set(rmse)
    logger.info("RMSE calculado: %.2f", rmse)
    return rmse

# 5. Wrapper de predicción
THRESHOLD_RMSE = 5000

def predict_with_analytics(model, preprocessor, reference_df, x_raw, y_true):
    import torch
    start = time.time()
    try:
        x_proc = preprocessor.transform(pd.DataFrame([x_raw]) if isinstance(x_raw, dict) else x_raw)
        x_tensor = torch.tensor(x_proc.toarray(), dtype=torch.float32)
        with torch.no_grad():
            pred = model(x_tensor).item()

        latency = time.time() - start
        LATENCY.observe(latency)
        PRED_COUNT.inc()

        rmse = abs(pred - y_true)
        RMSE_GAUGE.set(rmse)
        send_ga4("predict", pred, rmse)

        if rmse > THRESHOLD_RMSE:
            logger.warning("ALERTA: RMSE %.0f supera umbral", rmse)

        return pred, rmse

    except Exception as exc:
        PRED_ERROR.inc()
        logger.exception("Error en predicción")
        raise exc

# 6. Servidor Prometheus
if __name__ == "__main__":
    start_http_server(8000)
    logger.info("Servidor de métricas Prometheus en :8000/metrics")
    while True:
        time.sleep(3600)