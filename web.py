import numpy as np
np.float_ = np.float64          # compat NumPy 2.0
import os
import time
import logging
import requests
import pandas as pd
import torch
from prometheus_client import Counter, Histogram, Gauge, start_http_server

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("bmw_analytics")

# ------------------------------------------------------------------
# Métricas Prometheus
# ------------------------------------------------------------------
PRED_COUNT = Counter("bmw_predictions_total", "Total de predicciones")
PRED_ERROR = Counter("bmw_prediction_errors", "Errores 500")
LATENCY    = Histogram("bmw_prediction_latency_seconds", "Latencia")
RMSE_GAUGE = Gauge("bmw_rmse_last", "RMSE último lote")

# ------------------------------------------------------------------
# Envío GA4 (placeholder: completar api_secret)
# ------------------------------------------------------------------
GA4_ID = os.getenv("GA4_ID", "G-XXXXXXXXXX")
GA4_URL = f"https://www.google-analytics.com/mp/collect?measurement_id={GA4_ID}&api_secret=YOUR_SECRET"

def send_ga4(event_name, price, rmse):
    payload = {
        "client_id": "bmw_backend",
        "events": [{"name": event_name, "params": {"price": float(price), "rmse": float(rmse)}}]
    }
    try:
        requests.post(GA4_URL, json=payload, timeout=2)
    except Exception as e:
        logger.debug("GA4 error: %s", e)

# ------------------------------------------------------------------
# RMSE simple entre dos dataframes (robusto)
# ------------------------------------------------------------------
def compute_metrics(reference: pd.DataFrame, current: pd.DataFrame, target_col: str = "Price_USD"):
    if target_col not in reference.columns or target_col not in current.columns:
        logger.warning("compute_metrics: columna target '%s' no encontrada", target_col)
        return float("nan")
    n = min(len(reference), len(current))
    if n == 0:
        logger.warning("compute_metrics: dataframes vacíos")
        return float("nan")
    ref = reference[target_col].iloc[:n].astype(float)
    cur = current[target_col].iloc[:n].astype(float)
    rmse = float(((ref - cur) ** 2).mean() ** 0.5)
    RMSE_GAUGE.set(rmse)
    logger.info("RMSE calculado: %.2f", rmse)
    return rmse

# ------------------------------------------------------------------
# Wrapper de predicción con métricas y alertas
# ------------------------------------------------------------------
THRESHOLD_RMSE = 5000.0  # USD

def predict_with_analytics(model, preprocessor, reference_df, x_raw, y_true):
    start = time.time()
    try:
        # preparar input
        x_input = pd.DataFrame([x_raw]) if isinstance(x_raw, dict) else x_raw
        x_proc = preprocessor.transform(x_input)
        # manejar sparse matrices
        if hasattr(x_proc, "toarray"):
            arr = x_proc.toarray()
        else:
            arr = np.asarray(x_proc)
        x_tensor = torch.tensor(arr, dtype=torch.float32)
        if x_tensor.ndim == 1:
            x_tensor = x_tensor.unsqueeze(0)

        with torch.no_grad():
            out = model(x_tensor)
            # convertir a numpy/float de forma segura
            if hasattr(out, "detach"):
                out = out.detach().cpu().numpy()
            pred = float(out[0]) if hasattr(out, "__len__") else float(out)

        latency = time.time() - start
        LATENCY.observe(latency)
        PRED_COUNT.inc()

        # RMSE online (1 muestra simplificado)
        try:
            y_val = float(y_true)
            rmse = abs(pred - y_val)
        except Exception:
            rmse = float("nan")
        RMSE_GAUGE.set(rmse)

        # negocio
        send_ga4("predict", pred, rmse)

        if not (rmse != rmse):  # rmse is not NaN
            if rmse > THRESHOLD_RMSE:
                logger.warning("ALERTA: RMSE %.0f supera umbral", rmse)

        return pred, rmse
    except Exception as exc:
        PRED_ERROR.inc()
        logger.exception("Error en predicción")
        raise

# ------------------------------------------------------------------
# Arranque de servidor de métricas
# ------------------------------------------------------------------
if __name__ == "__main__":
    start_http_server(8000)
    logger.info("Servidor de métricas Prometheus en :8000/metrics")
    while True:
        time.sleep(3600)