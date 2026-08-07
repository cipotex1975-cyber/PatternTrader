from app.ml.models import (  # noqa: F401  (registra modelos)
    autoencoder,
    catboost_model,
    cnn_model,
    isolation_forest,
    lightgbm_model,
    lstm_model,
    random_forest,
    transformer_model,
    xgboost_model,
)
from app.ml.models.autoencoder import AutoEncoderModel
from app.ml.models.catboost_model import CatBoostModel
from app.ml.models.cnn_model import CNNModel
from app.ml.models.isolation_forest import IsolationForestModel
from app.ml.models.lightgbm_model import LightGBMModel
from app.ml.models.lstm_model import LSTMModel
from app.ml.models.random_forest import RandomForestModel
from app.ml.models.transformer_model import TransformerModel
from app.ml.models.xgboost_model import XGBoostModel

__all__ = [
    "AutoEncoderModel",
    "CatBoostModel",
    "CNNModel",
    "IsolationForestModel",
    "LightGBMModel",
    "LSTMModel",
    "RandomForestModel",
    "TransformerModel",
    "XGBoostModel",
]
