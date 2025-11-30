
# ============================================
# APP STREAMLIT - BÔNUS DE INOVAÇÃO (CrediFast)
# Classificação de risco, SHAP, Clusters (KMeans + PCA),
# Outliers (DBSCAN), Upload e filtros
# Versão robusta para Streamlit Cloud (imports seguros)
# ============================================

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

# ---- Imports de pacotes com tratamento de falha ----
HAS_SMOTE = True
try:
    from imbalanced_learn.over_sampling import SMOTE
except Exception:
    HAS_SMOTE = False

HAS_XGB = True
try:
    import xgboost as xgb
except Exception:
    HAS_XGB = False

HAS_LGBM = True
try:
    import lightgbm as lgb
except Exception:
    HAS_LGBM = False

HAS_SHAP = True
try:
    import shap
except Exception:
    HAS_SHAP = False

import joblib

# -----------------------------
# Configurações gerais
# -----------------------------
st.set_page_config(page_title="CrediFast - Risco de Crédito", layout="wide")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_context("talk")

# Avisos de ambiente
if not HAS_SMOTE:
    st.warning("SMOTE indisponível no ambiente. O treino seguirá SEM balanceamento. "
               "Verifique 'imbalanced-learn' no requirements.txt.")
if not HAS_XGB:
    st.warning("XGBoost indisponível. O app seguirá sem XGBoost.")
if not HAS_LGBM:
    st.warning("LightGBM indisponível. O app seguirá sem LightGBM.")
if not HAS_SHAP:
    st.warning("SHAP indisponível. A aba de explicabilidade ficará limitada.")

# -----------------------------
# Funções utilitárias
# -----------------------------
@st.cache_data
def load_data(uploaded_file=None, local_path="credit_risk_dataset.csv"):
    """
    Carrega o dataset:
    - Se houver arquivo enviado (upload), usa esse.
    - Senão, tenta ler o arquivo local credit_risk_dataset.csv.
    """
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            return df
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
            return df
        return None
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

def split_and_preprocess(df, target_col="loan_status"):
    """
    Separa treino/teste (estratificado) e cria o pré-processador:
    - Numéricas: imputação mediana + StandardScaler
    - Categóricas: imputação mais frequente + OneHotEncoder(handle_unknown='ignore')
    Aplica SMOTE no treino transformado (se disponível).
    """
    # Detectar colunas
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
