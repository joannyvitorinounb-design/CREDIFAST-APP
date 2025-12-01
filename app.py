
# ============================================
# APP STREAMLIT - VERSÃO ULTRA-RÁPIDA
# Cumpre requisitos do bônus com desempenho otimizado
# ============================================

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors

# SHAP (explicabilidade)
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

# -----------------------------
# Funções utilitárias
# -----------------------------
@st.cache_data
def load_data(uploaded_file=None, local_path="credit_risk_dataset.csv"):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    return None

def split_and_preprocess(df, target_col="loan_status"):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]
    cat_cols = [c for c in df.columns if c not in num_cols + [target_col]]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    return preprocessor, X_train_prep, y_train, X_test_prep, y_test, num_cols, cat_cols

@st.cache_data
def cached_split_and_preprocess(df):
    return split_and_preprocess(df)

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    auc = roc_auc_score(y_test, y_score)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    ax.plot([0,1], [0,1], 'k--')
    ax.set_title(f'ROC - {name}')
    ax.legend()

    return {"name": name, "auc": auc, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}, fig

@st.cache_data
def cached_evaluate(name, model, X_train, y_train, X_test, y_test):
    return evaluate_model(name, model, X_train, y_train, X_test, y_test)

def get_feature_names(preprocessor, num_cols, cat_cols):
    onehot = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_features = list(onehot.get_feature_names_out(cat_cols))
    return num_cols + cat_features

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📥 Upload")
uploaded = st.sidebar.file_uploader("Envie o arquivo CSV", type=["csv"])
df = load_data(uploaded)

if df is None:
    st.warning("Nenhum arquivo encontrado. Envie o CSV.")
    st.stop()

subset_frac = 0.3  # fixo para clusters/outliers
shap_sample_size = 200  # fixo para SHAP

# -----------------------------
# Layout principal
# -----------------------------
st.title("CrediFast • APP Rápido")
tabs = st.tabs(["Dados", "Modelos", "SHAP", "Clusters", "Outliers", "Recomendações"])

# -----------------------------
# Aba: Dados
# -----------------------------
with tabs[0]:
    st.subheader("Amostra dos dados")
    st.dataframe(df.head(20))
    fig, ax = plt.subplots(figsize=(5,3))
    sns.countplot(x="loan_status", data=df, ax=ax)
    ax.set_title("Distribuição da variável alvo")
    st.pyplot(fig)

# -----------------------------
# Pré-processamento
# -----------------------------
preprocessor, X_train, y_train, X_test, y_test, num_cols, cat_cols = cached_split_and_preprocess(df)

# -----------------------------
# Aba: Modelos
# -----------------------------
with tabs[1]:
    st.subheader("Treinar modelos (rápido)")
    start_train = st.button("▶️ Treinar")
    if not start_train:
        st.info("Clique para treinar os modelos.")
        st.stop()

    models = [
        ("DecisionTree", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
        ("RandomForest", RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)),
        ("SVM", SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE))
    ]

    results = []
    cols = st.columns(3)
    i = 0
    for name, model in models:
        res, fig = cached_evaluate(name, model, X_train, y_train, X_test, y_test)
        results.append(res)
        with cols[i % 3]:
            st.pyplot(fig)
        i += 1

    df_res = pd.DataFrame(results).sort_values(by=["auc", "recall"], ascending=[False, False])
    st.dataframe(df_res)

    best_model_name = df_res.iloc[0]["name"]
    st.success(f"Melhor modelo: {best_model_name}")

# -----------------------------
# Aba: SHAP
# -----------------------------
with tabs[2]:
    st.subheader("Explicabilidade (SHAP)")
    if not HAS_SHAP:
        st.warning("SHAP não disponível.")
    else:
        try:
            feature_names = get_feature_names(preprocessor, num_cols, cat_cols)
            explainer = shap.TreeExplainer(models[1][1])  # RandomForest
            sample_idx = np.random.choice(np.arange(X_test.shape[0]), size=min(shap_sample_size, X_test.shape[0]), replace=False)
            X_shap = X_test[sample_idx]
            shap_values = explainer.shap_values(X_shap)
            shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

            fig = plt.figure(figsize=(8,6))
            shap.summary_plot(shap_vals, X_shap, feature_names=feature_names, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erro SHAP: {e}")

# -----------------------------
# Aba: Clusters
# -----------------------------
with tabs[3]:
    st.subheader("Clusters (subset)")
    df_sub = df.sample(frac=subset_frac, random_state=RANDOM_STATE)
    X_scaled = RobustScaler().fit_transform(preprocessor.transform(df_sub.drop(columns=["loan_status"])))
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE).fit(X_pca)
    clusters = kmeans.labels_

    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette="tab10", ax=ax)
    st.pyplot(fig)

# -----------------------------
# Aba: Outliers
# -----------------------------
with tabs[4]:
    st.subheader("Outliers (subset)")
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(X_scaled)
    distances, _ = neigh.kneighbors(X_scaled)
    eps_val = np.percentile(np.sort(distances[:, -1]), 95)
    db = DBSCAN(eps=eps_val, min_samples=5).fit(X_scaled)
    outliers = (db.labels_ == -1).sum()
    st.write(f"Outliers detectados: {outliers}")

# -----------------------------
# Aba: Recomendações
# -----------------------------
with tabs[5]:
    st.write("Clientes com parcela/renda alta e juros elevados devem ter limites reduzidos ou exigência de garantias. "
             "Clusters com maior inadimplência e outliers devem passar por análise reforçada. "
             "Monitorar continuamente métricas e ajustar políticas conforme padrões SHAP.")
