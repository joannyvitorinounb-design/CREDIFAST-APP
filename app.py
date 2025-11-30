
# ============================================
# APP STREAMLIT - BÔNUS DE INOVAÇÃO (CrediFast)
# Classificação de risco, SHAP, Clusters (KMeans + PCA),
# Outliers (DBSCAN), Upload e filtros
# ============================================

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

from imbalanced_learn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb
import shap
import joblib
import os

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
    """
    Carrega o dataset:
    - Se houver arquivo enviado (upload), usa esse.
    - Senão, tenta ler o arquivo local credit_risk_dataset.csv.
    - Retorna df ou None.
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
    Separa treino/teste com estratificação e cria o pré-processador:
    - Numéricas: imputação mediana + StandardScaler
    - Categóricas: imputação mais frequente + OneHotEncoder(handle_unknown='ignore')
    Aplica SMOTE no treino transformado.
    Retorna: (preprocessor, X_train_bal, y_train_bal, X_test_prep, y_test, num_cols, cat_cols)
    """
    # Detectar colunas
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]
    cat_cols = [c for c in df.columns if c not in num_cols + [target_col]]

    # Pipelines
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
        ],
        remainder="drop"
    )

    # Split estratificado
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # Transformar + SMOTE no treino
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep  = preprocessor.transform(X_test)

    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_prep, y_train)

    return preprocessor, X_train_bal, y_train_bal, X_test_prep, y_test, num_cols, cat_cols

def get_models():
    """
    Define o conjunto de modelos a comparar (exigidos no enunciado).
    """
    models = [
        ("KNN", KNeighborsClassifier(n_neighbors=7)),
        ("SVM", SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)),
        ("DecisionTree", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
        ("RandomForest", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)),
        ("AdaBoost", AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ("XGBoost", xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)),
        ("LightGBM", lgb.LGBMClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)),
        ("MLP", MLPClassifier(hidden_layer_sizes=(100,50), max_iter=400, random_state=RANDOM_STATE)),
    ]
    return models

def evaluate_model(name, model, X_train, y_train, X_test, y_test, plot_roc=True):
    """
    Treina e avalia um modelo. Retorna dicionário com métricas e o objeto treinado.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        try:
            y_score = model.decision_function(X_test)
            y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-8)
        except Exception:
            y_score = y_pred

    auc = roc_auc_score(y_test, y_score)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    res = {"name": name, "model": model, "auc": auc, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm}
    fig = None
    if plot_roc:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
        ax.plot([0,1], [0,1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate (Recall)')
        ax.set_title(f'ROC - {name}')
        ax.legend()

    return res, fig

def get_feature_names_from_preprocessor(preprocessor, num_cols, cat_cols):
    """
    Reconstrói nomes das features após OneHotEncoder.
    """
    onehot = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = list(onehot.get_feature_names_out(cat_cols))
    return num_cols + cat_feature_names

# -----------------------------
# Lado esquerdo (upload e filtros)
# -----------------------------
st.sidebar.title("📥 Dados e Filtros")
uploaded = st.sidebar.file_uploader("Envie o arquivo CSV (credit_risk_dataset.csv)", type=["csv"])
df = load_data(uploaded)

if df is None:
    st.warning("Nenhum arquivo foi enviado e o arquivo local 'credit_risk_dataset.csv' não foi encontrado. Envie um CSV com as colunas do dataset de crédito.")
    st.stop()

# Mostrar amostra e permitir filtro rápido por colunas principais (se existirem)
st.sidebar.markdown("### Filtros rápidos")
col_options = ["loan_percent_income", "loan_int_rate", "person_income", "loan_amnt"]
filters = {}
for col in col_options:
    if col in df.columns and df[col].dtype != "object":
        min_v = float(df[col].min())
        max_v = float(df[col].max())
        filters[col] = st.sidebar.slider(f"{col}", min_value=min_v, max_value=max_v, value=(min_v, max_v))

# Aplicar filtros (apenas para exibição na aba de Dados)
df_filtered = df.copy()
for col, (a, b) in filters.items():
    df_filtered = df_filtered[(df_filtered[col] >= a) & (df_filtered[col] <= b)]

# -----------------------------
# Layout principal: abas
# -----------------------------
st.title("CrediFast • Sistema de Apoio à Decisão de Risco de Crédito")
tabs = st.tabs(["Dados", "Modelos e Métricas", "Explicabilidade (SHAP)", "Clusters (KMeans + PCA)", "Outliers (DBSCAN)", "Recomendações"])

# -----------------------------
# Aba: Dados
# -----------------------------
with tabs[0]:
    st.subheader("Visão geral dos dados")
    st.write("Abaixo uma amostra dos dados (após filtros da barra lateral):")
    st.dataframe(df_filtered.head(20))

    # Distribuição da variável alvo
    if "loan_status" in df.columns:
        st.subheader("Distribuição da variável-alvo (loan_status)")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(x="loan_status", data=df, ax=ax)
        ax.set_title("Distribuição da variável target (loan_status)")
        ax.set_xticklabels(["Good (0)", "Bad (1)"])
        st.pyplot(fig)

        # Correlações numéricas com a target
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "loan_status" in num_cols:
            num_cols = [c for c in num_cols if c != "loan_status"]
        corrs = df[num_cols + ["loan_status"]].corr()["loan_status"].drop("loan_status").abs().sort_values(ascending=False)
        st.subheader("Top correlações absolutas com loan_status (numéricas)")
        st.write(corrs.head(10))

# -----------------------------
# Pré-processar e dividir dados
# -----------------------------
try:
    preprocessor, X_train_bal, y_train_bal, X_test_prep, y_test, num_cols, cat_cols = split_and_preprocess(df, target_col="loan_status")
except Exception as e:
    st.error(f"Erro no pré-processamento: {e}")
    st.stop()

# -----------------------------
# Aba: Modelos e Métricas
# -----------------------------
with tabs[1]:
    st.subheader("Treinamento e avaliação dos modelos (AUC, ROC, Accuracy, Precision, Recall, F1, Confusão)")
    models = get_models()
    results = []

    # Treinar e avaliar
    roc_cols = st.columns(3)
    i_plot = 0
    for name, model in models:
        res, fig = evaluate_model(name, model, X_train_bal, y_train_bal, X_test_prep, y_test, plot_roc=True)
        results.append(res)
        if fig is not None:
            with roc_cols[i_plot % 3]:
                st.pyplot(fig)
            i_plot += 1

    # Tabela resumida
    results_df = pd.DataFrame([{
        "name": r["name"], "auc": r["auc"], "accuracy": r["accuracy"], "precision": r["precision"], "recall": r["recall"], "f1": r["f1"]
    } for r in results]).sort_values(by=["auc", "recall"], ascending=[False, False]).reset_index(drop=True)
    st.write("**Resumo (ordenado por AUC e desempate por Recall):**")
    st.dataframe(results_df, use_container_width=True)

    # Melhor modelo
    best_row = results_df.iloc[0]
    best_name = best_row["name"]
    best_model = [r["model"] for r in results if r["name"] == best_name][0]
    st.success(f"🟢 Modelo vencedor: **{best_name}** (AUC={best_row['auc']:.3f} | Recall={best_row['recall']:.3f})")

    # Botões para salvar artefatos
    c1, c2 = st.columns(2)
    if c1.button("Salvar pré-processador e melhor modelo (joblib)"):
        joblib.dump(preprocessor, "preprocessor.joblib")
        joblib.dump(best_model, f"best_model_{best_name}.joblib")
        st.info("Arquivos salvos: preprocessor.joblib e best_model_*.joblib")

# -----------------------------
# Aba: Explicabilidade (SHAP)
# -----------------------------
with tabs[2]:
    st.subheader("Explicabilidade global e local com SHAP (modelo vencedor)")

    try:
        feature_names = get_feature_names_from_preprocessor(preprocessor, num_cols, cat_cols)
        # TreeExplainer para modelos de árvore/boosting
        explainer = shap.TreeExplainer(best_model)

        # Sample para o summary plot
        sample_idx = np.random.choice(np.arange(X_test_prep.shape[0]), size=min(1000, X_test_prep.shape[0]), replace=False)
        X_shap = X_test_prep[sample_idx]
        shap_values = explainer.shap_values(X_shap)
        shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

        st.write("**Summary plot (impacto global das variáveis):**")
        fig = plt.figure(figsize=(9,7))
        shap.summary_plot(shap_vals, X_shap, feature_names=feature_names, show=False)
        st.pyplot(fig)

        # Local: escolher um índice qualquer de bad e um de good
        st.write("**Waterfalls (explicação local):**")
        # Encontrar índices em y_test
        y_test_arr = y_test.values
        idx_bad = np.where(y_test_arr == 1)[0]
        idx_good = np.where(y_test_arr == 0)[0]
        if len(idx_bad) == 0 or len(idx_good) == 0:
            st.warning("Não foi possível encontrar exemplos de ambas as classes no conjunto de teste.")
        else:
            i_bad = int(idx_bad[0])
            i_good = int(idx_good[0])

            # Recalcular shap para 2 pontos
            X_local = X_test_prep[[i_bad, i_good]]
            shap_values_local = explainer.shap_values(X_local)
            shap_vals_local = shap_values_local[1] if isinstance(shap_values_local, list) else shap_values_local
            base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

            st.write(f"Exemplo bad (índice teste={i_bad})")
            fig1 = shap.plots.waterfall(shap.Explanation(values=shap_vals_local[0],
                                                         base_values=base_val,
                                                         data=X_local[0],
                                                         feature_names=feature_names), show=False)
            st.pyplot(fig1)

            st.write(f"Exemplo good (índice teste={i_good})")
            fig2 = shap.plots.waterfall(shap.Explanation(values=shap_vals_local[1],
                                                         base_values=base_val,
                                                         data=X_local[1],
                                                         feature_names=feature_names), show=False)
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Erro ao gerar gráficos SHAP: {e}")
        st.info("Dica: o SHAP funciona melhor com modelos de árvore/boosting (RF/XGB/LGBM/GB). Se selecionou outro, considere trocar o modelo vencedor.")

# -----------------------------
# Aba: Clusters (KMeans + PCA)
# -----------------------------
with tabs[3]:
    st.subheader("Clusterização com KMeans (visualização via PCA)")

    try:
        # Preparar todo o dataset (sem a target) com o mesmo preprocessor
        X_all = df.drop(columns=["loan_status"])
        X_prep_all = preprocessor.transform(X_all)

        scaler_for_cluster = RobustScaler()
        X_scaled = scaler_for_cluster.fit_transform(X_prep_all)

        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_scaled)

        # Escolha de K por silhouette
        from sklearn.metrics import silhouette_score
        k_list = list(range(2, 7))
        scores = []
        for k in k_list:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels_k = km.fit_predict(X_pca)
            sc = silhouette_score(X_pca, labels_k)
            scores.append(sc)

        best_k = k_list[int(np.argmax(scores))]
        st.write(f"Melhor K por silhouette: **{best_k}** (score={max(scores):.3f})")

        kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
        clusters = kmeans.fit_predict(X_pca)

        # Scatter PCA com clusters
        fig, ax = plt.subplots(figsize=(7,5))
        sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette="tab10", ax=ax, s=20, legend="full")
        ax.set_title(f"KMeans (k={best_k}) sobre PCA(2)")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        st.pyplot(fig)

        # Bad rate por cluster
        clusters_series = pd.Series(clusters, index=df.index, name="cluster")
        cluster_risk = pd.concat([df["loan_status"], clusters_series], axis=1).groupby("cluster")["loan_status"].agg(["count", "sum"])
        cluster_risk["bad_rate"] = cluster_risk["sum"] / cluster_risk["count"]
        st.write("**Bad rate por cluster:**")
        st.dataframe(cluster_risk.sort_values("bad_rate", ascending=False))
    except Exception as e:
        st.error(f"Erro ao calcular clusters: {e}")

# -----------------------------
# Aba: Outliers (DBSCAN)
# -----------------------------
with tabs[4]:
    st.subheader("Detecção de outliers com DBSCAN")

    try:
        # Usar o mesmo X_scaled (recalcular aqui para garantir independência da aba)
        X_all = df.drop(columns=["loan_status"])
        X_prep_all = preprocessor.transform(X_all)
        scaler_for_cluster = RobustScaler()
        X_scaled = scaler_for_cluster.fit_transform(X_prep_all)

        # k-distance (k=5) para escolher eps (percentil 95)
        neigh = NearestNeighbors(n_neighbors=5)
        nbrs = neigh.fit(X_scaled)
        distances, _ = nbrs.kneighbors(X_scaled)
        kdist = np.sort(distances[:, -1])

        fig, ax = plt.subplots(figsize=(7,3))
        ax.plot(kdist)
        ax.set_title("k-distance (k=5) - escolha visual de eps")
        ax.set_ylabel("5-NN distance"); ax.set_xlabel("Amostras (ordenadas)")
        st.pyplot(fig)

        eps_val = float(np.percentile(kdist, 95))
        st.write(f"eps escolhido (percentil 95): **{eps_val:.4f}**")

        db = DBSCAN(eps=eps_val, min_samples=5)
        db_labels = db.fit_predict(X_scaled)

        outlier_mask = (db_labels == -1)
        df_out = df.copy()
        df_out["is_outlier"] = outlier_mask.astype(int)

        summary_out = df_out.groupby("is_outlier")["loan_status"].agg(["count", "sum"])
        summary_out["bad_rate"] = summary_out["sum"] / summary_out["count"]

        st.write("**Resumo de risco (inliers vs outliers):**")
        st.dataframe(summary_out)
    except Exception as e:
        st.error(f"Erro no DBSCAN: {e}")

# -----------------------------
# Aba: Recomendações
# -----------------------------
with tabs[5]:
    st.subheader("Recomendações gerenciais baseadas nos resultados")
    st.write(
        "Com base nos resultados dos modelos e na explicabilidade com SHAP, recomenda-se implantar o modelo vencedor "
        "com limiar de decisão ajustado para favorecer recall, reduzindo falsos negativos que geram perdas diretas. "
        "A análise global mostra que a proporção da parcela na renda (loan_percent_income) e a taxa de juros (loan_int_rate) "
        "são os fatores que mais elevam o risco quando estão altos; assim, clientes com parcela/renda elevada e juros altos "
        "devem receber limites mais conservadores, exigência de entrada maior ou comprovações adicionais antes da aprovação. "
        "Em contrapartida, renda mais alta e histórico de crédito mais longo tendem a reduzir o risco, sinalizando aprovação "
        "com condições padrão. A segmentação por clusters evidencia grupos com taxas de inadimplência diferentes; clusters com "
        "bad_rate superior devem ser submetidos a triagens reforçadas e limites iniciais mais baixos, enquanto clusters de menor "
        "risco podem receber condições mais competitivas. Já a detecção de outliers com DBSCAN mostrou risco significativamente "
        "mais alto entre casos fora do padrão, justificando esteira específica de verificação (documentos, referências), "
        "eventual solicitação de garantias e acompanhamento proativo nos primeiros ciclos de pagamento. Por fim, recomenda-se "
        "monitorar continuamente o desempenho (AUC/Recall/Precision), reavaliar o limiar de decisão periodicamente e ajustar "
        "as políticas conforme os padrões observados no SHAP e nas taxas por segmento."
    )
