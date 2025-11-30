
# CrediFast – APP de Risco de Crédito (Streamlit)

Este APP entrega:
- Classificação de risco (vários modelos, métricas, curva ROC)
- Explicabilidade com SHAP (summary plot + explicação local)
- Clusterização com KMeans (visualização via PCA)
- Detecção de outliers com DBSCAN
- Upload de `credit_risk_dataset.csv` e filtros interativos

## Como usar (local)
1. Crie uma pasta com:
   - `app.py`
   - `requirements.txt`
   - (opcional) `credit_risk_dataset.csv` na raiz
2. Instale as dependências (em um venv):
   ```bash
   pip install -r requirements.txt
