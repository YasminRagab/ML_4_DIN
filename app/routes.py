from flask import Blueprint, request, render_template
import pandas as pd
import os
import sys
import joblib

# Add the app directory to the system path
# This ensures that the app/functions module can be found and imported by Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.functions.data_processing import load_json_to_df, preprocess_data
from app.functions.model_utils import predict_lgbm, predict_catboost, evaluate_model, retrain_models
from app.functions.plot_utils import plot_confusion_matrix

main = Blueprint('main', __name__)

# Load models
lgbm_model_path = 'app/models/Lgbm_model.joblib'
catboost_model_path = 'app/models/catboost_model.joblib'
preprocessing_path = 'app/models/enc_data.joblib'

lgbm_model = joblib.load(lgbm_model_path)
catboost_model = joblib.load(catboost_model_path)
preprocessing_pipeline = joblib.load(preprocessing_path)

@main.route('/', methods=['GET', 'POST'])
def index():
    lgbm_f1, lgbm_conf_matrix, catboost_f1, catboost_conf_matrix = None, None, None, None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)

            df = load_json_to_df(filepath)
            df.dropna(subset=['Kostengruppe'], inplace=True)
            X_test_encoded, X_test, feature_types, df_processed = preprocess_data(df, preprocessing_pipeline)

            try:
                X_test_dense = X_test_encoded.toarray() if hasattr(X_test_encoded, 'toarray') else X_test_encoded
                df_processed_lgbm = df_processed.copy()
                lgbm_predictions = lgbm_model.predict(X_test_dense)
                lgbm_f1, lgbm_conf_matrix = evaluate_model(df_processed_lgbm, lgbm_predictions, 'LGBM')
                df_processed_lgbm['Kostengruppe'] = df_processed_lgbm['Kostengruppe'].astype(str)
                labels = sorted(pd.unique(df_processed_lgbm['Kostengruppe'].values))
                lgbm_plot_path = 'lgbm_conf_matrix.png'
                plot_confusion_matrix(lgbm_conf_matrix, labels, "Confusion Matrix - LGBM", lgbm_plot_path)
            except Exception as e:
                print(f"Error processing LGBM model: {e}")

            try:
                df_processed_catboost = df_processed.copy()
                catboost_predictions = predict_catboost(catboost_model, X_test, feature_types).flatten()
                catboost_f1, catboost_conf_matrix = evaluate_model(df_processed_catboost, catboost_predictions, 'CatBoost')
                df_processed_catboost['Kostengruppe'] = df_processed_catboost['Kostengruppe'].astype(str)
                labels = sorted(pd.unique(df_processed_catboost['Kostengruppe'].values))
                catboost_plot_path = 'catboost_conf_matrix.png'
                plot_confusion_matrix(catboost_conf_matrix, labels, "Confusion Matrix - CatBoost", catboost_plot_path)
            except Exception as e:
                print(f"Error processing CatBoost model: {e}")

            results = {
                'lgbm_f1': lgbm_f1,
                'catboost_f1': catboost_f1,
                'lgbm_conf_matrix': 'lgbm_conf_matrix.png',
                'catboost_conf_matrix': 'catboost_conf_matrix.png'
            }

            return render_template('results.html', results=results)

    return render_template('index.html')
