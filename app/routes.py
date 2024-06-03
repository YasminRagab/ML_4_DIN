from flask import Blueprint, request, render_template
import pandas as pd
import io
import numpy as np
import joblib
import base64
import matplotlib.pyplot as plt
from app.functions.data_processing import load_json_to_df, preprocess_data
from app.functions.model_utils import predict_lgbm, predict_catboost, evaluate_model
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
    return render_template('index.html')

@main.route('/model_checker', methods=['GET', 'POST'])
def model_checker():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            json_data = file.read().decode('utf-8')
            df = load_json_to_df(io.StringIO(json_data))
            df.dropna(subset=['Kostengruppe'], inplace=True)
            X_test_encoded, X_test, feature_types, df_processed = preprocess_data(df, preprocessing_pipeline)

            lgbm_f1, lgbm_conf_matrix, lgbm_conf_matrix_img = None, None, None
            try:
                X_test_dense = X_test_encoded.toarray() if hasattr(X_test_encoded, 'toarray') else X_test_encoded
                df_processed_lgbm = df_processed.copy()
                lgbm_predictions = lgbm_model.predict(X_test_dense)
                lgbm_f1, lgbm_conf_matrix = evaluate_model(df_processed_lgbm, lgbm_predictions, 'LGBM')
                df_processed_lgbm['LGBM_Predicted_Kostengruppe'] = lgbm_predictions
                lgbm_conf_matrix_img = plot_confusion_matrix(lgbm_conf_matrix, sorted(pd.unique(df_processed_lgbm['Kostengruppe'].values)), "Confusion Matrix - LGBM")
            except Exception as e:
                print(f"Error processing LGBM model: {e}")

            catboost_f1, catboost_conf_matrix, catboost_conf_matrix_img = None, None, None
            try:
                df_processed_catboost = df_processed.copy()
                catboost_predictions = predict_catboost(catboost_model, X_test, feature_types).flatten()
                catboost_f1, catboost_conf_matrix = evaluate_model(df_processed_catboost, catboost_predictions, 'CatBoost')
                df_processed_catboost['CatBoost_Predicted_Kostengruppe'] = catboost_predictions
                catboost_conf_matrix_img = plot_confusion_matrix(catboost_conf_matrix, sorted(pd.unique(df_processed_catboost['Kostengruppe'].values)), "Confusion Matrix - CatBoost")
            except Exception as e:
                print(f"Error processing CatBoost model: {e}")

            if lgbm_f1 and catboost_f1:
                best_model = 'LGBM' if lgbm_f1 > catboost_f1 else 'CatBoost'
                df_processed['Best_Predicted_Kostengruppe'] = df_processed_lgbm['LGBM_Predicted_Kostengruppe'] if lgbm_f1 > catboost_f1 else df_processed_catboost['CatBoost_Predicted_Kostengruppe']
            else:
                best_model = None

            df_processed['Mismatch'] = df_processed.apply(lambda row: row['Kostengruppe'] != row['Best_Predicted_Kostengruppe'], axis=1)
            output_csv = df_processed[['GUID', 'Id', 'Kostengruppe', 'Best_Predicted_Kostengruppe', 'Mismatch']].to_csv(index=False)

            results = {
                'lgbm_f1': lgbm_f1,
                'catboost_f1': catboost_f1,
                'best_model': best_model,
                'output_csv': base64.b64encode(output_csv.encode()).decode(),
                'lgbm_conf_matrix': lgbm_conf_matrix_img,
                'catboost_conf_matrix': catboost_conf_matrix_img,
            }

            return render_template('results_checker.html', results=results)

    return render_template('index.html')

@main.route('/kg_prediction', methods=['GET', 'POST'])
def kg_prediction():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            json_data = file.read().decode('utf-8')
            df = load_json_to_df(io.StringIO(json_data))
            if 'Kostengruppe' in df.columns:
                df.drop(columns=['Kostengruppe'], inplace=True)
                
            X_test_encoded, X_test, feature_types, df_processed = preprocess_data(df, preprocessing_pipeline)

            lgbm_predictions = lgbm_model.predict(X_test_encoded)
            catboost_predictions = predict_catboost(catboost_model, X_test, feature_types).flatten()

            df_processed['LGBM_Predicted_Kostengruppe'] = lgbm_predictions
            df_processed['CatBoost_Predicted_Kostengruppe'] = catboost_predictions

            df_processed['Agreed_Kostengruppe'] = df_processed.apply(
                lambda row: row['LGBM_Predicted_Kostengruppe'] if row['LGBM_Predicted_Kostengruppe'] == row['CatBoost_Predicted_Kostengruppe'] else 'Disagreement', axis=1
            )

            agreed_df = df_processed[df_processed['Agreed_Kostengruppe'] != 'Disagreement']
            disagreed_df = df_processed[df_processed['Agreed_Kostengruppe'] == 'Disagreement']

            agreed_csv = io.StringIO()
            disagreed_csv = io.StringIO()

            agreed_df[['GUID', 'Id', 'Agreed_Kostengruppe']].to_csv(agreed_csv, index=False)
            disagreed_df[['GUID', 'Id', 'LGBM_Predicted_Kostengruppe', 'CatBoost_Predicted_Kostengruppe']].rename(
                columns={
                    'LGBM_Predicted_Kostengruppe': 'Prediction_Option_01',
                    'CatBoost_Predicted_Kostengruppe': 'Prediction_Option_02'
                }
            ).to_csv(disagreed_csv, index=False)

            results = {
                'agreed_file': base64.b64encode(agreed_csv.getvalue().encode()).decode(),
                'disagreed_file': base64.b64encode(disagreed_csv.getvalue().encode()).decode()
            }

            return render_template('results_prediction.html', results=results)

    return render_template('index.html')


