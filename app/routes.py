from flask import Blueprint, request, render_template, send_file
import pandas as pd
import os
import sys
import joblib

# Add the app directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
                df_processed_lgbm['LGBM_Predicted_Kostengruppe'] = lgbm_predictions
            except Exception as e:
                lgbm_f1, lgbm_conf_matrix = None, None
                print(f"Error processing LGBM model: {e}")

            try:
                df_processed_catboost = df_processed.copy()
                catboost_predictions = predict_catboost(catboost_model, X_test, feature_types).flatten()
                catboost_f1, catboost_conf_matrix = evaluate_model(df_processed_catboost, catboost_predictions, 'CatBoost')
                df_processed_catboost['CatBoost_Predicted_Kostengruppe'] = catboost_predictions
            except Exception as e:
                catboost_f1, catboost_conf_matrix = None, None
                print(f"Error processing CatBoost model: {e}")

            if lgbm_f1 and catboost_f1:
                best_model = 'LGBM' if lgbm_f1 > catboost_f1 else 'CatBoost'
                df_processed['Best_Predicted_Kostengruppe'] = df_processed_lgbm['LGBM_Predicted_Kostengruppe'] if lgbm_f1 > catboost_f1 else df_processed_catboost['CatBoost_Predicted_Kostengruppe']
            else:
                best_model = None

            output_df = df_processed[['GUID', 'Id', 'Kostengruppe', 'Best_Predicted_Kostengruppe']]
            output_path = os.path.join('uploads', 'model_checker_output.csv')
            output_df.to_csv(output_path, index=False)

            results = {
                'lgbm_f1': lgbm_f1,
                'catboost_f1': catboost_f1,
                'best_model': best_model,
                'output_file': 'model_checker_output.csv'
            }

            return render_template('results_checker.html', results=results)

    return render_template('index.html')

@main.route('/kg_prediction', methods=['GET', 'POST'])
def kg_prediction():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)

            df = load_json_to_df(filepath)
            if 'Kostengruppe' not in df.columns or df['Kostengruppe'].isnull().all():
                X_test_encoded, X_test, feature_types, df_processed = preprocess_data(df, preprocessing_pipeline)

                df_processed_lgbm = df_processed.copy()
                df_processed_catboost = df_processed.copy()

                lgbm_predictions = lgbm_model.predict(X_test_encoded)
                catboost_predictions = predict_catboost(catboost_model, X_test, feature_types).flatten()

                df_processed['LGBM_Predicted_Kostengruppe'] = lgbm_predictions
                df_processed['CatBoost_Predicted_Kostengruppe'] = catboost_predictions

                df_processed['Agreed_Kostengruppe'] = df_processed.apply(
                    lambda row: row['LGBM_Predicted_Kostengruppe'] if row['LGBM_Predicted_Kostengruppe'] == row['CatBoost_Predicted_Kostengruppe'] else 'Disagreement', axis=1
                )

                agreed_df = df_processed[df_processed['Agreed_Kostengruppe'] != 'Disagreement']
                disagreed_df = df_processed[df_processed['Agreed_Kostengruppe'] == 'Disagreement']

                output_agreed_path = os.path.join('uploads', 'kg_prediction_agreed.csv')
                output_disagreed_path = os.path.join('uploads', 'kg_prediction_disagreed.csv')

                agreed_df[['GUID', 'Id', 'Agreed_Kostengruppe']].to_csv(output_agreed_path, index=False)
                disagreed_df[['GUID', 'Id', 'LGBM_Predicted_Kostengruppe', 'CatBoost_Predicted_Kostengruppe']].to_csv(output_disagreed_path, index=False)

                results = {
                    'agreed_file': 'kg_prediction_agreed.csv',
                    'disagreed_file': 'kg_prediction_disagreed.csv'
                }

                return render_template('results_prediction.html', results=results)

    return render_template('index.html')

@main.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_file(os.path.join('uploads', filename), as_attachment=True)
