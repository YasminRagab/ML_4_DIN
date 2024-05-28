from flask import Blueprint, request, render_template, jsonify
import pandas as pd
import io
import joblib
from app.functions.data_processing import load_json_to_df, preprocess_data
from app.functions.model_utils import predict_lgbm, predict_catboost, evaluate_model
from app.functions.plot_utils import plot_confusion_matrix
import base64

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
            
            if not df.empty and 'Kostengruppe' in df.columns and not df['Kostengruppe'].isnull().all():
                df.dropna(subset=['Kostengruppe'], inplace=True)
                X_test_encoded, X_test, feature_types, df_processed = preprocess_data(df, preprocessing_pipeline)

                lgbm_f1, catboost_f1 = None, None
                best_model = None
                lgbm_plot_data = None
                catboost_plot_data = None

                # Process LGBM model
                try:
                    X_test_dense = X_test_encoded.toarray() if hasattr(X_test_encoded, 'toarray') else X_test_encoded
                    df_processed_lgbm = df_processed.copy()
                    lgbm_predictions = lgbm_model.predict(X_test_dense)
                    lgbm_f1, lgbm_conf_matrix = evaluate_model(df_processed_lgbm, lgbm_predictions, 'LGBM')
                    df_processed_lgbm['LGBM_Predicted_Kostengruppe'] = lgbm_predictions
                    lgbm_plot_data = plot_confusion_matrix(lgbm_conf_matrix, sorted(pd.unique(df_processed_lgbm['Kostengruppe'].values)), "Confusion Matrix - LGBM")
                except Exception as e:
                    print(f"Error processing LGBM model: {e}")

                # Process CatBoost model
                try:
                    df_processed_catboost = df_processed.copy()
                    catboost_predictions = predict_catboost(catboost_model, X_test, feature_types).flatten()
                    catboost_f1, catboost_conf_matrix = evaluate_model(df_processed_catboost, catboost_predictions, 'CatBoost')
                    df_processed_catboost['CatBoost_Predicted_Kostengruppe'] = catboost_predictions
                    catboost_plot_data = plot_confusion_matrix(catboost_conf_matrix, sorted(pd.unique(df_processed_catboost['Kostengruppe'].values)), "Confusion Matrix - CatBoost")
                except Exception as e:
                    print(f"Error processing CatBoost model: {e}")

                # Determine the best model
                if lgbm_f1 and catboost_f1:
                    best_model = 'LGBM' if lgbm_f1 > catboost_f1 else 'CatBoost'
                    df_processed['Best_Predicted_Kostengruppe'] = df_processed_lgbm['LGBM_Predicted_Kostengruppe'] if lgbm_f1 > catboost_f1 else df_processed_catboost['CatBoost_Predicted_Kostengruppe']

                # Ensure 'Best_Predicted_Kostengruppe' is added to df_processed
                if 'Best_Predicted_Kostengruppe' in df_processed.columns:
                    output_df = df_processed[['GUID', 'Id', 'Kostengruppe', 'Best_Predicted_Kostengruppe']]
                    output_csv = output_df.to_csv(index=False)
                else:
                    output_csv = "No Best_Predicted_Kostengruppe generated."

                results = {
                    'lgbm_f1': lgbm_f1,
                    'catboost_f1': catboost_f1,
                    'best_model': best_model,
                    'output_csv': output_csv.encode(),  # Ensure it is encoded as bytes
                    'lgbm_conf_matrix': base64.b64encode(lgbm_plot_data.getvalue()).decode('utf-8') if lgbm_plot_data else None,
                    'catboost_conf_matrix': base64.b64encode(catboost_plot_data.getvalue()).decode('utf-8') if catboost_plot_data else None
                }

                return render_template('results_checker.html', results=results)
            else:
                return "The 'Kostengruppe' column is empty or missing.", 400

    return render_template('index.html')
