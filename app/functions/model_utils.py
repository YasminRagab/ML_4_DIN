from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import joblib

def predict_lgbm(model, X_test):
    return model.predict(X_test, num_iteration=model.best_iteration)

def predict_catboost(model, X_test, feature_types):
    return model.predict(X_test)

def evaluate_model(df, predictions, model_name):
    if len(predictions) == len(df):
        df[f'{model_name}_Predicted_Kostengruppe'] = predictions
        df.dropna(subset=['Kostengruppe', f'{model_name}_Predicted_Kostengruppe'], inplace=True)
        true_labels = df['Kostengruppe'].astype(str)
        pred_labels = df[f'{model_name}_Predicted_Kostengruppe'].astype(str)
        f1 = f1_score(true_labels, pred_labels, average='weighted')
        conf_matrix = confusion_matrix(true_labels, pred_labels, labels=np.unique(true_labels))
        return f1, conf_matrix
    else:
        raise ValueError("Mismatch in lengths between predictions and actual data")

def retrain_models(df, lgbm_model, catboost_model, preprocessing_pipeline):
    existing_data = pd.read_csv('path_to_existing_training_data.csv')
    new_data = existing_data.append(df, ignore_index=True)
    new_data.to_csv('path_to_existing_training_data.csv', index=False)

    # Define feature and target columns
    feature_columns = ['Name', 'cs_function', 'material_class', 'Function', 'Type', 'LoadBearing']
    target_column = ['Kostengruppe']

    # Preprocess the data
    new_data[feature_columns] = preprocessing_pipeline.transform(new_data[feature_columns])

    # Split the data
    X = new_data[feature_columns]
    y = new_data[target_column]

    # Train models
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

    # Retrain LGBM model
    lgbm_model.fit(X_train, y_train)
    joblib.dump(lgbm_model, 'app/models/lgbm_model.joblib')

    # Retrain CatBoost model
    catboost_model.fit(X_train, y_train)
    joblib.dump(catboost_model, 'app/models/catboost_model.joblib')
