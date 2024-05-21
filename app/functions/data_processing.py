import pandas as pd
import json

def load_json_to_df(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    df = pd.DataFrame.from_dict(json_data, orient='index').reset_index()
    df.rename(columns={'index': 'GUID'}, inplace=True)
    return df

def preprocess_data(df, preprocessing_pipeline):
    df.loc[df['LoadBearing'] == '', 'LoadBearing'] = 0
    df.loc[~df['Type'].isin(['Basic Wall', 'Curtain Wall']), 'Type'] = 'Generic Models'
    df.loc[df['cs_layers'] == '', 'cs_layers'] = 1
    df.loc[(df['Type'] == 'Generic Models') & (df['cs_function'] == ''), 'cs_function'] = 'Finish1'
    df.loc[(df['Type'] == 'Generic Models') & (df['material_class'] == ''), 'material_class'] = 'Verschiedene'
    df.loc[(df['Function'] == 1) | (df['Function'] == ''), 'Function'] = 'Exterior'
    df.loc[(df['Kostengruppe'] == '335') & (df['cs_function'] == 'Structure') & (df['Name'] == 'HA_WAN_MET_0,090_xxx'), 'cs_function'] = 'Finish1'
    df.loc[(df['Kostengruppe'] == '332') & (df['cs_function'] == 'Finish1'), 'cs_function'] = 'Structure'
    df.loc[(df['Kostengruppe'] == '332') & (df['cs_function'] == 'Substrate'), 'cs_function'] = 'Structure'
    df.loc[(df['Kostengruppe'] == '345') & (df['cs_function'] == 'Structure'), 'cs_function'] = 'Finish1'
    df.loc[(df['Kostengruppe'] == '342') & (df['cs_function'] == 'Finish1'), 'cs_function'] = 'Structure'
    df.loc[(df['Kostengruppe'] == '342') & (df['cs_function'] == 'Substrate'), 'cs_function'] = 'Structure'
    df = df[(df['Kostengruppe'] != "i.B.") & (df['Kostengruppe'] != '690') & (df['Kostengruppe'] != '354') & (df['Kostengruppe'] != '344')]
    df = df.replace('', pd.NA)
    df = df.dropna(thresh=len(df.columns) - 1)
    condition = df.isnull().sum(axis=1) == 1
    df = df[condition | df['Kostengruppe'].notna()]

    df['LoadBearing'] = df['LoadBearing'].astype(bool)
    df['cs_function'] = df['cs_function'].astype(str)
    df['material_class'] = df['material_class'].astype(str)
    df['Type'] = df['Type'].astype(str)

    feature_columns = ['Name', 'cs_function', 'material_class', 'Function', 'Type', 'LoadBearing']
    X_test = df[feature_columns]
    X_test_encoded = preprocessing_pipeline.transform(X_test)
    return X_test_encoded, X_test, feature_columns, df
