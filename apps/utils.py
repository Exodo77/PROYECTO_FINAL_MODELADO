from pathlib import Path
import joblib
import pandas as pd
import numpy as np


def load_model_and_preprocessor(models_dir: Path = None):
    """Carga el modelo y el preprocessor desde la carpeta `models`.

    Devuelve una tupla (model, preprocessor). Si alguno no existe, devuelve None en su lugar.
    """
    if models_dir is None:
        models_dir = Path(__file__).resolve().parent.parent / "models"
    model = None
    preprocessor = None
    try:
        mpath = models_dir / "modelo_ganador.pkl"
        if mpath.exists():
            model = joblib.load(mpath)
    except Exception:
        model = None
    try:
        ppath = models_dir / "preprocessor.pkl"
        if ppath.exists():
            preprocessor = joblib.load(ppath)
    except Exception:
        preprocessor = None
    return model, preprocessor


def prepare_input_dataframe(inputs: dict, expected_columns: list = None, categorical_columns: list = None):
    """Construye un DataFrame a partir de `inputs`.

    Si `expected_columns` está provisto, se crea un DataFrame que contiene exactamente
    esas columnas (rellenando con 0s cuando falta información). En caso contrario,
    se crea un DataFrame minimal con las columnas principales usadas en la demo.
    """
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Si el usuario nos pasa la lista de columnas esperadas, construiremos el df acorde
    if expected_columns is not None:
        row = {}
        sel_month = inputs.get('Month', None)
        browser_value = int(inputs.get('Browser', 12) or 12)
        traffic_type_value = int(inputs.get('TrafficType', 1) or 1)
        for col in expected_columns:
            if col == 'PageValues':
                row[col] = float(inputs.get('PageValues', 0.0))
            elif col == 'Administrative_Duration':
                row[col] = float(inputs.get('Administrative_Duration', 0.0))
            elif col == 'ProductRelated_Duration':
                row[col] = float(inputs.get('ProductRelated_Duration', 0.0))
            elif col == 'TrafficType':
                row[col] = traffic_type_value
            elif col.startswith('TrafficType_'):
                try:
                    row[col] = 1 if int(col.split('_', 1)[1]) == traffic_type_value else 0
                except Exception:
                    row[col] = 0
            elif col == 'Browser':
                row[col] = browser_value
            elif col == 'Browser_12':
                row[col] = int(browser_value == 12)
            elif col == 'Month':
                # si el preprocessor espera la columna 'Month' como categórica
                row[col] = str(inputs.get('Month', '') )
            elif col.startswith('Month_'):
                # e.g. Month_Nov
                if sel_month is not None and col.lower().endswith(sel_month.lower()[:3]):
                    row[col] = 1
                else:
                    row[col] = 0
            else:
                # si la columna es categórica conocida, rellenar con string vacío o con el valor provisto
                if categorical_columns and col in categorical_columns:
                    row[col] = str(inputs.get(col, ''))
                else:
                    # rellenar con 0 por defecto
                    row[col] = 0
        return pd.DataFrame([row])

    # comportamiento por defecto (mini esquema)
    data = {}
    data['PageValues'] = float(inputs.get('PageValues', 0.0))
    data['Administrative_Duration'] = float(inputs.get('Administrative_Duration', 0.0))
    data['ProductRelated_Duration'] = float(inputs.get('ProductRelated_Duration', 0.0))
    browser_value = int(inputs.get('Browser', 12) or 12)
    traffic_type_value = int(inputs.get('TrafficType', 1) or 1)
    data['Browser'] = browser_value
    data['Browser_12'] = int(browser_value == 12)
    data['TrafficType'] = traffic_type_value
    for t in range(1, 21):
        data[f'TrafficType_{t}'] = 1 if t == traffic_type_value else 0
    sel_month = inputs.get('Month', None)
    for m in months:
        key = f"Month_{m}"
        data[key] = 1 if (sel_month is not None and sel_month.lower().startswith(m.lower()[:3])) else 0
    return pd.DataFrame([data])


def predict_from_inputs(inputs: dict, model, preprocessor=None):
    """Recibe un dict de inputs, aplica el preprocessor si existe y devuelve (prob, pred).

    - prob: probabilidad de clase positiva (float entre 0 y 1)
    - pred: clase predicha (0 o 1)
    """
    # Si hay preprocessor y tiene información de columnas entrenadas, respetarlas
    df = None
    try:
        # Si el modelo ya es un Pipeline que incluye ColumnTransformer, NO apliquemos
        # el preprocessor por separado porque causaría doble transformación.
        model_contains_preproc = False
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer
            if isinstance(model, Pipeline):
                for _name, step in model.steps:
                    if isinstance(step, ColumnTransformer):
                        model_contains_preproc = True
                        break
            if hasattr(model, 'named_steps'):
                for step in getattr(model, 'named_steps').values():
                    if isinstance(step, ColumnTransformer):
                        model_contains_preproc = True
                        break
        except Exception:
            model_contains_preproc = False

        if preprocessor is not None and hasattr(preprocessor, 'feature_names_in_'):
            expected = list(preprocessor.feature_names_in_)
            # detectar columnas categóricas (OneHotEncoder) en el preprocessor
            cat_cols = []
            try:
                from sklearn.preprocessing import OneHotEncoder
                def contains_onehot(trans):
                    from sklearn.pipeline import Pipeline
                    if isinstance(trans, OneHotEncoder):
                        return True
                    if isinstance(trans, Pipeline):
                        for step in trans.steps:
                            if contains_onehot(step[1]):
                                return True
                    return False

                for name, transformer, cols in getattr(preprocessor, 'transformers_', []):
                    try:
                        if contains_onehot(transformer):
                            if isinstance(cols, (list, tuple)):
                                for c in cols:
                                    cat_cols.append(c)
                    except Exception:
                        continue
            except Exception:
                cat_cols = []

            # Si el modelo contiene su propio preprocessor, debemos pasarle el DataFrame
            # con las 17 columnas originales; en caso contrario aplicamos el preprocessor
            if model_contains_preproc:
                df = prepare_input_dataframe(inputs, expected_columns=expected, categorical_columns=cat_cols)
            else:
                df = prepare_input_dataframe(inputs, expected_columns=expected, categorical_columns=cat_cols)
        else:
            df = prepare_input_dataframe(inputs)
    except Exception:
        df = prepare_input_dataframe(inputs)
    X = None
    try:
        if preprocessor is not None and not model_contains_preproc:
            X = preprocessor.transform(df)
            model_input = X
        else:
            # pasar un DataFrame al modelo en caso de que sea un Pipeline/ColumnTransformer
            model_input = df
    except Exception:
        model_input = df

    # manejar modelos que devuelven probabilidad
    try:
        probs = model.predict_proba(model_input)
        prob = float(probs[:, 1][0])
    except Exception:
        # si no hay predict_proba, usar predict y convertir a 0/1
        pred = int(model.predict(model_input)[0])
        prob = float(pred)
    try:
        pred = int(prob >= 0.5)
    except Exception:
        pred = int(model.predict(model_input)[0])
    return prob, pred
