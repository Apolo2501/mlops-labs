from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow
import os


# -------------------------
# FUNCIONES
# -------------------------

def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url, engine="pyarrow")

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df.to_dict()


def create_X(df_dict, dv_dict=None):
    df = pd.DataFrame(df_dict)

    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv_dict is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        dv = pickle.loads(dv_dict)
        X = dv.transform(dicts)

    return X, pickle.dumps(dv)


def train_model(X_train_dict, y_train, X_val_dict, y_val, dv_dict):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("nyc-taxi-experiment")

    os.makedirs(os.path.expanduser("~/airflow/models"), exist_ok=True)

    X_train = X_train_dict
    X_val = X_val_dict
    dv = pickle.loads(dv_dict)

    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        os.makedirs(os.path.expanduser("~/airflow/models"), exist_ok=True)

        with open(os.path.expanduser("~/airflow/models/preprocessor.b"), "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact(os.path.expanduser("~/airflow/models/preprocessor.b"), artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id


# -------------------------
# DAG
# -------------------------

with DAG(
    dag_id="duration_prediction_airflow",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    def load_train_data(**context):
        df = read_dataframe(2021, 1)
        path = "/tmp/df_train.parquet"
        pd.DataFrame(df).to_parquet(path)
        context["ti"].xcom_push(key="df_train_path", value=path)

    def load_val_data(**context):
        df = read_dataframe(2021, 2)
        path = "/tmp/df_val.parquet"
        pd.DataFrame(df).to_parquet(path)
        context["ti"].xcom_push(key="df_val_path", value=path)

    def create_train_features(**context):
        path = context["ti"].xcom_pull(key="df_train_path")
        df_train = pd.read_parquet(path)
        X_train, dv = create_X(df_train.to_dict())
        y_train = df_train["duration"].values

        context["ti"].xcom_push(key="X_train_path", value="/tmp/X_train.pkl")
        context["ti"].xcom_push(key="y_train_path", value="/tmp/y_train.pkl")
        context["ti"].xcom_push(key="dv_path", value="/tmp/dv.pkl")

        pickle.dump(X_train, open("/tmp/X_train.pkl", "wb"))
        pickle.dump(y_train, open("/tmp/y_train.pkl", "wb"))
        pickle.dump(dv, open("/tmp/dv.pkl", "wb"))

    def create_val_features(**context):
        # 1. Recuperamos la ruta del parquet de validación
        df_val_path = context["ti"].xcom_pull(key="df_val_path")

        # 2. Leemos el dataframe desde disco
        df_val = pd.read_parquet(df_val_path)

        # 3. Recuperamos el preprocessor (DictVectorizer) desde disco
        dv_path = context["ti"].xcom_pull(key="dv_path")
        dv = pickle.load(open(dv_path, "rb"))

        # 4. Creamos las features
        X_val, _ = create_X(df_val.to_dict(), dv)

        # 5. Extraemos y guardamos y_val
        y_val = df_val["duration"].values

        # 6. Guardamos X_val y y_val en disco
        X_val_path = "/tmp/X_val.pkl"
        y_val_path = "/tmp/y_val.pkl"

        pickle.dump(X_val, open(X_val_path, "wb"))
        pickle.dump(y_val, open(y_val_path, "wb"))

        # 7. Pasamos SOLO las rutas por XCom
        context["ti"].xcom_push(key="X_val_path", value=X_val_path)
        context["ti"].xcom_push(key="y_val_path", value=y_val_path)


    def train(**context):
        X_train = pickle.load(open(context["ti"].xcom_pull(key="X_train_path"), "rb"))
        y_train = pickle.load(open(context["ti"].xcom_pull(key="y_train_path"), "rb"))
        X_val = pickle.load(open(context["ti"].xcom_pull(key="X_val_path"), "rb"))
        y_val = pickle.load(open(context["ti"].xcom_pull(key="y_val_path"), "rb"))
        dv = pickle.load(open(context["ti"].xcom_pull(key="dv_path"), "rb"))

        run_id = train_model(X_train, y_train, X_val, y_val, dv)
        print("MLflow run:", run_id)

    t1 = PythonOperator(task_id="load_train_data", python_callable=load_train_data)
    t2 = PythonOperator(task_id="load_val_data", python_callable=load_val_data)
    t3 = PythonOperator(task_id="create_train_features", python_callable=create_train_features)
    t4 = PythonOperator(task_id="create_val_features", python_callable=create_val_features)
    t5 = PythonOperator(task_id="train_model", python_callable=train)

    t1 >> t2 >> t3 >> t4 >> t5
