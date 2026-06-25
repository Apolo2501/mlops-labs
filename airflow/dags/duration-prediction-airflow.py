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
    df = pd.read_parquet(url)

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
        context["ti"].xcom_push(key="df_train", value=df)

    def load_val_data(**context):
        df = read_dataframe(2021, 2)
        context["ti"].xcom_push(key="df_val", value=df)

    def create_train_features(**context):
        df_train = context["ti"].xcom_pull(key="df_train")
        X_train, dv = create_X(df_train)
        y_train = pd.DataFrame(df_train)["duration"].values

        context["ti"].xcom_push(key="X_train", value=X_train)
        context["ti"].xcom_push(key="y_train", value=y_train)
        context["ti"].xcom_push(key="dv", value=dv)

    def create_val_features(**context):
        df_val = context["ti"].xcom_pull(key="df_val")
        dv = context["ti"].xcom_pull(key="dv")

        X_val, _ = create_X(df_val, dv)
        y_val = pd.DataFrame(df_val)["duration"].values

        context["ti"].xcom_push(key="X_val", value=X_val)
        context["ti"].xcom_push(key="y_val", value=y_val)

    def train(**context):
        X_train = context["ti"].xcom_pull(key="X_train")
        y_train = context["ti"].xcom_pull(key="y_train")
        X_val = context["ti"].xcom_pull(key="X_val")
        y_val = context["ti"].xcom_pull(key="y_val")
        dv = context["ti"].xcom_pull(key="dv")

        run_id = train_model(X_train, y_train, X_val, y_val, dv)
        print("MLflow run:", run_id)

    t1 = PythonOperator(task_id="load_train_data", python_callable=load_train_data)
    t2 = PythonOperator(task_id="load_val_data", python_callable=load_val_data)
    t3 = PythonOperator(task_id="create_train_features", python_callable=create_train_features)
    t4 = PythonOperator(task_id="create_val_features", python_callable=create_val_features)
    t5 = PythonOperator(task_id="train_model", python_callable=train)

    t1 >> t2 >> t3 >> t4 >> t5
