import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt  
import plotly.express as px 
import streamlit as st 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from dateutil.relativedelta import relativedelta
import ast

# Function to perform time series clustering
def class_wise_clustering_A_on_trade(df_pred: pd.DataFrame, df_class: pd.DataFrame,c,nu_cl):
    """
    Perform class-wise clustering on a DataFrame.

    Args:
        df_pred (pd.DataFrame): DataFrame containing the prepared data for modelling.
        df_class (pd.DataFrame): DataFrame containing the class information.
        class_for_clus: The class for which clustering will be performed.
        num_cluster (int): Number of clusters to be created.

    Returns:
        pd.DataFrame: DataFrame with cluster labels assigned to each data point.
    """
    nan_key=df_pred[df_pred["volume"].isnull()]["key"].unique().tolist()
    df_pred=df_pred[~df_pred["key"].isin(nan_key)]
    df_pre_mod = df_pred.merge(df_class[["key", "class"]], on="key")
    if len(c)==1:
        df_class = df_pre_mod[df_pre_mod["class"]==c]
    else:
        df_class = df_pre_mod[df_pre_mod["class"].isin(c)]

    # Adding a quarter column
    df_class['quarter'] = pd.PeriodIndex(df_class.month, freq='Q')

    # Standard scaling
    df_class['volume'] = df_class[['key', 'volume']].groupby('key').transform(
        lambda x: StandardScaler().fit_transform(x.values[:, np.newaxis]).ravel())

    def quant25(g):
        return np.percentile(g, 25)

    def quant50(g):
        return np.percentile(g, 50)

    def quant75(g):
        return np.percentile(g, 75)

    # Create a DataFrame with all numerical descriptions for each key quarterly
    df_final = df_class.pivot_table(index=['key', 'sector'], values='volume', columns='quarter',
                                    aggfunc={'volume': ['mean', 'std', 'max', 'min', quant25, quant50, quant75]})
    df_final.columns = ['_'.join(str(s).strip() for s in col if s) for col in df_final.columns]
    df_final = df_final.reset_index()

    # Perform one-hot encoding for the 'sector' column
    df_final = pd.get_dummies(
        df_final,
        columns=['sector'],
        drop_first=True,
    )

    df_final = df_final.set_index('key')

    # Perform clustering using K-means algorithm
    clustering_kmeans = KMeans(
        init="random",
        n_clusters=nu_cl,
        n_init=10,
        max_iter=300,
        random_state=None
    )
    df_final['k_mean_clusters'] = clustering_kmeans.fit_predict(df_final)
    df_final = df_final.reset_index()

    return df_final

# Function to tune Prophet hyperparameters for a specific group
def tune_prophet_hyperparameters(df_gb):
    """
    Tune Prophet hyperparameters for a specific group.

    Args:
        df_gb (DataFrame): Grouped DataFrame containing "month" and "volume" columns.

    Returns:
        tuple: Best hyperparameters, best accuracy, and results DataFrame.
    """
    hyperparameters = {
        'yearly_seasonality': [True],
        'changepoint_prior_scale': [0.05, 0.1, 0.5, 1.0],
        'seasonality_prior_scale': [0.05, 0.1, 0.5, 1.0],
        'changepoint_range': [0.8, 0.9],
        'seasonality_mode': ['additive'],
    }

    best_accuracy = -np.inf
    best_hyperparameters = None
    results = pd.DataFrame()

    # Iterate over the hyperparameter grid and select the best combination
    for params in ParameterGrid(hyperparameters):
        accuracy, for_actual, pred_pred = run_prophet_model(df_gb, params)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hyperparameters = params
            results = for_actual

    return best_hyperparameters, best_accuracy, results


# Function to fit Prophet model, make predictions, and calculate accuracy
def run_prophet_model(df_gb, hyperparameters):
    """
    Fit Prophet model, make predictions, and calculate accuracy.

    Args:
        df_gb (DataFrame): Grouped DataFrame containing "month" and "volume" columns.
        hyperparameters (dict): Hyperparameters for the Prophet model.

    Returns:
        tuple: Accuracy, DataFrame with actual and predicted values, and predicted DataFrame.
    """
    forecast_result = {}
    df_gb = df_gb[["month", "volume"]].rename(columns={"month": "ds", "volume": "y"})
    df_gb = df_gb.reset_index()
    req_date = (pd.to_datetime(df_gb["ds"]).max() - relativedelta(months=3)).strftime("%Y-%m-%d")
    train = df_gb[df_gb["ds"] <= req_date]
    test = df_gb[df_gb["ds"] > req_date]

    model_p = Prophet(**hyperparameters)
    model_p.fit(train)
    forecast = model_p.make_future_dataframe(periods=42, freq='MS', include_history=True)
    pred_p = model_p.predict(forecast)

    forecast_result["test"] = test[["ds", "y"]]
    forecast_result["prophet"] = pred_p[pred_p["ds"].isin(test["ds"])][["ds", "yhat"]]

    test["ds"] = pd.to_datetime(test["ds"])
    pred_p["ds"] = pd.to_datetime(pred_p["ds"])
    for_actual = test[["ds", "y"]].merge(pred_p[pred_p["ds"].isin(test["ds"])], on="ds", suffixes=("_actual", "_pred"))
    for_actual["accuracy"] = 100 - mean_absolute_percentage_error(for_actual["y"], for_actual["yhat"]) * 100

    return for_actual["accuracy"].mean(), for_actual, pred_p


# Function to perform Prophet tuning and predictions for a given cluster
def prophet_tuning(combo):
    """
    Perform Prophet tuning and predictions for a given cluster.

    Args:
        combo (tuple): Tuple containing DataFrame with cluster information and the cluster value.

    Returns:
        tuple: Accuracy DataFrame and best hyperparameters for the cluster.
    """
    df_with_cluster, cluster = combo
    cluster_df = df_with_cluster[df_with_cluster["k_mean_clusters"] == int(cluster)]
    grouped = cluster_df.groupby("key")
    best_hyp = {}
    result_df = {}

    # Iterate over unique keys in the cluster and tune hyperparameters
    for k in cluster_df["key"].unique().tolist()[:3]:
        df_key = grouped.get_group(k)
        best_hyperparameters, _, results = tune_prophet_hyperparameters(df_key)
        best_hyp[k] = best_hyperparameters
        result_df[k] = results

    # Combine accuracy results for all keys in the cluster
    acc_df = pd.concat([result_df[key].assign(key=key) for key in cluster_df["key"].unique().tolist()[:3]])
    accuracy_df = acc_df.groupby("key").mean()["accuracy"].reset_index().sort_values(by="accuracy", ascending=False)
    best_acc_key = accuracy_df.iloc[0]["key"]
    best_hyp_clu = best_hyp[best_acc_key]
    best_hyp_clu = {'cluster': cluster, 'hyp_para': best_hyp_clu}
    return accuracy_df, best_hyp_clu


# Main function to perform Prophet tuning and predictions for all clusters of ontrade and offtrade
def prophet_final(df_ontrade,clu_hyp):
    """
    Perform Prophet tuning and predictions for all clusters of ontrade and offtrade.

    Args:
        df_ontrade (DataFrame): DataFrame containing ontrade data.

    Returns:
        tuple: DataFrame with best hyperparameters for each cluster and accuracy DataFrame.
    """
    best_key_acc = pd.DataFrame()
    combinations = [(df_ontrade, clu) for clu in [clu_hyp]]

    result_tuples_multiproc = Parallel(n_jobs=-1)(delayed(prophet_tuning)(combo) for combo in combinations)
    dict_list = []
    for result in result_tuples_multiproc:
        dict_list.append(result[1])
        df_hyp = pd.DataFrame(dict_list)
        df_hyp = df_hyp.reset_index()
        best_key_acc = pd.concat([best_key_acc, result[0]])

    return df_hyp, best_key_acc

def run_prophet_model_for_key(df2, hyperparameters,k):
    """
    Fit Prophet model, make predictions, and calculate accuracy.

    Args:
        df_gb (DataFrame): Grouped DataFrame containing "month" and "volume" columns.
        hyperparameters (dict): Hyperparameters for the Prophet model.

    Returns:
        tuple: Accuracy, DataFrame with actual and predicted values, and predicted DataFrame.
    """
    df_gb=df2[df2["key"]==k]
    df_his=df_gb.copy()
    forecast_result = {}
    df_gb = df_gb[["month", "volume"]].rename(columns={"month": "ds", "volume": "y"})
    df_gb = df_gb.reset_index()
    req_date = (pd.to_datetime(df_gb["ds"]).max() - relativedelta(months=3)).strftime("%Y-%m-%d")
    train = df_gb[df_gb["ds"] <= req_date]
    test = df_gb[df_gb["ds"] > req_date]

    model_p = Prophet(**hyperparameters)
    model_p.fit(train)
    forecast = model_p.make_future_dataframe(periods=42, freq='MS', include_history=True)
    pred_p = model_p.predict(forecast)

    forecast_result["test"] = test[["ds", "y"]]
    forecast_result["prophet"] = pred_p[pred_p["ds"].isin(test["ds"])][["ds", "yhat"]]

    test["ds"] = pd.to_datetime(test["ds"])
    pred_p["ds"] = pd.to_datetime(pred_p["ds"])
    for_actual = test[["ds", "y"]].merge(pred_p[pred_p["ds"].isin(test["ds"])], on="ds", suffixes=("_actual", "_pred"))
    for_actual["accuracy"] = 100 - mean_absolute_percentage_error(for_actual["y"], for_actual["yhat"]) * 100
    df_his=df_his.rename(columns={"month":"ds"})
    df_his["ds"]=pd.to_datetime(df_his["ds"])
    pred_p["ds"]=pd.to_datetime(pred_p["ds"])
    act_for = pred_p.merge(df_his,on="ds",how="left")
    act_for["accuracy"]=for_actual["accuracy"].mean()
    return for_actual["accuracy"].mean(), for_actual, pred_p,act_for

def plot_level(df, level, country, channel,sector,price_tier):
    df_t = df[df["geo"] == country]
    df_t = df_t.groupby(level + ["month"]).sum()["volume"].reset_index()
    
    if channel is not None:
        df_t = df_t[df_t["channel"] == channel]
    
    if sector is not None:
        df_t = df_t[df_t["sector"] == sector]
    
    if price_tier is not None:
        df_t = df_t[df_t["price_tier"] == price_tier]
    if len(df_t)==0:
        st.write(" ### This Combination does not exist")

    return df_t

import streamlit as st
import plotly.express as px

# Set Streamlit page config
st.set_page_config(
    page_title="Time Series Clustering Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS style
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f0f0;
    }
    .sidebar .sidebar-content .block-container {
        padding: 1rem;
    }
    .header {
        font-size: 2.2rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
    }
    .subheader {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .title {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .plot-container {
        border: 1px solid #ccc;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)






