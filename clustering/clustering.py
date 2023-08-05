
from function import class_wise_clustering_A_on_trade,prophet_final,run_prophet_model_for_key,plot_level
import streamlit as st
import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data(persist=True)  # Cache the CSV file reading and persist the cache across sessions
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def select_countrt(d):
    country = st.sidebar.selectbox("Select Country:-",d["geo"].unique().tolist())
    return country


import streamlit as st

import streamlit as st

def select_level(d):
    levels = ["geo", "channel", "sector", "price_tier"]
    selected_levels = st.sidebar.multiselect("Select Levels", levels, default=["geo"])

    selected_channel = None
    selected_sector = None
    selected_price_tier = None
    
    if "channel" in selected_levels:
        st.sidebar.header("Channel")
        channel_options = d["channel"].unique().tolist()
        selected_channel = st.sidebar.selectbox("Select Channel:", channel_options)

    if "sector" in selected_levels:
        st.sidebar.header("Sector")
        sector_options = d["sector"].unique().tolist()
        selected_sector = st.sidebar.selectbox("Select Sector:", sector_options)

    if "price_tier" in selected_levels:
        st.sidebar.header("Price Tier")
        price_tier_options = d["price_tier"].unique().tolist()
        selected_price_tier = st.sidebar.selectbox("Select Price Tier:", price_tier_options)

    return selected_levels, selected_channel, selected_sector, selected_price_tier



def user_input():
    num_clu = st.sidebar.slider("Number of Clusters:", 2, 30, 10)
    return num_clu

@st.cache_data  # Cache the user_input_tuning function
def user_input_tuning(n):
    cluster = st.sidebar.selectbox("Cluster for Tuning:", range(0, n))
    return cluster

def user_input_class():
    cl_for_clustering = st.sidebar.selectbox("Class for Clustering:", ["A", "B", "C", ["A", "B"], ["A", "B", "C"]])
    return cl_for_clustering


def user_input_tuning(n):
    cluster = st.sidebar.selectbox("Cluster for Tuning:", range(0, n))
    return cluster


def user_key(d):
    key =st.selectbox("Select key for forecasting",d["key"].unique().tolist())
    return key



# Custom SessionState class to store variables across Streamlit sessions
class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@st.cache_data(persist=True)  # Cache the clustering result
def perform_clustering(df_ts, df_cls, cl_for_clustering, num_cluster):
    cluster_df = class_wise_clustering_A_on_trade(df_ts, df_cls, cl_for_clustering, num_cluster)
    return cluster_df

def main():
    st.write("""
    <style>
    .title {
        font-size: 24px;
        color: purple;
    }
    </style>
    """, unsafe_allow_html=True)

    # Read the CSV files and cache the DataFrames
    file_path_ts = "/Users/rahulkushwaha/Desktop/streamlit_dashboard/time_series clustering/timeseries_clustering/clustering/aggregated_his_vol_exogs_price_gt.csv"
    file_path_cls = "/Users/rahulkushwaha/Desktop/streamlit_dashboard/time_series clustering/timeseries_clustering/clustering/consumption_abc_classification.csv"
    df_ts = load_data(file_path_ts)
    df_cls = load_data(file_path_cls)

    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.title("Time Series Clustering Dashboard")
    st.header("Introduction")
    st.write(
        "The Time Series Clustering Dashboard organizes time-series data into different groups or clusters. "
        "It selects key elements from each cluster and fine-tunes models based on these elements. The goal is "
        "to find the best combination of settings, known as hyper-parameters, resulting in the highest accuracy "
        "for the chosen elements. Once the best hyper-parameters are identified, they are applied to all the "
        "elements within that particular cluster."
    )

    # Visualization Section
    st.header("Visualization of a Particular Time at Different Levels")
    st.write(
        "Please select the level from the sidebar to visualize the time series data at that level."
    )

    # Additional Explanation
    st.header("Additional Explanation")
    st.write(
        "In this dashboard, you can interactively select various levels to explore the time series data at "
        "different granularities. The available levels include:\n\n"
        "- **geo**: Geographic location\n"
        "- **channel**: Channel information\n"
        "- **sector**: Sector information\n"
        "- **price_tier**: Price tier information\n\n"
        "By selecting different levels, you can gain insights into how the time series data is distributed "
        "and clustered across different dimensions. Have fun exploring the data!"
    )
    country =select_countrt(df_ts)
    level = select_level(df_ts)
    df_level = plot_level(df_ts,level[0],country,level[1],level[2],level[3])
    non_none_levels = [country]

    # Add non-None level elements to the list
    if level[1] is not None:
        non_none_levels.append(level[1])

    if level[2] is not None:
        non_none_levels.append(level[2])

    if level[3] is not None:
        non_none_levels.append(level[3])

    # Combine the non-None level elements into the title variable
    title = "_".join(non_none_levels)
    fig = px.line(data_frame=df_level, x="month", y="volume", title=title)

    # Update the layout of the figure to increase height and width
    fig.update_layout(height=600, width=1500)

    # Display the plot using st.plotly_chart
    st.plotly_chart(fig)

    st.sidebar.header("***Clustering Parameters***")
    num_cluster = user_input()
    cl_for_clustering = user_input_class()
    st.header("Clustering")
    st.markdown("""
            **Please select Parameter for Clustering**""")
    st.markdown("<div class='selected-params'>", unsafe_allow_html=True)
    st.markdown("<div class='selected-params-title'>You selected:</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='selected-params-item'>Class: {cl_for_clustering}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='selected-params-item'>Number of Clusters: {num_cluster}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Perform clustering (and cache the result if not done already)
    cluster_df = perform_clustering(df_ts, df_cls, cl_for_clustering, num_cluster)

    st.write("""
        ###### Number of keys in each cluster:
    """)
    st.write(cluster_df["k_mean_clusters"].value_counts())

    df_temp = df_ts.merge(cluster_df[["key", "k_mean_clusters"]], on="key")

    st.header("Cluster Visualization")
    for c in df_temp["k_mean_clusters"].unique().tolist():
        cl_df = df_temp[df_temp["k_mean_clusters"] == c]
        fig = px.line(data_frame=cl_df.sort_values(by="month"), x="month", y="volume", color="key")

        # Create a larger figure with plotly.graph_objects
        fig.update_layout(height=600, width=1500)  # Adjust height and width as needed

        st.subheader(f"Cluster {c}")
        st.plotly_chart(fig)
    st.markdown("""
        <h3 style="color: blue; text-decoration: underline; font-weight: bold;">
        Hyper-Parameter Tuning
        </h3>
    """, unsafe_allow_html=True)

    st.sidebar.header("***Modelling Parameters***")

    if st.sidebar.checkbox("Perform Hyperparameter Tuning"):
        cluster = user_input_tuning(num_cluster)
        st.markdown(f"#### Selected cluster for tuning: {cluster}")

        # Filter data for the selected cluster
        cluster_df_tune = df_temp[df_temp["k_mean_clusters"] == cluster]

        hyp_result = prophet_final(cluster_df_tune, cluster)

        st.markdown(""" #### Selected cluster""")
        st.write("Cluster:", cluster)
        st.write("Best Hyperparameter:", hyp_result[0][["hyp_para"]])

        # Get the best hyperparameter value
        best_hyp = hyp_result[0]["hyp_para"].values[0]

        # Filter data for the selected cluster
        k = user_key(df_temp[df_temp["k_mean_clusters"] == cluster])

        # Run the Prophet model for the selected cluster and hyperparameter
        fo_df = run_prophet_model_for_key(df_temp, best_hyp, k)

        for_his_df = fo_df[-1]

        st.plotly_chart(px.line(data_frame=for_his_df, x="ds", y=["volume", "yhat"],
                                title=for_his_df["key"].unique()[0] + "||" + str(round(for_his_df["accuracy"].unique()[0]))))
    else:
        st.write("Hyperparameter Tuning is not selected. Only the clustering visualization will be displayed.")
    st.markdown("</div>", unsafe_allow_html=True)
if __name__ == "__main__":
    main()
