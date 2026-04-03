import streamlit as st
import pandas as pd
import plotly.graph_objects as gr
from plotly.subplots import make_subplots
from io import StringIO
import os

st.set_page_config(layout="wide")
st.title("📊 Financial Data Explorer")

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# -----------------------------
# 1. FILE MANAGEMENT CONTROLS
# -----------------------------
st.sidebar.header("📁 Data File Manager")

# List saved CSV files
saved_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]

selected_file = st.sidebar.selectbox(
    "Open Saved File",
    ["-- Select File --"] + saved_files
)

if selected_file != "-- Select File --":
    file_path = os.path.join(DATA_FOLDER, selected_file)
    df = pd.read_csv(file_path)
    st.session_state["df"] = df
    st.sidebar.success(f"Loaded: {selected_file}")

# Upload / Paste New CSV
st.sidebar.subheader("📥 Paste CSV Data")

csv_text = st.sidebar.text_area("Paste CSV", height=200)

if st.sidebar.button("Load From Paste"):
    try:
        df = pd.read_csv(StringIO(csv_text), engine="python")
        st.session_state["df"] = df
        st.sidebar.success("Data Loaded from Paste")
    except Exception as e:
        st.sidebar.error(f"CSV Error: {e}")

# -----------------------------
# 2. DEFAULT LOAD
# -----------------------------
if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()

df = st.session_state["df"]

# -----------------------------
# 3. EDITABLE TABLE
# -----------------------------
st.subheader("🛠 Editable Financial Data")

df = st.data_editor(
    df,
    use_container_width=True,
    num_rows="dynamic",
    key="editor"
)

st.session_state["df"] = df

# -----------------------------
# 4. SAVE FILE CONTROL
# -----------------------------
st.sidebar.subheader("💾 Save Current Data")

save_name = st.sidebar.text_input("File Name (without .csv)")

if st.sidebar.button("Save CSV"):
    if save_name:
        save_path = os.path.join(DATA_FOLDER, f"{save_name}.csv")
        df.to_csv(save_path, index=False)
        st.sidebar.success(f"Saved as {save_name}.csv")
    else:
        st.sidebar.warning("Enter a file name first")

st.markdown("---")
st.header("📊 Chart Controls")

if not df.empty and "Metric" in df.columns:

    time_cols = df.columns[1:]

    col1, col2 = st.columns(2)

    with col1:
        var_left = st.selectbox(
            "Left Axis (Primary):",
            df["Metric"].unique(),
            index=0,
            key="left_axis"
        )

    with col2:
        var_right = st.selectbox(
            "Right Axis (Secondary):",
            df["Metric"].unique(),
            index=1,
            key="right_axis"
        )

    st.subheader(f"Analysis: {var_left} vs {var_right}")

    y1_data = df[df["Metric"] == var_left].iloc[0, 1:].values
    y2_data = df[df["Metric"] == var_right].iloc[0, 1:].values

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        gr.Scatter(
            x=time_cols,
            y=y1_data,
            name=var_left,
            mode='lines+markers',
            line=dict(width=3)
        ),
        secondary_y=False,
    )

    fig.add_trace(
        gr.Scatter(
            x=time_cols,
            y=y2_data,
            name=var_right,
            mode='lines+markers',
            line=dict(dash='dot')
        ),
        secondary_y=True,
    )

    fig.update_layout(
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_yaxes(title_text=f"{var_left} Scale", secondary_y=False)
    fig.update_yaxes(title_text=f"{var_right} Scale", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)


