import altair as alt
import pandas as pd
import streamlit as st
#from openpyxl import Workbook
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from scipy.optimize import curve_fit



# Show the page title and description.
st.set_page_config(page_title="Movies dataset (Plumberg)", page_icon="🎬")
st.title("🎬 Movies dataset (Plumberg 2)")
st.write(
    """
    This app performs non-linear regression for data collected from an LRC circuit experiment and visualizes the resulting best-fit curve in comparison to the collected data points.
    """
)

#st.text_input("Your name", key="name")
#st.write("Welcome, ", st.session_state.name, "!")


st.write("Please choose how you want to enter your data:")

# Choose how to enter data
option_map = {
    0: "Upload data from file",
    1: "Enter data manually",
}
selection = st.pills(
    "Tool",
    options=option_map.keys(),
    format_func=lambda option: option_map[option],
    selection_mode="single",
)
st.write(
    "Your selected option: "
    f"{None if selection is None else option_map[selection]}"
)

# Load the data from a CSV. We're caching this so it doesn't reload every time the app
# reruns (e.g. if the user interacts with the widgets).
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    #df = pd.read_excel(uploaded_file)
    return df


dataframe = None

if selection == 0:
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = load_data(uploaded_file)
        st.write(dataframe)
else:
    df = pd.DataFrame(
        [
            {"x": 0, "y": 0}
        ]
    )
    dataframe = st.data_editor(df, num_rows="dynamic",\
            column_config={
            "x": st.column_config.NumberColumn(
                step=1e-16,      # Set a float step to allow decimal entry
                format="%.16f", # Use a float format string
            ),
            "y": st.column_config.NumberColumn(
                step=1e-16,      # Set a float step to allow decimal entry
                format="%.16f", # Use a float format string
            ),
        },)

if dataframe is not None:
    # 1. Define the nonlinear function
    def sigmoid_function(x, a, b, Offset):
        return 1.0 / (1.0 + np.exp(-a * (x - b))) + Offset
        
    # (Assuming xData and yData are defined as numpy arrays)
    xData = dataframe['x'].to_numpy()
    yData = dataframe['y'].to_numpy()
    
    # 4. Use curve_fit to find optimal parameters
    # Provide initial guesses for parameters if possible (optional but recommended for complex models)
    initial_guesses = [0.1, 10.0, 0.0] 
    fitted_params, pcov = curve_fit(sigmoid_function, xData, yData, initial_guesses)
    
    # 5. Get predictions and evaluate
    model_predictions = sigmoid_function(xData, *fitted_params)
    # Calculate R-squared or RMSE here (see search results for examples)
    
    st.write("fitted_params = ", fitted_params)
    
    fig = px.line(dataframe, x='x', y='y')

    st.plotly_chart(fig)

