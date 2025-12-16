import altair as alt
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from scipy.optimize import curve_fit

# --- 1. Define Callback Function ---
# This function is run when the st.data_editor changes and commits its value.
# It stores the current, edited data into the session state.
def update_manual_df():
    """Callback function to store the data editor's current state."""
    # The key 'data_editor' is used to get the widget's current value
    st.session_state['manual_df'] = st.session_state['data_editor']

# Show the page title and description.
st.set_page_config(page_title="Non-Linear Regression App", page_icon="📈")
st.title("📈 Non-Linear Regression & Data Visualization")
st.write(
    """
    This app performs non-linear regression using the function $f(x) = c \\cdot \\tanh(a \\cdot (x - b)) + \\text{Offset}$
    and visualizes the resulting best-fit curve in comparison to the collected data points.
    """
)

# --- 2. Define the Fit Function ---
def fit_function(x, a, b, c, Offset):
    # The model function for curve_fit
    return c * np.tanh(a * (x - b)) + Offset
    

# --- 3. Data Loading and Selection UI ---
st.write("Please choose how you want to enter your data:")

# Choose how to enter data
option_map = {
    0: "Upload data from file",
    1: "Enter data manually",
}
selection = st.pills(
    "Input Method",
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
    # Only handles CSV as per original script, modify for Excel if needed
    df = pd.read_csv(uploaded_file)
    return df


#========================================
# --- Option 0: Upload data from file ---
#========================================
if selection == 0:
    uploaded_file = st.file_uploader("Choose a CSV file with 'x' and 'y' columns")
    
    if uploaded_file is not None:
        try:
            dataframe = load_data(uploaded_file)
            st.write("### Uploaded Data Preview")
            st.write(dataframe)

            # Ensure 'x' and 'y' columns exist
            if 'x' in dataframe.columns and 'y' in dataframe.columns:
                
                xData = dataframe['x'].to_numpy()
                yData = dataframe['y'].to_numpy()
                
                st.write("### Regression Results")

                # Perform curve fitting
                initial_guesses = [1.0, 0.0, 1.0, 0.0] 
                fitted_params, pcov = curve_fit(fit_function, xData, yData, initial_guesses)
                
                st.write("Fitted Parameters (a, b, c, Offset):", fitted_params)
                
                # Get predictions for a smooth plot
                x_fit = np.linspace(xData.min(), xData.max(), 500)
                model_predictions = fit_function(x_fit, *fitted_params)

                # Prepare data for plotting
                df_fit = pd.DataFrame({'x': x_fit, 'y': model_predictions, 'Type': 'Fit Curve'})
                df_data = pd.DataFrame({'x': xData, 'y': yData, 'Type': 'Data Points'})
                
                # Create the scatter plot for data and line plot for the fit
                fig = px.scatter(df_data, x='x', y='y', title="Data Points and Fitted Curve (File Upload)")
                fig.add_scatter(x=df_fit['x'], y=df_fit['y'], mode='lines', name='Fit Curve', line=dict(color='red'))
                
                st.plotly_chart(fig)
            else:
                st.error("The uploaded file must contain columns named 'x' and 'y'.")

        except Exception as e:
            st.error(f"Error processing file: {e}")


#========================================
# --- Option 1: Enter data manually ---
#========================================
elif selection == 1:
    
    # 1. Initialize DataFrame in session state if it doesn't exist
    if 'manual_df' not in st.session_state:
        st.session_state['manual_df'] = pd.DataFrame(
            [
                {"x": 0.0, "y": 0.0}
            ]
        )

    st.write("### Data Input")
    
    # 2. Use st.data_editor with key and callback for persistence
    dataframe = st.data_editor(
        st.session_state['manual_df'],
        num_rows="dynamic",
        key="data_editor", # Unique key for the editor widget
        on_change=update_manual_df, # Function to call when data changes
        column_config={
            "x": st.column_config.NumberColumn(
                label="X Value",
                step=1e-16,
                format="%.16f",
            ),
            "y": st.column_config.NumberColumn(
                label="Y Value",
                step=1e-16,
                format="%.16f",
            ),
        }
    )
    
    st.write("### Regression Results")
    
    # Ensure there are at least 4 data points (for 4 parameters) before attempting curve fitting, 
    # but at least 2 points for a basic plot/initial guess.
    if len(dataframe) >= 4:
        
        # (Assuming xData and yData are defined as numpy arrays)
        xData = dataframe['x'].to_numpy()
        yData = dataframe['y'].to_numpy()
        
        # 4. Use curve_fit to find optimal parameters
        try:
            # Initial guesses for a, b, c, Offset
            initial_guesses = [1.0, xData.mean(), (yData.max() - yData.min()) / 2, yData.mean()] 
            fitted_params, pcov = curve_fit(fit_function, xData, yData, p0=initial_guesses)
            
            # 5. Get predictions and evaluate
            st.write("Fitted Parameters (a, b, c, Offset):", fitted_params)
            
            # Create smooth X values for a smooth curve plot
            x_fit = np.linspace(xData.min(), xData.max(), 500)
            model_predictions = fit_function(x_fit, *fitted_params)
            
            # Prepare data for plotting
            df_fit = pd.DataFrame({'x': x_fit, 'y': model_predictions, 'Type': 'Fit Curve'})
            df_data = pd.DataFrame({'x': xData, 'y': yData, 'Type': 'Data Points'})
            
            # Create the scatter plot for data and line plot for the fit
            fig = px.scatter(df_data, x='x', y='y', title="Data Points and Fitted Curve (Manual Input)")
            fig.add_scatter(x=df_fit['x'], y=df_fit['y'], mode='lines', name='Fit Curve', line=dict(color='red'))
            
            st.plotly_chart(fig)
            
        except RuntimeError:
            st.warning("Could not find optimal parameters for the fit function. Check your data or initial guesses.")

    elif len(dataframe) > 1:
        st.info(f"You have {len(dataframe)} data points. Input at least 4 points for the regression to run reliably (4 parameters: a, b, c, Offset).")
        # Plot just the raw data for visual feedback
        fig = px.scatter(dataframe, x='x', y='y', title="Data Points (Regression requires more data)")
        st.plotly_chart(fig)
    else:
        st.info("Please input at least two data points.")