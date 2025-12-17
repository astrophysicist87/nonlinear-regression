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
    This app performs non-linear regression using the function $f(t) = A e^{-t/\\tau}\\cdot \\cos(\\omega \\cdot t + \\phi)$
    and visualizes the resulting best-fit curve in comparison to the collected data points.
    """
)

# --- 2. Define the relevant functions ---
# ---
def fit_function(t, A, tau, omega, phi):
    # The model function for curve_fit
    return A * np.exp(-t / tau) * np.cos(omega * t + phi)

# ---
def get_r_squared(f_data, f_pred):
    #Calculate the sum of squares of residual
    ss_res=np.sum((f_data-f_pred)**2) 

    #Calculate the total sum of squares
    ss_tot=np.sum((f_data-np.mean(f_data))**2)

    #Return R-Squared
    return 1.-(ss_res/ss_tot)
    

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

# Display guesses for initial parameters
st.write("Initial guesses for parameters ($A$, $\\tau$, $\\omega$, $\\phi$):",)
initial_guesses = st.data_editor(
    pd.DataFrame({
    "Parameter": ['A', 'tau', 'omega', 'phi'],
    "Value": [1.0, 1.0, 1.0, 1.0]
}),
    # Lock the number of rows so users can't add/delete parameters
    num_rows="fixed",
    # Hide the 0,1,2... index column
    hide_index=True,
    # Configure specific column behaviors
    column_config={
        "Parameter Name": st.column_config.TextColumn(
            "Parameter",
            disabled=True  # This makes the first column Read-Only
        ),
        "Value": st.column_config.Column(
            "Value",
            help="Click to edit this value"
        )
    },
    # Ensure the editor takes up reasonable width
    width='stretch'
)

#========================================
# --- Option 0: Upload data from file ---
#========================================
if selection == 0:
    uploaded_file = st.file_uploader("Choose a CSV file with 't' and 'f(t)' columns")
    
    if uploaded_file is not None:
        try:
            dataframe = load_data(uploaded_file)
            st.write("### Uploaded Data Preview")
            st.write(dataframe)

            # Ensure 't' and 'f(t)' columns exist
            if 't' in dataframe.columns and 'f(t)' in dataframe.columns:
                
                xData = dataframe['t'].to_numpy()
                yData = dataframe['f(t)'].to_numpy()
                
                st.write("### Regression Results")

                # Perform curve fitting
                #initial_guesses = [1.0, 0.0, 1.0, 0.0]
                fitted_params, pcov = curve_fit(fit_function, xData, yData, initial_guesses['Value'].to_numpy())
                
                # Display fit parameters and corresponding uncertainties
                st.write("Fitted Parameters ($A$, $\\tau$, $\\omega$, $\\phi$):")
                st.dataframe(pd.DataFrame({"Parameter": fitted_params,
                                           "Uncertainty": np.sqrt(np.diag(pcov))}),
                             hide_index = True)

                # Get predictions for a smooth plot
                x_fit = np.linspace(xData.min(), xData.max(), 500)
                model_predictions = fit_function(x_fit, *fitted_params)

                # Prepare data for plotting
                df_fit = pd.DataFrame({'t': x_fit, 'f(t)': model_predictions, 'Type': 'Fit Curve'})
                df_data = pd.DataFrame({'t': xData, 'f(t)': yData, 'Type': 'Data Points'})
                
                # Print r^2
                r2 = get_r_squared(yData, fit_function(xData, *fitted_params))
                st.write(rf'''Goodness-of-fit $r^2$ = {r2}''')
            
                # Create the scatter plot for data and line plot for the fit
                fig = px.scatter(df_data, x='t', y='f(t)', title="Data Points and Fitted Curve (File Upload)")
                fig.add_scatter(x=df_fit['t'], y=df_fit['f(t)'], mode='lines', name='Fit', line=dict(color='red'))
                
                st.plotly_chart(fig)
            else:
                st.error("The uploaded file must contain columns named 't' and 'f(t)'.")

        except Exception as e:
            st.error(f"Error processing file: {e}")


#========================================
# --- Option 1: Enter data manually ---
#========================================
elif selection == 1:
    
    # 1. Robust Initialization Check: Check if the session state key exists OR if it's not a DataFrame 
    # OR if it's an empty DataFrame. Re-initialize if any condition is true.
    if ('manual_df' not in st.session_state or 
        not isinstance(st.session_state.manual_df, pd.DataFrame) or 
        st.session_state.manual_df.empty):
        
        st.session_state['manual_df'] = pd.DataFrame(
            [
                {'t': 0.0, 'f(t)': 0.0}
            ]
        )

    st.write("### Data Input")
    
    # 2. Use st.data_editor with key and callback for persistence
    # The dataframe variable holds the currently visible data, which is passed to the analysis.
    dataframe = st.data_editor(
        st.session_state['manual_df'],
        num_rows="dynamic",
        key="data_editor", # Unique key for the editor widget
        on_change=update_manual_df, # Function to call when data changes
        column_config={
            't': st.column_config.NumberColumn(
                label='t',
                step=1e-16,
                format="%.8f",
            ),
            'f(t)': st.column_config.NumberColumn(
                label='f(t)',
                step=1e-16,
                format="%.8f",
            ),
        }
    )
    
    st.write("### Regression Results")
    
    # Check if the DataFrame has any rows after editing
    if dataframe.empty:
        st.info("The data editor is empty. Please enter data points.")
    
    # Ensure there are at least 4 data points (for 4 parameters) before attempting curve fitting
    elif len(dataframe) >= 4:
        
        # (Assuming xData and yData are defined as numpy arrays)
        xData = dataframe['t'].to_numpy()
        yData = dataframe['f(t)'].to_numpy()
        
        # 4. Use curve_fit to find optimal parameters
        try:
            # Initial guesses: use statistics from the input data
            #initial_guesses = [1.0, xData.mean(), (yData.max() - yData.min()) / 2, yData.mean()] 
            fitted_params, pcov = curve_fit(fit_function, xData, yData, p0=initial_guesses['Value'].to_numpy())
            
            # 5. Get predictions and evaluate
            # Display fit parameters and corresponding uncertainties
            st.write("Fitted Parameters ($A$, $\\tau$, $\\omega$, $\\phi$):")
            st.dataframe(pd.DataFrame({"Parameter": fitted_params,
                                       "Uncertainty": np.sqrt(np.diag(pcov))}),
                         hide_index = True)
            
            # Create smooth X values for a smooth curve plot
            x_fit = np.linspace(xData.min(), xData.max(), 500)
            model_predictions = fit_function(x_fit, *fitted_params)
            
            # Prepare data for plotting
            df_fit = pd.DataFrame({'t': x_fit, 'f(t)': model_predictions, 'Type': 'Fit Curve'})
            df_data = pd.DataFrame({'t': xData, 'f(t)': yData, 'Type': 'Data Points'})
            
            # Print r^2
            r2 = get_r_squared(yData, fit_function(xData, *fitted_params))
            st.write(rf'''Goodness-of-fit $r^2$ = {r2}''')
            
            # Create the scatter plot for data and line plot for the fit
            fig = px.scatter(df_data, x='t', y='f(t)', title="Data Points and Fitted Curve (Manual Input)")
            fig.add_scatter(x=df_fit['t'], y=df_fit['f(t)'], mode='lines', name='Fit', line=dict(color='red'))
            
            st.plotly_chart(fig)
            
        except RuntimeError:
            st.warning("Could not find optimal parameters for the fit function. This can happen if the data does not match the fit model well or if there isn't enough variance in the data.")
        except ValueError as e:
            st.error(f"Error during fitting (ValueError): {e}. This might be due to identical t or f(t) values.")


    elif len(dataframe) > 1:
        st.info(f"You currently have {len(dataframe)} data points. Input at least 4 points for the regression to run reliably (4 parameters: $A$, $\\tau$, $\\omega$, $\\phi$).")
        # Plot just the raw data for visual feedback
        fig = px.scatter(dataframe, x='t', y='f(t)', title="Data Points (Regression requires more data)")
        st.plotly_chart(fig)
    else:
        st.info("Please input at least two data points.")