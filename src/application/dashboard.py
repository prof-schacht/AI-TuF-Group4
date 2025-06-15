import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Household Power Consumption Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard title
st.title("Household Power Consumption Dashboard")
st.markdown("### Prediction and Analysis Tool")

# Sidebar
st.sidebar.header("Controls")

# Forecast horizon slider
forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (hours)",
    min_value=1,
    max_value=24,
    value=6,
    step=1,
    help="Select how many hours ahead to predict"
)

# Placeholder for model selection
model_option = st.sidebar.selectbox(
    "Select Model",
    ["LSTM", "GRU", "TCN", "Transformer"],
    index=0,
    help="Choose the model architecture for prediction"
)

# Placeholder for feature importance visualization toggle
show_feature_importance = st.sidebar.checkbox(
    "Show Feature Importance",
    value=True,
    help="Display integrated gradients heatmap"
)

# Main content area
st.header("Power Consumption Forecast")

# Placeholder for forecast visualization
with st.container():
    st.subheader(f"Forecast for Next {forecast_horizon} Hours")
    
    # Placeholder chart
    chart_data = pd.DataFrame(
        np.random.randn(24, 2),
        columns=['Actual', 'Predicted']
    )
    
    fig = px.line(
        chart_data, 
        labels={"index": "Time (hours)", "value": "Power Consumption (kW)"},
        title="Power Consumption Forecast"
    )
    fig.add_scatter(x=chart_data.index, y=chart_data['Actual'], mode='lines', name='Actual')
    fig.add_scatter(x=chart_data.index, y=chart_data['Predicted'], mode='lines', name='Predicted')
    
    st.plotly_chart(fig, use_container_width=True)

# Placeholder for feature importance visualization
if show_feature_importance:
    st.header("Feature Importance")
    st.markdown("Integrated Gradients Heatmap showing which time steps had the most influence on the prediction")
    
    # Placeholder heatmap
    fig, ax = plt.subplots(figsize=(10, 3))
    data = np.random.rand(1, 24)
    im = ax.imshow(data, cmap='viridis')
    ax.set_xlabel('Time Steps')
    ax.set_title('Integrated Gradients')
    fig.colorbar(im)
    
    st.pyplot(fig)

# Metrics section
st.header("Performance Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Mean Absolute Error", value="0.42 kW")

with col2:
    st.metric(label="Root Mean Square Error", value="0.65 kW")

with col3:
    st.metric(label="R² Score", value="0.78")

# About section
with st.expander("About this Dashboard"):
    st.markdown("""
    This dashboard visualizes household power consumption predictions based on the UCI Household Power Consumption dataset.
    
    Features:
    - Adjustable forecast horizon
    - Model performance metrics
    - Feature importance visualization using integrated gradients
    
    The model is optimized for edge deployment.
    """)

# Footer
st.markdown("---")
st.markdown("Household Power Consumption Prediction | AI-TuF-Group4")
