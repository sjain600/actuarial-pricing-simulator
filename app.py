pip install plotly
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import joblib

# Page configuration
st.set_page_config(
    page_title="Actuarial Pricing Simulator",
    page_icon="📊",
    layout="wide"
)
# Load models
@st.cache_resource
def load_models():
    with open('preprocessor3.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open('frequency_model3.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_info3.pkl', 'rb') as f:
        feature_info = pickle.load(f)
    return preprocessor, model, feature_info

try:
    preprocessor, model, feature_info = load_models()
    st.success("✅ Models loaded successfully!")
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please ensure the app is in the correct directory.")
    st.stop()

# Sidebar for user input
st.sidebar.header("Policyholder Characteristics")

# Define input fields based on dataset
# Numerical inputs
st.sidebar.subheader("Numerical Features")
driv_age = st.sidebar.slider(
    "Driver Age (years)", 
    min_value=18, max_value=95, value=45, step=1,
    help="Age of the primary driver"
)

veh_age = st.sidebar.slider(
    "Vehicle Age (years)", 
    min_value=0, max_value=40, value=5, step=1,
    help="Age of the vehicle"
)

bonus_malus = st.sidebar.slider(
    "Bonus-Malus Coefficient", 
    min_value=50, max_value=350, value=100, step=5,
    help="100 is baseline. <100 = bonus (fewer claims), >100 = malus (more claims)"
)

density = st.sidebar.number_input(
    "Population Density (inhabitants/km²)", 
    min_value=1, max_value=30000, value=1000, step=100,
    help="Population density of the driver's city"
)

# Categorical inputs
st.sidebar.subheader("Categorical Features")

area = st.sidebar.selectbox(
    "Geographic Area",
    options=['A', 'B', 'C', 'D', 'E', 'F'],
    help="Area code (A=least urban, F=most urban)"
)

veh_brand = st.sidebar.selectbox(
    "Vehicle Brand",
    options=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B10', 'B11', 'B12', 'B13', 'B14'],
    help="Vehicle brand category"
)

veh_gas = st.sidebar.radio(
    "Fuel Type",
    options=['Regular', 'Diesel'],
    horizontal=True
)

region = st.sidebar.selectbox(
    "Region",
    options=['R11', 'R21', 'R22', 'R23', 'R24', 'R25', 'R31', 'R41', 'R42', 'R43', 'R52', 'R53', 
             'R72', 'R73', 'R74', 'R81', 'R82', 'R83', 'R91', 'R93', 'R94'],
    help="French region code"
)

veh_power = st.sidebar.selectbox(
    "Vehicle Power",
    options=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    index=4,
    help="Engine power category"
)

# Exposure (for prediction scaling)
st.sidebar.subheader("Policy Details")
exposure = st.sidebar.number_input(
    "Policy Duration (years)", 
    min_value=0.1, max_value=1.0, value=1.0, step=0.1,
    help="Length of coverage in years"
)

# Prediction button
predict_button = st.sidebar.button("Predict Claim Frequency", type="primary")

# Main area for results
if predict_button:
    # Create input dataframe
    # First, derive engineered features
    if veh_age <= 2:
        veh_age_binned = '0-2 years'
    elif veh_age <= 5:
        veh_age_binned = '3-5 years'
    elif veh_age <= 10:
        veh_age_binned = '6-10 years'
    else:
        veh_age_binned = '10+ years'
    
    if driv_age <= 25:
        driv_age_binned = '18-25'
    elif driv_age <= 35:
        driv_age_binned = '26-35'
    elif driv_age <= 50:
        driv_age_binned = '36-50'
    elif driv_age <= 65:
        driv_age_binned = '51-65'
    else:
        driv_age_binned = '65+'
    
    if veh_power <= 6:
        veh_power_group = 'Low (<6)'
    elif veh_power <= 8:
        veh_power_group = 'Medium (6-7)'
    elif veh_power <= 10:
        veh_power_group = 'High (8-9)'
    else:
        veh_power_group = 'Very High (10+)'
    
    log_density = np.log1p(density)
    bonus_malus_deviation = bonus_malus - 100
    
    # Create input dataframe
    input_data = pd.DataFrame([{
        'Area': area,
        'VehBrand': veh_brand,
        'VehGas': veh_gas,
        'Region': region,
        'VehAge_binned': veh_age_binned,
        'DrivAge_binned': driv_age_binned,
        'VehPower_group': veh_power_group,
        'VehAge': veh_age,
        'DrivAge': driv_age,
        'BonusMalus': bonus_malus,
        'Log_Density': log_density,
        'BonusMalus_deviation': bonus_malus_deviation
    }])
    
    input_processed = preprocessor.transform(input_data)
    
    # Make prediction (frequency per year)
    pred_freq = model.predict(input_processed)[0]
    
    # Scale by exposure
    expected_claims = pred_freq * exposure
    
    # Display results in a nice format
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Expected Claim Frequency",
            value=f"{pred_freq:.3f}",
            delta=None,
            help="Claims per year of exposure"
        )
    
    with col2:
        st.metric(
            label="Expected Claims",
            value=f"{expected_claims:.3f}",
            delta=None,
            help=f"Expected number of claims over {exposure} year(s)"
        )
    
    with col3:
        # Compare to average
        avg_freq = feature_info.get('baseline_rate', 0.1)
        pct_diff = ((pred_freq - avg_freq) / avg_freq) * 100
        st.metric(
            label="vs. Population Average",
            value=f"{pct_diff:+.1f}%",
            delta=f"{'Above' if pct_diff > 0 else 'Below'} average",
            delta_color="inverse" if pct_diff > 0 else "normal"
        )
    
    # Risk assessment
    st.subheader("Risk Assessment")
    if pred_freq < 0.05:
        risk_level = "Low Risk"
        color = "green"
        icon = "🟢"
    elif pred_freq < 0.10:
        risk_level = "Moderate Risk"
        color = "orange"
        icon = "🟠"
    else:
        risk_level = "High Risk"
        color = "red"
        icon = "🔴"
    
    st.markdown(f"### {icon} {risk_level}")
    
    # Feature importance visualization
    st.subheader("Factor Contributions")
    
    # Create a simple bar chart of feature values
    feature_contributions = pd.DataFrame({
        'Feature': ['Driver Age', 'Vehicle Age', 'Bonus-Malus', 'Density', 'Area', 'Fuel Type'],
        'Value': [driv_age, veh_age, bonus_malus, density, area, veh_gas],
        'Impact': ['Moderate' if 30 < driv_age < 70 else 'High' if driv_age < 25 else 'Moderate',
                   'Low' if veh_age < 5 else 'Moderate' if veh_age < 15 else 'High',
                   'High' if bonus_malus > 120 else 'Moderate' if bonus_malus > 105 else 'Low',
                   'Moderate' if density > 500 else 'Low',
                   'High' if area in ['E', 'F'] else 'Moderate' if area in ['C', 'D'] else 'Low',
                   'Moderate' if veh_gas == 'D' else 'Low']
    })
    
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Feature', 'Value', 'Risk Impact'],
                   fill_color='paleturquoise',
                   align='left'),
        cells=dict(values=[feature_contributions.Feature, 
                          feature_contributions.Value,
                          feature_contributions.Impact],
                  fill_color='lavender',
                  align='left'))
    ])
    
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    # Explanation
    with st.expander("How is this calculated?"):
        st.markdown("""
        The prediction uses a **Poisson regression model** with a log link function:
        
        `log(claims) = log(exposure) + β₀ + β₁x₁ + ... + βₖxₖ`
        
        **Interpretation:**
        - Each coefficient βᵢ represents the log of the multiplicative effect on claim rate
        - exp(βᵢ) > 1 means the factor increases claim risk
        - exp(βᵢ) < 1 means the factor decreases claim risk
        
        The model was trained on 678,013 French motor insurance policies and validated 
        using 5-fold cross-validation.
        """)

else:
    # Show placeholder when no prediction yet
    st.info("👈 Adjust the parameters in the sidebar and click 'Predict' to see results")
    
    # Show sample visualization of the data
    st.subheader("About the Dataset")
    st.markdown("""
    The model is trained on the **freMTPL2freq dataset**, which contains:
    
    - **678,013** motor insurance policies
    - **11** policyholder and vehicle characteristics
    - **Claim counts** during the exposure period
    - **Exposure** in years (policy duration)
    
    The dataset represents a realistic portfolio of a French motor insurer.
    """)
    
    # Show a sample distribution
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=np.random.gamma(shape=0.5, scale=0.1, size=1000),
        nbinsx=50,
        name='Sample Distribution'
    ))
    fig.update_layout(
        title='Typical Claim Frequency Distribution',
        xaxis_title='Claims per Year',
        yaxis_title='Density',
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Python, scikit-learn, and Streamlit | Data: French Motor Third-Party Liability Claims</p>
</div>
""", unsafe_allow_html=True)
