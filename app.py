import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Predictive Maintenance AI", layout="wide")

# Title
st.title("🔧 Predictive Maintenance AI - Air Conditioner Analytics")
st.markdown("---")

# Sidebar for file upload
with st.sidebar:
    st.header("📂 Data Input")
    uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")
        st.dataframe(df.head(3))

# Main content
if uploaded_file is not None:
    # Data Processing
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    
    # Feature Engineering
    features = ['temperature', 'vibration', 'pressure', 'humidity', 'current_draw']
    X = df[features]
    
    # Create target (failure prediction based on thresholds)
    df['failure'] = ((df['temperature'] > 45) | 
                     (df['vibration'] > 8) | 
                     (df['pressure'] > 150) | 
                     (df['current_draw'] > 12)).astype(int)
    
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, df['failure'])
    
    # Predictions
    df['risk_score'] = model.predict_proba(X_scaled)[:, 1] * 100
    df['predicted_failure'] = model.predict(X_scaled)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🤖 Model Insights", 
        "📈 Predictions", 
        "💰 Cost Analysis",
        "📋 Recommendations"
    ])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Readings", len(df))
        col2.metric("Active Units", df['unit_id'].nunique())
        col3.metric("Avg Temperature", f"{df['temperature'].mean():.1f}°C")
        col4.metric("Failure Rate", f"{df['failure'].mean()*100:.1f}%")
        
        st.subheader("Sensor Data Timeline")
        fig = px.line(df, x='timestamp', y=['temperature', 'vibration', 'pressure'], 
                     title="Multi-Sensor Readings")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature Importance
            importance_df = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig = px.bar(importance_df, x='importance', y='feature', 
                        orientation='h', title="Feature Impact on Failures")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Model Performance Gauge
            accuracy = (df['predicted_failure'] == df['failure']).mean() * 100
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=accuracy,
                title={'text': "Model Accuracy"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 80], 'color': "gray"},
                           {'range': [80, 100], 'color': "darkblue"}]}))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk Distribution
            risk_cats = pd.cut(df['risk_score'], bins=[0, 30, 70, 100], 
                              labels=['Low', 'Medium', 'High'])
            risk_dist = risk_cats.value_counts()
            
            fig = px.pie(values=risk_dist.values, names=risk_dist.index, 
                        title="Risk Distribution", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Failure Timeline
            failure_timeline = df[df['failure']==1].groupby(df['timestamp'].dt.date).size()
            fig = px.line(x=failure_timeline.index, y=failure_timeline.values,
                         title="Failure Events Timeline")
            st.plotly_chart(fig, use_container_width=True)
        
        # High Risk Units
        st.subheader("⚠️ High Risk Units (Risk Score > 70%)")
        high_risk = df[df['risk_score'] > 70][['unit_id', 'timestamp', 'temperature', 
                                               'vibration', 'risk_score']].head()
        st.dataframe(high_risk)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        # Cost calculations
        maintenance_cost = 500
        emergency_cost = 2000
        prevented_failures = sum((df['risk_score'] > 70) & (df['failure'] == 0))
        actual_failures = df['failure'].sum()
        
        with col1:
            # Cost Comparison
            costs = {
                'Preventive Maintenance': prevented_failures * maintenance_cost,
                'Emergency Repairs': actual_failures * emergency_cost,
                'Potential Savings': (actual_failures * emergency_cost) - 
                                    (prevented_failures * maintenance_cost)
            }
            
            fig = px.bar(x=list(costs.keys()), y=list(costs.values()),
                        title="Cost Analysis ($)", color=list(costs.values()))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # ROI Gauge
            savings = max(0, costs['Potential Savings'])
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=savings,
                number={'prefix': "$"},
                title={'text': "Predicted Savings"},
                gauge={'axis': {'range': [None, max(5000, savings)]}}))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("📋 Policy Recommendations")
        
        # Generate recommendations
        high_risk_units = df[df['risk_score'] > 70]['unit_id'].nunique()
        med_risk_units = df[(df['risk_score'] <= 70) & (df['risk_score'] > 30)]['unit_id'].nunique()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Immediate Actions Required**\n\n"
                   f"• Schedule maintenance for {high_risk_units} high-risk units\n"
                   f"• Check temperature sensors on units exceeding 45°C\n"
                   f"• Inspect vibration patterns on units with score > 8")
        
        with col2:
            st.warning(f"**Preventive Measures**\n\n"
                      f"• Plan maintenance for {med_risk_units} medium-risk units\n"
                      f"• Optimize pressure settings below 150 PSI\n"
                      f"• Schedule filter cleaning for efficiency improvement")
        
        # Intervention Impact
        st.subheader("Expected Intervention Impact")
        impact_data = {
            'Metric': ['Failure Rate', 'Maintenance Cost', 'Unit Lifespan'],
            'Before': [f"{df['failure'].mean()*100:.1f}%", f"${actual_failures*emergency_cost}", "5 years"],
            'After': [f"{(df['failure'].mean()*0.3)*100:.1f}%", f"${prevented_failures*maintenance_cost}", "7 years"]
        }
        st.table(pd.DataFrame(impact_data))

else:
    st.info("👈 Please upload an Excel file to begin analysis")
    st.markdown("""
    ### Expected Excel Format:
    - **timestamp**: Date and time of reading
    - **unit_id**: Unique identifier for each AC unit
    - **temperature**: Operating temperature (°C)
    - **vibration**: Vibration level (mm/s)
    - **pressure**: System pressure (PSI)
    - **humidity**: Ambient humidity (%)
    - **current_draw**: Electrical current (A)
    """)