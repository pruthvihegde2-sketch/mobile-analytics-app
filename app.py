import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(page_title="Mobile Analytics Dashboard", layout="wide")

# App Header
st.title("📱 Mobile Price & Market Analytics Dashboard")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("mobile_data_clean.csv")

df = load_data()

# Navigation Tabs
tab1, tab2 = st.tabs(["📊 Analytics Results Overview", "🔮 Custom Prediction Model"])

# --- TAB 1: RESULTS OVERVIEW ---
with tab1:
    st.header("Project Analytics Results & Key Insights")
    
    st.subheader("1. Executed Machine Learning Models Summary")
    results_summary = """
    | Technique / Model | Primary Metric | Key Finding / Result |
    | :--- | :--- | :--- |
    | **Linear Regression** | R² = 0.6026 | Price is heavily driven by RAM and internal storage specs. |
    | **Decision Tree Classifier** | Accuracy = 60.17% | Feature thresholds clearly split high vs. low sales volume products. |
    | **K-Means Clustering** | 3 Segments | Segmented catalog into Budget, Mid-Tier, and High-End tiers. |
    | **Random Forest Classifier** | Accuracy = 53.00% | Ranked Battery Power & Price as top drivers of sales performance. |
    | **Factor Analysis** | 2 Latent Factors | Extracted 'Customer Perception' and 'Market Engagement' dimensions. |
    | **Logistic Regression** | Accuracy = 60.50% | Binary classification for high-sales product probability. |
    """
    st.markdown(results_summary)
    
    st.subheader("2. Core Strategic Recommendations")
    st.markdown("""
    1. **Optimize Pricing Tiers:** Align mid-tier specs closely with competitive market prices to balance volume and margin.
    2. **Highlight Battery & Ratings:** Focus marketing campaigns around battery longevity and customer satisfaction reviews.
    3. **Targeted Product Bundling:** Customize promotional offers based on K-Means customer segments (Budget vs. Premium).
    """)

# --- TAB 2: CUSTOM PREDICTION MODEL ---
with tab2:
    st.header("Custom Phone Performance Predictor")
    st.write("Drag the sliders below to adjust specifications and test model predictions:")

    # Back to sliders!
    ram_input = st.slider("RAM (MB)", min_value=512, max_value=16384, value=4096, step=512)
    battery_input = st.slider("Battery Power (mAh)", min_value=1000, max_value=10000, value=4500, step=100)
    memory_input = st.slider("Internal Memory (GB)", min_value=8, max_value=1024, value=128, step=8)
    price_input = st.slider("Actual Price (USD)", min_value=50.0, max_value=3000.0, value=500.0, step=25.0)

    if st.button("Run Prediction"):
        X = df[['RAM_MB', 'Battery_Power_mAh', 'Internal_Memory_GB', 'Actual_Price_USD']]
        y = df['High_Sales_Product'].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        user_data = pd.DataFrame([[ram_input, battery_input, memory_input, price_input]], columns=X.columns)
        prob = rf.predict_proba(user_data)[0][1]

        st.markdown("---")
        st.markdown(f"**Model Confidence Score (High Sales Probability):** `{prob * 100:.1f}%`")
        
        if prob >= 0.5:
            st.markdown("### 🌟 Result: **High Sales Potential Product!**")
            st.write("This configuration offers a strong value-to-performance ratio aligned with top market demands.")
        else:
            st.markdown("### ⚠️ Result: **Standard / Low Sales Volume Product**")
            st.write("Consider lowering the price or upgrading RAM/Battery specs to improve market competitiveness.")
