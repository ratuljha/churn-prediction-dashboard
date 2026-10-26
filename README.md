# Customer Churn Analytics & Risk Dashboard

An interactive web application built in Python with Streamlit and Plotly, serving real-time churn risk predictions from an optimized XGBoost classifier.

## Overview

This dashboard bridges machine learning models with business stakeholders. Retention and relationship managers can:
- Input customer attributes (tenure, contract type, monthly charges, services) to predict individual churn probability in real time
- Inspect top risk factors with visual indicators
- Explore high-risk customer segments and cohort patterns across the entire subscriber base
- Download filtered customer segments for proactive retention outreach

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io) *(Deployable on Streamlit Community Cloud with 1 click)*

## Tech Stack

- **Python 3.8+**
- **Streamlit** for the responsive web application
- **Plotly Express & Graph Objects** for interactive charts
- **XGBoost & Scikit-learn** for model inference
- **Pandas & NumPy** for data manipulation

## Features

1. **Real-Time Customer Risk Scoring:**
   - Sliders and select boxes for instant customer profiling
   - Gauge visualization of predicted churn probability
   - Dynamic risk categorization (Low: <30%, Moderate: 30-60%, High: >60%)
   - Tailored actionable retention recommendations

2. **Cohort & Segment Analytics:**
   - Churn rate comparison across contract types (Month-to-month vs 1-Year vs 2-Year)
   - Monthly billing distributions between churned and retained subscribers
   - Payment method impact analysis

3. **Data Explorer & Batch Filter:**
   - Searchable, sortable subscriber table
   - Filter by contract, internet type, and churn status
   - One-click CSV export for CRM integration

## Local Installation & Setup

```bash
# Clone the repository
git clone https://github.com/ratuljha/churn-prediction-dashboard.git
cd churn-prediction-dashboard

# Install requirements
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## Deployment to Streamlit Cloud (Free)

1. Push this repository to your GitHub account (`ratuljha/churn-prediction-dashboard`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Select your repo and branch (`main`), set the main file path to `app.py`
4. Click **Deploy** — your dashboard will be live on a public URL in 2 minutes!

## License

MIT License. Open for educational and portfolio demonstration.
