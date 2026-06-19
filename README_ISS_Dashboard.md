# Real-Time ISS Pay-for-Performance Scoring Dashboard

This is a working Streamlit rebuild of the uploaded draft. It includes:

- Company compensation inputs
- Peer-group pay and TSR ranking
- Simplified RDA, MoM, and PTA-style tests
- Qualitative scoring factors
- Predicted Say-on-Pay vote support
- Predicted ISS recommendation
- Concern flags with recommendations
- Peer CSV upload and downloadable template

## Run locally

```bash
pip install -r requirements.txt
streamlit run iss_dashboard.py
```

## Peer CSV format

Required columns:

- `company`
- `total_pay`
- `tsr_3yr`

Optional columns:

- `tsr_1yr`
- `revenue_growth`
- `pay_trend_score`

## Important note

This app is a simplified analytical demo inspired by ISS pay-for-performance concepts. It is not an official ISS model and should not be used as a substitute for the current ISS policy guidance or company-specific proxy analysis.
