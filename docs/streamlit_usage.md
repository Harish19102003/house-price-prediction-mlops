# Streamlit House Price Predictor App

## 🚀 Quick Start

### Option 1: Using the run script
```bash
python run_streamlit.py
```

### Option 2: Direct Streamlit command
```bash
streamlit run src/streamlit_app.py
```

### Option 3: From project root
```bash
python -m streamlit run src/streamlit_app.py
```

## 🌐 Access the App

Once started, the app will be available at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://your-ip:8501

## 🏠 Features

### Input Form
The app provides an intuitive form with the most important house features:

- **Overall Quality** (1-10): Rates the overall material and finish
- **Above Ground Living Area**: Square footage of living space
- **Total Basement Area**: Square footage of basement
- **Garage Cars Capacity**: Number of cars the garage can hold
- **Year Built**: Original construction date
- **Full Bathrooms**: Number of full bathrooms
- **Kitchen Quality**: Quality rating (Ex/Gd/TA/Fa/Po)
- **Neighborhood**: Location within Ames city limits

### Model Information
The sidebar displays:
- Model type and performance metrics
- R² score, RMSE, MAE, and MAPE
- Overall performance rating
- Number of test samples used for evaluation

### Visualizations
- **Performance Chart**: Interactive gauge showing R² score
- **Error Metrics**: Bar chart comparing MAE and RMSE
- **Model Details**: Algorithm information and interpretation

### Prediction Results
- **Prominent Display**: Large, styled prediction result
- **Formatted Price**: Dollar amount with proper formatting
- **Auto-save**: Predictions are automatically saved to `docs/streamlit_predictions.json`

## 📊 Model Performance

The app uses the best trained model with:
- **R² Score**: ~0.907 (Excellent performance)
- **RMSE**: ~$26,712 (Good prediction accuracy)
- **MAPE**: ~10.1% (Moderate accuracy)

## 💾 Data Storage

All predictions made through the app are saved to:
- `docs/streamlit_predictions.json`: JSON file with timestamp, input features, and predictions

## 🎨 UI Features

- **Responsive Design**: Works on desktop and mobile
- **Clean Styling**: Professional gradient backgrounds and cards
- **Interactive Elements**: Hover tooltips and help text
- **Real-time Updates**: Instant predictions with loading indicators

## 🔧 Technical Details

- **Framework**: Streamlit 1.44.1
- **Visualizations**: Plotly for interactive charts
- **Model Loading**: Cached for performance
- **Error Handling**: Graceful error messages and fallbacks

## 🐛 Troubleshooting

### Common Issues

1. **Module not found errors**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Port already in use**:
   ```bash
   streamlit run src/streamlit_app.py --server.port 8502
   ```

3. **Model not found**:
   Ensure you've run the training pipeline first:
   ```bash
   python src/model_training.py
   ```

### Logs
Check the terminal output for detailed error messages and debugging information.

## 📱 Mobile Support

The app is fully responsive and works well on:
- Desktop browsers (Chrome, Firefox, Safari, Edge)
- Mobile browsers (iOS Safari, Chrome Mobile)
- Tablet browsers

## 🔒 Security Notes

- The app runs locally by default
- No data is sent to external servers
- All predictions are stored locally in the `docs/` directory
- Model files remain in the local `models/` directory 