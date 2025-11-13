# Product Return Prediction - Web Application

A modern, interactive two-page web application for predicting product returns with a clean, professional interface.

## Features

- **Dashboard Page**: View statistics and prediction history
- **Prediction Calculator**: Make predictions using the trained ML model
- **Modern UI**: Clean, minimal design with pastel color palette
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Local Storage**: Predictions persist in browser storage
- **API Integration**: Connects to Flask backend for real-time predictions

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

The web application requires the Flask API server to be running for predictions:

```bash
python api_server.py
```

The API will start on `http://127.0.0.1:5000`

### 3. Open the Web Application

Simply open `index.html` in your web browser. You can:

- **Option A**: Double-click `index.html` to open in your default browser
- **Option B**: Use a local server (recommended for development):
  ```bash
  # Python 3
  python -m http.server 8000
  
  # Then open: http://localhost:8000/index.html
  ```

## File Structure

```
├── index.html          # Dashboard/Home page
├── predict.html        # Prediction calculator page
├── styles.css          # Shared stylesheet
├── script.js           # JavaScript functionality
├── api_server.py       # Flask API server
└── README_WEB.md       # This file
```

## Usage

1. **Start the API server** (required for predictions):
   ```bash
   python api_server.py
   ```

2. **Open the web application**:
   - Open `index.html` in your browser
   - Or use a local server: `python -m http.server 8000`

3. **Make Predictions**:
   - Click "Predict" in the navigation
   - Fill out the prediction form
   - Click "Predict" button
   - View results and see them saved to the dashboard

4. **View Dashboard**:
   - See total predictions, successful returns, and average accuracy
   - Browse prediction history in the table

## API Endpoints

- `POST /api/predict` - Make a prediction
- `GET /api/health` - Check API health status

## Notes

- The web app will fall back to mock predictions if the API server is not running
- Predictions are stored in browser localStorage
- The API server must be running for real model predictions
- Make sure `artifacts/New_model.pkl` exists before starting the API server

## Troubleshooting

**API not connecting?**
- Ensure `api_server.py` is running
- Check that the API URL in `script.js` matches your server address
- Verify the model file exists at `artifacts/New_model.pkl`

**Predictions not working?**
- Check browser console for errors
- Verify all form fields are filled
- Ensure API server is running and accessible

