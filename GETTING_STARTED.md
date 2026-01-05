# Getting Started

## Quick Start

### Step 1: Train Your Model
1. Open `notebooks/training_pipeline.ipynb` in Google Colab
2. Run all cells (takes 10-15 minutes)
3. Download the model files when done

### Step 2: Set Up Locally
```bash
./setup.sh
```

### Step 3: Start the API
```bash
uvicorn app.main:app --reload
```

### Step 4: Test It Works
```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/predictions/ -X POST \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "horizon": 5}'
```

## What You Get

- Stock data from Yahoo Finance
- AI-powered price forecasts
- Technical analysis indicators
- REST API endpoints

## How It Works

1. Download stock prices and calculate indicators
2. Train LSTM model on historical patterns
3. Model forecasts future prices
4. REST API provides predictions

## Troubleshooting

**Database won't start?**
```bash
docker-compose down
docker-compose up -d postgres
```

**API won't start?**
- Check if database is running: `docker-compose ps`
- Check logs: `docker-compose logs postgres`

**Model not found?**
- Train the model in Google Colab
- Download model files to local machine
- Put them in the `models/` folder
