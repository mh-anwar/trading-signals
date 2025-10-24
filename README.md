# Stock Trader AI

A stock prediction system using PyTorch LSTM models and FastAPI.

## Features

- Downloads stock data from Yahoo Finance
- Trains LSTM model to predict stock prices
- Provides REST API for predictions
- PostgreSQL database for data storage
- Google Colab training notebook

## Quick Start

### 1. Train the Model
Open `notebooks/training_pipeline.ipynb` in Google Colab and run all cells.

### 2. Set Up Locally
```bash
pip install -r requirements.txt
cp env.example .env
docker-compose up -d postgres
alembic upgrade head
uvicorn app.main:app --reload
```

### 3. Test the API
```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/predictions/ -X POST \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "horizon": 5}'
```

## API Endpoints

- `GET /health` - Health check
- `GET /api/v1/stocks/` - List stocks
- `POST /api/v1/predictions/` - Get predictions
- `GET /api/v1/signals/` - Get trading signals

## Environment Variables

```env
DATABASE_URL=postgresql://user:password@localhost:5432/stocktrader
DEBUG=True
```

## License

MIT License
