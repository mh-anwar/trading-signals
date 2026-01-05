# Stock Trader AI - Setup Guide

## Quick Start

### 1. Train Your Model
1. Open `notebooks/training_pipeline.ipynb` in Google Colab
2. Run all cells (takes 10-15 minutes)
3. Download the model files: `lstm_model.pth` and `target_scaler.pkl`

### 2. Set Up Locally
```bash
pip install -r requirements.txt
cp env.example .env
docker-compose up -d postgres
alembic upgrade head
uvicorn app.main:app --reload
```

### 3. Test It Works
```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/predictions/ -X POST \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "horizon": 5}'
```

## Configuration

Edit `.env` file:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/stocktrader
DEBUG=True
```

## API Endpoints

- `GET /health` - Health check
- `GET /api/v1/stocks/` - List stocks
- `POST /api/v1/predictions/` - Get predictions
- `GET /api/v1/signals/` - Get trading signals

## Troubleshooting

**Database won't start?**
```bash
docker-compose down
docker-compose up -d postgres
```

**API won't start?**
```bash
docker-compose ps
docker-compose logs postgres
```

**Model not found?**
- Train the model in Google Colab
- Download model files to local machine
- Put them in the `models/` folder
