"""
Prediction API routes
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.schemas.stock import PredictionRequest, PredictionResponse
from app.services.prediction_service import PredictionService

router = APIRouter()


@router.post("/", response_model=PredictionResponse)
async def get_prediction(
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    """Get stock price predictions"""
    prediction_service = PredictionService(db)
    
    try:
        prediction = await prediction_service.get_prediction(
            symbol=request.symbol,
            horizon=request.horizon,
            include_technical_indicators=request.include_technical_indicators
        )
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}")
async def get_latest_prediction(
    symbol: str,
    horizon: int = 5,
    db: Session = Depends(get_db)
):
    """Get latest prediction for a stock"""
    prediction_service = PredictionService(db)
    
    try:
        prediction = await prediction_service.get_latest_prediction(
            symbol=symbol.upper(),
            horizon=horizon
        )
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def get_batch_predictions(
    symbols: List[str],
    horizon: int = 5,
    db: Session = Depends(get_db)
):
    """Get predictions for multiple stocks"""
    prediction_service = PredictionService(db)
    
    try:
        predictions = await prediction_service.get_batch_predictions(
            symbols=symbols,
            horizon=horizon
        )
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
