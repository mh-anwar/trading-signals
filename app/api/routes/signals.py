"""
Trading signals API routes
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from app.core.database import get_db
from app.schemas.stock import TradingSignal
from app.services.signal_service import SignalService

router = APIRouter()


@router.get("/", response_model=List[TradingSignal])
async def get_signals(
    symbol: Optional[str] = Query(None),
    signal_type: Optional[str] = Query(None),
    active_only: bool = Query(True),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get trading signals"""
    signal_service = SignalService(db)
    
    signals = signal_service.get_signals(
        symbol=symbol,
        signal_type=signal_type,
        active_only=active_only,
        limit=limit
    )
    return signals


@router.get("/{symbol}/latest")
async def get_latest_signal(
    symbol: str,
    db: Session = Depends(get_db)
):
    """Get latest signal for a stock"""
    signal_service = SignalService(db)
    
    signal = signal_service.get_latest_signal(symbol.upper())
    if not signal:
        raise HTTPException(status_code=404, detail="No signal found for this stock")
    return signal


@router.post("/{symbol}/generate")
async def generate_signal(
    symbol: str,
    db: Session = Depends(get_db)
):
    """Generate new trading signal for a stock"""
    signal_service = SignalService(db)
    
    try:
        signal = await signal_service.generate_signal(symbol.upper())
        return signal
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch/generate")
async def generate_batch_signals(
    symbols: List[str],
    db: Session = Depends(get_db)
):
    """Generate signals for multiple stocks"""
    signal_service = SignalService(db)
    
    try:
        signals = await signal_service.generate_batch_signals(symbols)
        return signals
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{signal_id}/deactivate")
async def deactivate_signal(
    signal_id: int,
    db: Session = Depends(get_db)
):
    """Deactivate a trading signal"""
    signal_service = SignalService(db)
    
    try:
        result = signal_service.deactivate_signal(signal_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
