"""
Stock data API routes
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from app.core.database import get_db
from app.models.stock import Stock, StockPrice, TechnicalIndicator
from app.schemas.stock import Stock as StockSchema, StockPrice as StockPriceSchema
from app.services.data_service import DataService

router = APIRouter()


@router.get("/", response_model=List[StockSchema])
async def get_stocks(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    active_only: bool = Query(True),
    db: Session = Depends(get_db)
):
    """Get list of stocks"""
    data_service = DataService(db)
    stocks = data_service.get_stocks(skip=skip, limit=limit, active_only=active_only)
    return stocks


@router.get("/{symbol}", response_model=StockSchema)
async def get_stock(symbol: str, db: Session = Depends(get_db)):
    """Get specific stock by symbol"""
    data_service = DataService(db)
    stock = data_service.get_stock_by_symbol(symbol.upper())
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")
    return stock


@router.get("/{symbol}/prices", response_model=List[StockPriceSchema])
async def get_stock_prices(
    symbol: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(1000, ge=1, le=10000),
    db: Session = Depends(get_db)
):
    """Get historical price data for a stock"""
    data_service = DataService(db)
    
    # Default to last 30 days if no dates provided
    if not start_date:
        start_date = datetime.now() - timedelta(days=30)
    if not end_date:
        end_date = datetime.now()
    
    prices = data_service.get_stock_prices(
        symbol=symbol.upper(),
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )
    return prices


@router.get("/{symbol}/indicators")
async def get_technical_indicators(
    symbol: str,
    indicator_name: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(1000, ge=1, le=10000),
    db: Session = Depends(get_db)
):
    """Get technical indicators for a stock"""
    data_service = DataService(db)
    
    if not start_date:
        start_date = datetime.now() - timedelta(days=30)
    if not end_date:
        end_date = datetime.now()
    
    indicators = data_service.get_technical_indicators(
        symbol=symbol.upper(),
        indicator_name=indicator_name,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )
    return indicators


@router.post("/{symbol}/refresh")
async def refresh_stock_data(
    symbol: str,
    db: Session = Depends(get_db)
):
    """Refresh stock data from external sources"""
    data_service = DataService(db)
    try:
        result = await data_service.refresh_stock_data(symbol.upper())
        return {"message": f"Data refreshed for {symbol}", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
