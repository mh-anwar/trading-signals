"""
Pydantic schemas for stock data validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class StockBase(BaseModel):
    """Base stock schema"""
    symbol: str = Field(..., max_length=10)
    name: str = Field(..., max_length=255)
    sector: Optional[str] = Field(None, max_length=100)
    industry: Optional[str] = Field(None, max_length=100)
    market_cap: Optional[float] = None
    is_active: bool = True


class StockCreate(StockBase):
    """Schema for creating a stock"""
    pass


class StockUpdate(BaseModel):
    """Schema for updating a stock"""
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    is_active: Optional[bool] = None


class Stock(StockBase):
    """Schema for stock response"""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class StockPriceBase(BaseModel):
    """Base stock price schema"""
    symbol: str = Field(..., max_length=10)
    date: datetime
    open_price: float = Field(..., gt=0)
    high_price: float = Field(..., gt=0)
    low_price: float = Field(..., gt=0)
    close_price: float = Field(..., gt=0)
    volume: int = Field(..., ge=0)
    adjusted_close: Optional[float] = None


class StockPriceCreate(StockPriceBase):
    """Schema for creating stock price data"""
    pass


class StockPrice(StockPriceBase):
    """Schema for stock price response"""
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class TechnicalIndicatorBase(BaseModel):
    """Base technical indicator schema"""
    symbol: str = Field(..., max_length=10)
    date: datetime
    indicator_name: str = Field(..., max_length=50)
    indicator_value: float
    parameters: Optional[str] = None


class TechnicalIndicatorCreate(TechnicalIndicatorBase):
    """Schema for creating technical indicator"""
    pass


class TechnicalIndicator(TechnicalIndicatorBase):
    """Schema for technical indicator response"""
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class TradingSignalBase(BaseModel):
    """Base trading signal schema"""
    symbol: str = Field(..., max_length=10)
    signal_type: str = Field(..., regex="^(BUY|SELL|HOLD)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: Optional[str] = None
    model_version: Optional[str] = None
    expires_at: Optional[datetime] = None


class TradingSignalCreate(TradingSignalBase):
    """Schema for creating trading signal"""
    pass


class TradingSignal(TradingSignalBase):
    """Schema for trading signal response"""
    id: int
    created_at: datetime
    is_active: bool
    
    class Config:
        from_attributes = True


class PredictionRequest(BaseModel):
    """Schema for prediction request"""
    symbol: str = Field(..., max_length=10)
    horizon: int = Field(5, ge=1, le=30)
    include_technical_indicators: bool = True


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    symbol: str
    predictions: List[dict]
    confidence: float
    model_version: str
    generated_at: datetime
