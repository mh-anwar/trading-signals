"""
Data service for stock data management
"""
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from typing import List, Optional
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import structlog

from app.models.stock import Stock, StockPrice, TechnicalIndicator
from app.schemas.stock import StockCreate, StockPriceCreate, TechnicalIndicatorCreate

logger = structlog.get_logger()


class DataService:
    """Service for managing stock data"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_stocks(self, skip: int = 0, limit: int = 100, active_only: bool = True) -> List[Stock]:
        """Get list of stocks"""
        query = self.db.query(Stock)
        if active_only:
            query = query.filter(Stock.is_active == True)
        return query.offset(skip).limit(limit).all()
    
    def get_stock_by_symbol(self, symbol: str) -> Optional[Stock]:
        """Get stock by symbol"""
        return self.db.query(Stock).filter(Stock.symbol == symbol).first()
    
    def get_stock_prices(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime, 
        limit: int = 1000
    ) -> List[StockPrice]:
        """Get historical price data"""
        return (
            self.db.query(StockPrice)
            .filter(
                and_(
                    StockPrice.symbol == symbol,
                    StockPrice.date >= start_date,
                    StockPrice.date <= end_date
                )
            )
            .order_by(desc(StockPrice.date))
            .limit(limit)
            .all()
        )
    
    def get_technical_indicators(
        self,
        symbol: str,
        indicator_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[TechnicalIndicator]:
        """Get technical indicators"""
        query = self.db.query(TechnicalIndicator).filter(TechnicalIndicator.symbol == symbol)
        
        if indicator_name:
            query = query.filter(TechnicalIndicator.indicator_name == indicator_name)
        if start_date:
            query = query.filter(TechnicalIndicator.date >= start_date)
        if end_date:
            query = query.filter(TechnicalIndicator.date <= end_date)
        
        return query.order_by(desc(TechnicalIndicator.date)).limit(limit).all()
    
    async def refresh_stock_data(self, symbol: str) -> dict:
        """Refresh stock data from external sources"""
        try:
            # Get stock info from yfinance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Update or create stock record
            stock = self.get_stock_by_symbol(symbol)
            if not stock:
                stock_data = StockCreate(
                    symbol=symbol,
                    name=info.get('longName', symbol),
                    sector=info.get('sector'),
                    industry=info.get('industry'),
                    market_cap=info.get('marketCap')
                )
                stock = Stock(**stock_data.dict())
                self.db.add(stock)
            else:
                stock.name = info.get('longName', stock.name)
                stock.sector = info.get('sector', stock.sector)
                stock.industry = info.get('industry', stock.industry)
                stock.market_cap = info.get('marketCap', stock.market_cap)
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 1 year of data
            
            hist = ticker.history(start=start_date, end=end_date)
            
            # Store price data
            for date, row in hist.iterrows():
                price_data = StockPriceCreate(
                    symbol=symbol,
                    date=date,
                    open_price=float(row['Open']),
                    high_price=float(row['High']),
                    low_price=float(row['Low']),
                    close_price=float(row['Close']),
                    volume=int(row['Volume']),
                    adjusted_close=float(row['Close'])  # Assuming no adjustments for simplicity
                )
                
                # Check if price data already exists
                existing = self.db.query(StockPrice).filter(
                    and_(
                        StockPrice.symbol == symbol,
                        StockPrice.date == date
                    )
                ).first()
                
                if not existing:
                    price_record = StockPrice(**price_data.dict())
                    self.db.add(price_record)
            
            self.db.commit()
            
            # Calculate technical indicators
            await self._calculate_technical_indicators(symbol, hist)
            
            logger.info(f"Successfully refreshed data for {symbol}")
            return {"status": "success", "symbol": symbol, "records_added": len(hist)}
            
        except Exception as e:
            logger.error(f"Error refreshing data for {symbol}: {str(e)}")
            self.db.rollback()
            raise e
    
    async def _calculate_technical_indicators(self, symbol: str, hist_data: pd.DataFrame):
        """Calculate and store technical indicators"""
        try:
            import ta
            
            # Calculate various technical indicators
            indicators = {}
            
            # Moving averages
            indicators['SMA_20'] = ta.trend.sma_indicator(hist_data['Close'], window=20)
            indicators['SMA_50'] = ta.trend.sma_indicator(hist_data['Close'], window=50)
            indicators['EMA_12'] = ta.trend.ema_indicator(hist_data['Close'], window=12)
            indicators['EMA_26'] = ta.trend.ema_indicator(hist_data['Close'], window=26)
            
            # RSI
            indicators['RSI'] = ta.momentum.rsi(hist_data['Close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(hist_data['Close'])
            indicators['MACD'] = macd.macd()
            indicators['MACD_Signal'] = macd.macd_signal()
            indicators['MACD_Histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(hist_data['Close'])
            indicators['BB_Upper'] = bb.bollinger_hband()
            indicators['BB_Middle'] = bb.bollinger_mavg()
            indicators['BB_Lower'] = bb.bollinger_lband()
            
            # Store indicators in database
            for indicator_name, values in indicators.items():
                for date, value in values.items():
                    if pd.notna(value):
                        indicator_data = TechnicalIndicatorCreate(
                            symbol=symbol,
                            date=date,
                            indicator_name=indicator_name,
                            indicator_value=float(value)
                        )
                        
                        # Check if indicator already exists
                        existing = self.db.query(TechnicalIndicator).filter(
                            and_(
                                TechnicalIndicator.symbol == symbol,
                                TechnicalIndicator.date == date,
                                TechnicalIndicator.indicator_name == indicator_name
                            )
                        ).first()
                        
                        if not existing:
                            indicator_record = TechnicalIndicator(**indicator_data.dict())
                            self.db.add(indicator_record)
            
            self.db.commit()
            logger.info(f"Technical indicators calculated for {symbol}")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {str(e)}")
            raise e
