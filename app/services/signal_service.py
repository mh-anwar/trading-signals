"""
Trading signal generation service
"""
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import structlog

from app.models.stock import TradingSignal
from app.schemas.stock import TradingSignalCreate
from app.services.prediction_service import PredictionService

logger = structlog.get_logger()


class SignalService:
    """Service for generating trading signals"""
    
    def __init__(self, db: Session):
        self.db = db
        self.prediction_service = PredictionService(db)
    
    def get_signals(
        self,
        symbol: Optional[str] = None,
        signal_type: Optional[str] = None,
        active_only: bool = True,
        limit: int = 100
    ) -> List[TradingSignal]:
        """Get trading signals"""
        query = self.db.query(TradingSignal)
        
        if symbol:
            query = query.filter(TradingSignal.symbol == symbol)
        if signal_type:
            query = query.filter(TradingSignal.signal_type == signal_type)
        if active_only:
            query = query.filter(TradingSignal.is_active == True)
        
        return query.order_by(desc(TradingSignal.created_at)).limit(limit).all()
    
    def get_latest_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Get latest signal for a stock"""
        return (
            self.db.query(TradingSignal)
            .filter(
                and_(
                    TradingSignal.symbol == symbol,
                    TradingSignal.is_active == True
                )
            )
            .order_by(desc(TradingSignal.created_at))
            .first()
        )
    
    async def generate_signal(self, symbol: str) -> TradingSignal:
        """Generate trading signal for a stock"""
        try:
            # Get prediction
            prediction = await self.prediction_service.get_prediction(symbol, horizon=5)
            
            # Analyze prediction to generate signal
            signal_data = self._analyze_prediction(symbol, prediction)
            
            # Create signal record
            signal_create = TradingSignalCreate(**signal_data)
            signal = TradingSignal(**signal_create.dict())
            
            self.db.add(signal)
            self.db.commit()
            
            logger.info(f"Generated signal for {symbol}: {signal_data['signal_type']}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {str(e)}")
            self.db.rollback()
            raise e
    
    async def generate_batch_signals(self, symbols: List[str]) -> List[TradingSignal]:
        """Generate signals for multiple stocks"""
        signals = []
        for symbol in symbols:
            try:
                signal = await self.generate_signal(symbol)
                signals.append(signal)
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {str(e)}")
        return signals
    
    def deactivate_signal(self, signal_id: int) -> Dict[str, Any]:
        """Deactivate a trading signal"""
        signal = self.db.query(TradingSignal).filter(TradingSignal.id == signal_id).first()
        if not signal:
            raise ValueError(f"Signal {signal_id} not found")
        
        signal.is_active = False
        self.db.commit()
        
        return {"message": f"Signal {signal_id} deactivated", "signal_id": signal_id}
    
    def _analyze_prediction(self, symbol: str, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze prediction to generate trading signal"""
        predictions = prediction.get('predictions', [])
        confidence = prediction.get('confidence', 0.5)
        
        if not predictions:
            return {
                "symbol": symbol,
                "signal_type": "HOLD",
                "confidence": 0.0,
                "reasoning": "No prediction data available"
            }
        
        # Analyze price trend
        prices = [p['predicted_price'] for p in predictions]
        price_changes = [prices[i] - prices[0] for i in range(1, len(prices))]
        avg_change = sum(price_changes) / len(price_changes) if price_changes else 0
        change_percentage = (avg_change / prices[0]) * 100 if prices[0] > 0 else 0
        
        # Determine signal based on trend and confidence
        if change_percentage > 2 and confidence > 0.7:
            signal_type = "BUY"
            price_target = max(prices)
            stop_loss = min(prices) * 0.95
            take_profit = max(prices) * 1.1
            reasoning = f"Strong upward trend predicted ({change_percentage:.2f}% increase) with high confidence"
        elif change_percentage < -2 and confidence > 0.7:
            signal_type = "SELL"
            price_target = min(prices)
            stop_loss = max(prices) * 1.05
            take_profit = min(prices) * 0.9
            reasoning = f"Strong downward trend predicted ({change_percentage:.2f}% decrease) with high confidence"
        else:
            signal_type = "HOLD"
            price_target = prices[0]
            stop_loss = None
            take_profit = None
            reasoning = f"Uncertain trend ({change_percentage:.2f}% change) with confidence {confidence:.2f}"
        
        return {
            "symbol": symbol,
            "signal_type": signal_type,
            "confidence": confidence,
            "price_target": price_target,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reasoning": reasoning,
            "model_version": prediction.get('model_version', '1.0.0'),
            "expires_at": datetime.now() + timedelta(days=1)
        }
