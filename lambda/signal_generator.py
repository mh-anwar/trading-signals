"""
AWS Lambda function for trading signal generation
"""
import json
import boto3
import requests
from datetime import datetime, timedelta
import os
from typing import Dict, Any, List
import structlog

# Configure logging
logger = structlog.get_logger()

# Initialize AWS clients
s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')

# Environment variables
S3_BUCKET = os.environ.get('S3_BUCKET', 'stock-trader-data')
API_BASE_URL = os.environ.get('API_BASE_URL', 'https://your-api-domain.com')
MODEL_S3_KEY = os.environ.get('MODEL_S3_KEY', 'models/lstm_model.pth')


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Main Lambda handler for signal generation"""
    try:
        # Get symbols to generate signals for
        symbols = event.get('symbols', ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'])
        
        # Generate signals for each symbol
        signals = []
        for symbol in symbols:
            try:
                signal = generate_trading_signal(symbol)
                if signal:
                    signals.append(signal)
                    logger.info(f"Generated signal for {symbol}: {signal['signal_type']}")
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {str(e)}")
        
        # Store signals
        if signals:
            store_signals_in_s3(signals)
            send_signals_to_api(signals)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Signal generation completed',
                'signals_generated': len(signals),
                'signals': signals
            })
        }
        
    except Exception as e:
        logger.error(f"Error in signal generation: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }


def generate_trading_signal(symbol: str) -> Dict[str, Any]:
    """Generate trading signal for a symbol"""
    try:
        # Get latest data from API
        data = get_latest_data(symbol)
        if not data:
            logger.warning(f"No data available for {symbol}")
            return None
        
        # Get prediction from model
        prediction = get_model_prediction(symbol, data)
        if not prediction:
            logger.warning(f"No prediction available for {symbol}")
            return None
        
        # Analyze prediction and generate signal
        signal = analyze_prediction(symbol, prediction, data)
        
        return signal
        
    except Exception as e:
        logger.error(f"Error generating signal for {symbol}: {str(e)}")
        return None


def get_latest_data(symbol: str) -> Dict[str, Any]:
    """Get latest stock data from API"""
    try:
        # In a real implementation, this would call your FastAPI backend
        # For now, we'll simulate getting data
        response = {
            'symbol': symbol,
            'current_price': 150.0,  # This would come from API
            'volume': 1000000,
            'timestamp': datetime.now().isoformat()
        }
        return response
        
    except Exception as e:
        logger.error(f"Error getting data for {symbol}: {str(e)}")
        return None


def get_model_prediction(symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Get model prediction for a symbol"""
    try:
        # In a real implementation, this would:
        # 1. Load the model from S3
        # 2. Prepare features
        # 3. Make prediction
        
        # For now, simulate a prediction
        prediction = {
            'symbol': symbol,
            'predicted_prices': [150.5, 151.2, 150.8, 152.1, 151.9],
            'confidence': 0.75,
            'trend': 'upward',
            'model_version': '1.0.0'
        }
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error getting prediction for {symbol}: {str(e)}")
        return None


def analyze_prediction(symbol: str, prediction: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze prediction and generate trading signal"""
    try:
        predicted_prices = prediction['predicted_prices']
        confidence = prediction['confidence']
        current_price = data['current_price']
        
        # Calculate price change
        price_change = (predicted_prices[-1] - current_price) / current_price
        price_change_pct = price_change * 100
        
        # Determine signal based on prediction and confidence
        if price_change_pct > 2 and confidence > 0.7:
            signal_type = 'BUY'
            price_target = max(predicted_prices)
            stop_loss = min(predicted_prices) * 0.95
            take_profit = max(predicted_prices) * 1.1
            reasoning = f"Strong upward trend predicted ({price_change_pct:.2f}% increase) with high confidence"
        elif price_change_pct < -2 and confidence > 0.7:
            signal_type = 'SELL'
            price_target = min(predicted_prices)
            stop_loss = max(predicted_prices) * 1.05
            take_profit = min(predicted_prices) * 0.9
            reasoning = f"Strong downward trend predicted ({price_change_pct:.2f}% decrease) with high confidence"
        else:
            signal_type = 'HOLD'
            price_target = predicted_prices[0]
            stop_loss = None
            take_profit = None
            reasoning = f"Uncertain trend ({price_change_pct:.2f}% change) with confidence {confidence:.2f}"
        
        signal = {
            'symbol': symbol,
            'signal_type': signal_type,
            'confidence': confidence,
            'current_price': current_price,
            'price_target': price_target,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reasoning': reasoning,
            'model_version': prediction['model_version'],
            'generated_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        return signal
        
    except Exception as e:
        logger.error(f"Error analyzing prediction for {symbol}: {str(e)}")
        return None


def store_signals_in_s3(signals: List[Dict[str, Any]]):
    """Store signals in S3"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        key = f"signals/{timestamp}/signals.json"
        
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(signals, indent=2),
            ContentType='application/json'
        )
        
        logger.info(f"Signals stored in S3: s3://{S3_BUCKET}/{key}")
        
    except Exception as e:
        logger.error(f"Error storing signals in S3: {str(e)}")


def send_signals_to_api(signals: List[Dict[str, Any]]):
    """Send signals to the main API"""
    try:
        # In a real implementation, this would send signals to your FastAPI backend
        for signal in signals:
            logger.info(f"Would send signal to API: {signal['symbol']} - {signal['signal_type']}")
        
    except Exception as e:
        logger.error(f"Error sending signals to API: {str(e)}")


def send_alerts(signals: List[Dict[str, Any]]):
    """Send alerts for high-confidence signals"""
    try:
        high_confidence_signals = [s for s in signals if s['confidence'] > 0.8]
        
        if high_confidence_signals:
            # In a real implementation, this would send alerts via:
            # - Email (SES)
            # - SMS (SNS)
            # - Slack/Discord webhooks
            # - Push notifications
            
            logger.info(f"Would send alerts for {len(high_confidence_signals)} high-confidence signals")
        
    except Exception as e:
        logger.error(f"Error sending alerts: {str(e)}")
