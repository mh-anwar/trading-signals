"""
AWS Lambda function for data collection
"""
import json
import boto3
import requests
import yfinance as yf
from datetime import datetime, timedelta
import os
from typing import Dict, Any
import structlog

# Configure logging
logger = structlog.get_logger()

# Initialize AWS clients
s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')

# Environment variables
S3_BUCKET = os.environ.get('S3_BUCKET', 'stock-trader-data')
API_BASE_URL = os.environ.get('API_BASE_URL', 'https://your-api-domain.com')


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Main Lambda handler for data collection"""
    try:
        # Get symbols from event or environment
        symbols = event.get('symbols', ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'])
        
        # Collect data for each symbol
        results = []
        for symbol in symbols:
            try:
                result = collect_stock_data(symbol)
                results.append(result)
                logger.info(f"Successfully collected data for {symbol}")
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {str(e)}")
                results.append({
                    'symbol': symbol,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Store results in S3
        store_results_in_s3(results)
        
        # Trigger signal generation if data collection was successful
        if any(r.get('status') == 'success' for r in results):
            trigger_signal_generation()
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Data collection completed',
                'results': results
            })
        }
        
    except Exception as e:
        logger.error(f"Error in data collection: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }


def collect_stock_data(symbol: str) -> Dict[str, Any]:
    """Collect stock data for a symbol"""
    try:
        # Download data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        
        # Get historical data (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Get current info
        info = ticker.info
        
        # Prepare data
        data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'prices': hist.to_dict('records'),
            'info': {
                'name': info.get('longName', symbol),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'current_price': hist['Close'].iloc[-1] if not hist.empty else None
            }
        }
        
        return {
            'symbol': symbol,
            'status': 'success',
            'data_points': len(hist),
            'last_price': data['info']['current_price'],
            'timestamp': data['timestamp']
        }
        
    except Exception as e:
        logger.error(f"Error collecting data for {symbol}: {str(e)}")
        raise e


def store_results_in_s3(results: list):
    """Store collection results in S3"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        key = f"data_collection/{timestamp}/results.json"
        
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(results, indent=2),
            ContentType='application/json'
        )
        
        logger.info(f"Results stored in S3: s3://{S3_BUCKET}/{key}")
        
    except Exception as e:
        logger.error(f"Error storing results in S3: {str(e)}")
        raise e


def trigger_signal_generation():
    """Trigger signal generation Lambda function"""
    try:
        response = lambda_client.invoke(
            FunctionName='stock-trader-signal-generator',
            InvocationType='Event',  # Async invocation
            Payload=json.dumps({
                'trigger': 'data_collection_complete',
                'timestamp': datetime.now().isoformat()
            })
        )
        
        logger.info("Signal generation triggered")
        
    except Exception as e:
        logger.error(f"Error triggering signal generation: {str(e)}")


def send_data_to_api(symbol: str, data: Dict[str, Any]):
    """Send collected data to the main API"""
    try:
        # This would send data to your FastAPI backend
        # For now, we'll just log it
        logger.info(f"Would send data for {symbol} to API")
        
    except Exception as e:
        logger.error(f"Error sending data to API: {str(e)}")
