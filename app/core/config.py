"""
Application configuration settings
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/stocktrader"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "stocktrader"
    db_user: str = "user"
    db_password: str = "password"
    
    # AWS
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    aws_s3_bucket: Optional[str] = None
    
    # API Keys
    alpha_vantage_api_key: Optional[str] = None
    iex_cloud_api_key: Optional[str] = None
    
    # Application
    secret_key: str = "your-secret-key-change-in-production"
    debug: bool = True
    log_level: str = "INFO"
    
    # Model Configuration
    model_path: str = "./models/lstm_model.pth"
    batch_size: int = 32
    sequence_length: int = 60
    prediction_horizon: int = 5
    
    # Trading Configuration
    risk_tolerance: float = 0.02
    max_position_size: float = 0.1
    stop_loss_percentage: float = 0.05
    take_profit_percentage: float = 0.10
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
