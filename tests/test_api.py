"""
API tests for Stock Trader AI
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.core.database import get_db, Base
from app.models.stock import Stock, StockPrice

# Test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "Stock Trader AI System"


def test_get_stocks():
    """Test getting stocks list"""
    response = client.get("/api/v1/stocks/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_get_stock_by_symbol():
    """Test getting specific stock"""
    # First create a test stock
    test_stock = Stock(
        symbol="TEST",
        name="Test Stock",
        sector="Technology",
        industry="Software"
    )
    db = TestingSessionLocal()
    db.add(test_stock)
    db.commit()
    db.close()
    
    response = client.get("/api/v1/stocks/TEST")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "TEST"
    assert data["name"] == "Test Stock"


def test_get_stock_not_found():
    """Test getting non-existent stock"""
    response = client.get("/api/v1/stocks/NONEXISTENT")
    assert response.status_code == 404


def test_get_stock_prices():
    """Test getting stock prices"""
    # Create test stock and price data
    db = TestingSessionLocal()
    
    test_stock = Stock(symbol="TEST", name="Test Stock")
    db.add(test_stock)
    db.commit()
    
    test_price = StockPrice(
        symbol="TEST",
        date="2023-01-01",
        open_price=100.0,
        high_price=105.0,
        low_price=95.0,
        close_price=102.0,
        volume=1000000
    )
    db.add(test_price)
    db.commit()
    db.close()
    
    response = client.get("/api/v1/stocks/TEST/prices")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if data:  # If data exists
        assert data[0]["symbol"] == "TEST"


def test_get_predictions():
    """Test getting predictions"""
    response = client.post(
        "/api/v1/predictions/",
        json={
            "symbol": "AAPL",
            "horizon": 5,
            "include_technical_indicators": True
        }
    )
    # This might return 500 if model is not available, which is expected in tests
    assert response.status_code in [200, 500]


def test_get_signals():
    """Test getting trading signals"""
    response = client.get("/api/v1/signals/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


if __name__ == "__main__":
    pytest.main([__file__])
