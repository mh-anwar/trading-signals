#!/bin/bash

# Setup script for Stock Trader AI
echo "Setting up Stock Trader AI..."

# Check requirements
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required. Please install it first."
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "Docker is required. Please install it first."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Copy environment file
echo "Setting up environment..."
cp env.example .env

# Start database
echo "Starting database..."
docker-compose up -d postgres

# Wait for database
echo "Waiting for database..."
sleep 10

# Run migrations
echo "Setting up database..."
alembic upgrade head

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Train model in Google Colab"
echo "2. Start API: uvicorn app.main:app --reload"
echo "3. Test: curl http://localhost:8000/health"
