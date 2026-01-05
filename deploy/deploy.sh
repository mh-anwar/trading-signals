#!/bin/bash

# Stock Trader AI Setup Script

echo "Setting up Stock Trader AI..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install it first."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install it first."
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
echo "Environment file created. Edit .env with your settings."

# Start database
echo "Starting database..."
docker-compose up -d postgres

# Wait for database to be ready
echo "Waiting for database to start..."
sleep 10

# Run database migrations
echo "Setting up database tables..."
alembic upgrade head

echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Train your model in Google Colab (notebooks/training_pipeline.ipynb)"
echo "2. Download the model files and put them in the models/ folder"
echo "3. Start the API: uvicorn app.main:app --reload"
echo "4. Test it: curl http://localhost:8000/health"
