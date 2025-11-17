#!/bin/bash

# AI Code Review System - Quick Start Script
# This script sets up and runs the complete AI-powered code review system

set -e

echo "🚀 AI Code Review System - Quick Start"
echo "======================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating environment configuration..."
    cp .env.example .env
    echo "✅ Environment file created. Please review and update .env file if needed."
fi

# Build and start services
echo "🏗️  Building and starting services..."
docker-compose up -d --build

echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check PostgreSQL
if docker-compose exec -T postgres pg_isready -U postgres; then
    echo "✅ PostgreSQL is ready"
else
    echo "❌ PostgreSQL is not ready"
fi

# Check Redis
if docker-compose exec -T redis redis-cli ping; then
    echo "✅ Redis is ready"
else
    echo "❌ Redis is not ready"
fi

# Check Backend
if curl -f http://localhost:8000/health 2>/dev/null; then
    echo "✅ Backend is ready"
else
    echo "❌ Backend is not ready"
fi

# Check Frontend
if curl -f http://localhost:3000 2>/dev/null; then
    echo "✅ Frontend is ready"
else
    echo "❌ Frontend is not ready"
fi

echo ""
echo "🎉 AI Code Review System is running!"
echo "=================================="
echo ""
echo "📱 Access Points:"
echo "  • Frontend:     http://localhost:3000"
echo "  • Backend API:  http://localhost:8000"
echo "  • API Docs:     http://localhost:8000/docs"
echo "  • Grafana:      http://localhost:3001 (admin/admin123)"
echo "  • Prometheus:   http://localhost:9090"
echo ""
echo "🔧 Management Commands:"
echo "  • View logs:    docker-compose logs -f"
echo "  • Stop system:  docker-compose down"
echo "  • Restart:      docker-compose restart"
echo ""
echo "📖 Next Steps:"
echo "  1. Visit http://localhost:3000 to access the web interface"
echo "  2. Create an account or login"
echo "  3. Upload code for analysis"
echo "  4. View security and quality insights"
echo ""
echo "💡 For development setup, see README.md"
echo ""