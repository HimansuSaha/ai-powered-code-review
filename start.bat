@echo off
REM AI Code Review System - Quick Start Script for Windows
REM This script sets up and runs the complete AI-powered code review system

echo 🚀 AI Code Review System - Quick Start
echo ======================================

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo 📝 Creating environment configuration...
    copy .env.example .env
    echo ✅ Environment file created. Please review and update .env file if needed.
)

REM Build and start services
echo 🏗️  Building and starting services...
docker-compose up -d --build

echo ⏳ Waiting for services to be ready...
timeout /t 30 /nobreak >nul

REM Check service health
echo 🔍 Checking service health...

REM Check Backend
curl -f http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Backend is ready
) else (
    echo ❌ Backend is not ready yet
)

REM Check Frontend
curl -f http://localhost:3000 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Frontend is ready
) else (
    echo ❌ Frontend is not ready yet
)

echo.
echo 🎉 AI Code Review System is running!
echo ==================================
echo.
echo 📱 Access Points:
echo   • Frontend:     http://localhost:3000
echo   • Backend API:  http://localhost:8000
echo   • API Docs:     http://localhost:8000/docs
echo   • Grafana:      http://localhost:3001 (admin/admin123)
echo   • Prometheus:   http://localhost:9090
echo.
echo 🔧 Management Commands:
echo   • View logs:    docker-compose logs -f
echo   • Stop system:  docker-compose down
echo   • Restart:      docker-compose restart
echo.
echo 📖 Next Steps:
echo   1. Visit http://localhost:3000 to access the web interface
echo   2. Create an account or login
echo   3. Upload code for analysis
echo   4. View security and quality insights
echo.
echo 💡 For development setup, see README.md
echo.
echo Opening frontend in your default browser...
start http://localhost:3000

pause