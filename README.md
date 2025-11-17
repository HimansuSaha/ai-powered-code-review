# AI-Powered Code Review System

[![Build Status](https://github.com/yourusername/ai-powered-code-review/workflows/CI/badge.svg)](https://github.com/yourusername/ai-powered-code-review/actions)
[![Coverage Status](https://coveralls.io/repos/github/yourusername/ai-powered-code-review/badge.svg)](https://coveralls.io/github/yourusername/ai-powered-code-review)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![React 18](https://img.shields.io/badge/react-18-blue.svg)](https://reactjs.org/)

> **🚀 Production-ready AI-powered code review system with comprehensive security analysis, quality assessment, and automated insights**

A comprehensive, intelligent code review platform that leverages machine learning to automatically analyze code quality, detect security vulnerabilities, and provide actionable insights. Built with modern technologies including FastAPI, React, PostgreSQL, and custom ML models.

---

## 🌟 Key Features

### 🔐 **Advanced Security Analysis**
- **Vulnerability Detection**: Identifies SQL injection, XSS, command injection, and other OWASP Top 10 risks
- **ML-Powered Scanning**: Custom machine learning models trained on security datasets
- **Real-time Analysis**: Instant feedback on security issues as you code
- **SARIF Compliance**: Industry-standard security alert format

### 📊 **Comprehensive Quality Assessment**
- **Complexity Analysis**: Cyclomatic complexity, maintainability index calculation
- **Code Smells Detection**: Identifies long methods, duplicate code, and anti-patterns
- **Performance Metrics**: Algorithm efficiency and optimization suggestions
- **Best Practices**: Enforces coding standards and documentation requirements

### 🤖 **Machine Learning Integration**
- **Custom Models**: Trained vulnerability detection and quality assessment models
- **Pattern Recognition**: Advanced threat detection using AST analysis
- **Continuous Learning**: Model improvement based on user feedback
- **Multi-language Support**: Python, JavaScript, TypeScript, Java, C#, Go

### 🔗 **Seamless Integrations**
- **GitHub/GitLab**: Native webhook support for automated PR analysis
- **CI/CD Pipelines**: Jenkins, GitHub Actions, Azure DevOps integration
- **IDE Extensions**: VS Code, IntelliJ plugins for real-time analysis
- **API-First**: RESTful API for custom integrations

### 📈 **Real-time Analytics & Monitoring**
- **Interactive Dashboards**: Comprehensive metrics and trend visualization
- **Prometheus/Grafana**: Production-grade monitoring and alerting
- **Historical Analysis**: Track code quality improvements over time
- **Team Insights**: Collaborative analytics and reporting

---

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │   FastAPI Backend│    │   PostgreSQL DB  │
│   - Dashboard    │◄──►│   - ML Service   │◄──►│   - Analysis Data│
│   - Code Editor  │    │   - API Routes   │    │   - User Data    │
│   - Reports      │    │   - Auth System  │    │   - Models Info  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Redis Cache   │    │   ML Pipeline   │    │   Monitoring    │
│   - Analysis    │    │   - Training    │    │   - Prometheus  │
│   - Sessions    │    │   - Models      │    │   - Grafana     │
│   - Rate Limit  │    │   - Inference   │    │   - Alerts      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 🛠️ Technology Stack

### **Backend**
- **FastAPI** - Modern, fast web framework with automatic API documentation
- **Python 3.11+** - Core language with type hints and async support
- **SQLAlchemy** - Database ORM with async PostgreSQL support
- **Redis** - Caching, session storage, and background task queue
- **Celery** - Distributed task queue for background processing

### **Machine Learning**
- **scikit-learn** - Traditional ML algorithms for pattern recognition
- **PyTorch** - Deep learning models for advanced analysis
- **Transformers** - Pre-trained language models for code understanding
- **Tree-sitter** - Fast, incremental parsing for code analysis
- **NumPy/Pandas** - Efficient data processing and feature engineering

### **Frontend**
- **React 18** - Modern UI library with concurrent features
- **TypeScript** - Type-safe JavaScript for better development experience
- **Material-UI** - Comprehensive React component library
- **React Query** - Powerful data fetching and caching
- **Zustand** - Lightweight state management
- **Monaco Editor** - VS Code editor for in-browser code editing

### **Infrastructure**
- **Docker** - Containerization for consistent deployments
- **PostgreSQL 15** - Robust relational database with JSONB support
- **Prometheus** - Metrics collection and monitoring
- **Grafana** - Beautiful dashboards and alerting
- **Nginx** - Reverse proxy and load balancing

---

## 📋 Prerequisites

- **Docker** and **Docker Compose** (recommended)
- **Python 3.11+** (for local development)
- **Node.js 18+** (for frontend development)
- **PostgreSQL 15+** (if not using Docker)
- **Redis 7+** (if not using Docker)

---

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-powered-code-review.git
cd ai-powered-code-review

# Copy and configure environment
cp .env.example .env
# Edit .env with your configuration

# Start all services
docker-compose up -d

# Check service health
docker-compose ps
```

**Access Points:**
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Grafana**: http://localhost:3001 (admin/admin123)
- **Prometheus**: http://localhost:9090

### Option 2: Local Development

#### Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your database credentials

# Start PostgreSQL and Redis (via Docker)
docker-compose up -d postgres redis

# Run database migrations
alembic upgrade head

# Start the backend server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Configure environment
echo "REACT_APP_API_URL=http://localhost:8000/api/v1" > .env

# Start the development server
npm start
```

---

## 📖 Usage Examples

### 1. Analyze Single File via API

```python
import requests

# Login to get authentication token
auth_response = requests.post('http://localhost:8000/api/v1/auth/login', 
    json={'email': 'user@example.com', 'password': 'password'})
token = auth_response.json()['access_token']

# Analyze Python code
analysis_response = requests.post(
    'http://localhost:8000/api/v1/analysis/analyze',
    headers={'Authorization': f'Bearer {token}'},
    json={
        'code_content': '''
def vulnerable_function(user_input):
    import os
    os.system(f"echo {user_input}")  # Security vulnerability!
    return "done"
        ''',
        'file_path': 'example.py',
        'language': 'python',
        'analysis_type': 'full'
    }
)

results = analysis_response.json()
print(f"Overall Score: {results['overall_score']}")
print(f"Security Issues: {len([i for i in results['issues'] if i['category'] == 'security'])}")
print(f"Quality Issues: {len([i for i in results['issues'] if i['category'] == 'quality'])}")
```

### 2. Batch Analysis via CLI

```bash
# Install the CLI tool
pip install ai-code-review-cli

# Analyze entire project
ai-review analyze ./src --format=json --output=report.json

# Security-focused scan
ai-review scan --security-only ./src

# Generate HTML report
ai-review report --format=html --output=security-report.html
```

### 3. GitHub Integration

```yaml
# .github/workflows/code-review.yml
name: AI Code Review
on: [pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: AI Code Review
        uses: yourusername/ai-code-review-action@v1
        with:
          api-key: ${{ secrets.AI_REVIEW_API_KEY }}
          threshold: 80
          fail-on-security: true
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection string | - | ✅ |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` | ✅ |
| `SECRET_KEY` | JWT signing secret (32+ chars) | - | ✅ |
| `DEBUG` | Enable debug mode | `false` | ❌ |
| `ALLOWED_ORIGINS` | CORS allowed origins | `http://localhost:3000` | ❌ |
| `ML_MODEL_PATH` | Path to ML models | `./models` | ❌ |
| `GITHUB_WEBHOOK_SECRET` | GitHub webhook secret | - | ❌ |
| `VULNERABILITY_THRESHOLD` | Security alert threshold | `0.7` | ❌ |
| `QUALITY_THRESHOLD` | Quality gate threshold | `0.6` | ❌ |

### ML Model Configuration

```python
# backend/app/core/config.py
class MLSettings:
    BATCH_SIZE: int = 32
    MAX_FILE_SIZE: int = 1024 * 1024  # 1MB
    MODEL_UPDATE_INTERVAL: int = 3600  # 1 hour
    
    # Model thresholds
    VULNERABILITY_THRESHOLD: float = 0.7
    QUALITY_THRESHOLD: float = 0.6
    COMPLEXITY_THRESHOLD: int = 10
    
    # Supported languages
    SUPPORTED_LANGUAGES = [
        'python', 'javascript', 'typescript', 
        'java', 'csharp', 'go', 'rust'
    ]
```

---

## 📊 Monitoring & Observability

### Prometheus Metrics

The system exposes comprehensive metrics for monitoring:

```yaml
# HTTP metrics
http_requests_total{method, endpoint, status_code}
http_request_duration_seconds{method, endpoint}

# ML metrics  
ml_predictions_total{model_name, prediction_type}
ml_prediction_duration_seconds{model_name}
analysis_issues_found_total{issue_type, severity}

# System metrics
database_connections_active
redis_cache_hit_ratio
celery_tasks_total{status}
```

### Grafana Dashboards

Pre-configured dashboards include:

- **System Overview**: Request rates, response times, error rates
- **Security Metrics**: Vulnerability trends, threat detection rates
- **Quality Metrics**: Code quality trends, technical debt tracking
- **ML Performance**: Model accuracy, prediction latency, cache performance
- **Infrastructure**: Database performance, Redis metrics, system resources

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# Detailed system status
curl http://localhost:8000/api/v1/status

# ML models status
curl http://localhost:8000/api/v1/analysis/models/status
```

---

## 🧪 Testing

### Backend Tests

```bash
cd backend

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html --cov-report=term

# Run specific test categories
pytest -m security  # Security tests
pytest -m ml        # ML model tests
pytest -m api       # API endpoint tests

# Run integration tests
pytest tests/integration/
```

### Frontend Tests

```bash
cd frontend

# Run unit tests
npm test

# Run with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e

# Visual regression tests
npm run test:visual
```

### Load Testing

```bash
# Install k6
brew install k6  # macOS
# or
sudo apt install k6  # Ubuntu

# Run load tests
k6 run tests/load/api_load_test.js

# Stress test ML endpoints
k6 run tests/load/ml_stress_test.js
```

---

## 🚀 Deployment

### Production Docker Setup

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  backend:
    image: yourusername/ai-code-review-backend:latest
    environment:
      DEBUG: "false"
      WORKERS: 4
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx/prod.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    ports:
      - "80:80"
      - "443:443"
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Verify deployment
kubectl get pods -l app=ai-code-review

# Check logs
kubectl logs -f deployment/ai-code-review-backend

# Scale deployment
kubectl scale deployment ai-code-review-backend --replicas=5
```

### AWS ECS Deployment

```bash
# Build and push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-west-2.amazonaws.com
docker build -t ai-code-review-backend backend/
docker tag ai-code-review-backend:latest 123456789.dkr.ecr.us-west-2.amazonaws.com/ai-code-review-backend:latest
docker push 123456789.dkr.ecr.us-west-2.amazonaws.com/ai-code-review-backend:latest

# Deploy via ECS CLI
ecs-cli compose --project-name ai-code-review service up --cluster production
```

---

## 🔐 Security Considerations

### Authentication & Authorization
- **JWT Tokens**: Secure token-based authentication with configurable expiration
- **Role-Based Access Control**: Fine-grained permissions system
- **API Rate Limiting**: Protection against abuse and DoS attacks
- **Password Security**: Bcrypt hashing with salt rounds

### Data Protection
- **Input Validation**: Comprehensive request validation using Pydantic
- **SQL Injection Prevention**: Parameterized queries via SQLAlchemy ORM
- **XSS Protection**: Content Security Policy headers
- **HTTPS Enforcement**: SSL/TLS termination and redirect

### Infrastructure Security
- **Container Security**: Non-root users, minimal base images
- **Network Isolation**: Docker networks and security groups
- **Secrets Management**: Environment variables and secret stores
- **Regular Updates**: Automated dependency scanning and updates

---

## 📈 Performance Optimization

### Database Optimization
```sql
-- Recommended indexes
CREATE INDEX CONCURRENTLY idx_analyses_user_id_created_at ON analyses(user_id, created_at DESC);
CREATE INDEX CONCURRENTLY idx_analyses_repository_id ON analyses(repository_id);
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
```

### Caching Strategy
- **Application Cache**: Redis for session data and frequently accessed data
- **Query Cache**: Database query result caching for expensive operations
- **CDN**: Static asset caching via CloudFront or similar
- **Browser Cache**: Aggressive caching for static resources

### ML Performance
- **Model Caching**: In-memory model loading for faster predictions
- **Batch Processing**: Efficient batch analysis for multiple files
- **GPU Acceleration**: CUDA support for large-scale model inference
- **Model Quantization**: Reduced model size for faster inference

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes with tests
4. **Run** the test suite (`pytest` and `npm test`)
5. **Commit** your changes (`git commit -m 'Add amazing feature'`)
6. **Push** to the branch (`git push origin feature/amazing-feature`)
7. **Open** a Pull Request

### Code Standards

- **Python**: Follow PEP 8, use type hints, maintain >90% test coverage
- **TypeScript**: Follow ESLint rules, use strict type checking
- **Documentation**: Update docs for all public APIs and features
- **Testing**: Write unit tests for all new functionality

### Adding New Analyzers

```python
# Example: Adding a new security analyzer
from app.ml_models.base_analyzer import BaseAnalyzer

class CustomSecurityAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        self.name = "custom_security"
        self.supported_languages = ["python", "javascript"]
    
    def analyze(self, code_content: str, file_path: str) -> dict:
        # Your analysis logic here
        vulnerabilities = self._detect_custom_patterns(code_content)
        return {
            "vulnerabilities": vulnerabilities,
            "confidence": 0.85
        }
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🆘 Support & Troubleshooting

### Common Issues

**Q: Backend fails to start with database connection error**
```bash
# Solution: Ensure PostgreSQL is running and credentials are correct
docker-compose up -d postgres
# Check connection
psql -h localhost -U postgres -d ai_code_review
```

**Q: ML models not loading properly**
```bash
# Solution: Download or train models
cd backend
python -m app.ml_models.download_models
# Or train from scratch
python -m app.ml_models.train_models
```

**Q: Frontend build fails with memory error**
```bash
# Solution: Increase Node.js memory limit
export NODE_OPTIONS="--max-old-space-size=4096"
npm run build
```

### Performance Tuning

1. **Database Performance**
   - Add appropriate indexes for your query patterns
   - Use connection pooling with optimal pool size
   - Enable query result caching for expensive operations

2. **ML Model Performance**
   - Use GPU acceleration when available
   - Implement model result caching
   - Consider model quantization for faster inference

3. **Frontend Performance**
   - Enable code splitting for large applications
   - Implement virtual scrolling for large data sets
   - Use React.memo for expensive components

### Getting Help

- 📖 **Documentation**: [Full API Documentation](docs/api.md)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/yourusername/ai-powered-code-review/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-powered-code-review/discussions)
- 📧 **Email Support**: support@ai-code-review.com
- 💬 **Discord**: [Community Server](https://discord.gg/ai-code-review)

---

## 🎉 Acknowledgments

- **Tree-sitter** team for excellent code parsing capabilities
- **Hugging Face** for transformer models and infrastructure  
- **FastAPI** team for the incredible web framework
- **React** team for the powerful UI library
- **scikit-learn** community for machine learning tools
- **Open Source Community** for continuous contributions and feedback

---

<div align="center">

**Built with ❤️ for the developer community**

[⭐ Star on GitHub](https://github.com/yourusername/ai-powered-code-review) | [📖 Documentation](docs/) | [🚀 Live Demo](https://demo.ai-code-review.com)

*Making code review intelligent, one analysis at a time.*

</div>