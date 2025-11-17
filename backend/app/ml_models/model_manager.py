import asyncio
import logging
import pickle
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import re

from ..core.config import get_settings

logger = logging.getLogger(__name__)

class VulnerabilityDetector:
    """ML model for detecting security vulnerabilities in code"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_loaded = False
        
    async def load_model(self, model_path: Path):
        """Load pre-trained vulnerability detection model"""
        try:
            model_file = model_path / "vulnerability_detector.pkl"
            vectorizer_file = model_path / "vulnerability_vectorizer.pkl"
            
            if model_file.exists() and vectorizer_file.exists():
                self.model = joblib.load(model_file)
                self.vectorizer = joblib.load(vectorizer_file)
                self.is_loaded = True
                logger.info("Vulnerability detector model loaded successfully")
            else:
                # Create and train a simple model if files don't exist
                await self._create_default_model()
                logger.info("Created default vulnerability detector model")
                
        except Exception as e:
            logger.error(f"Failed to load vulnerability detector: {e}")
            await self._create_default_model()
    
    async def _create_default_model(self):
        """Create a default trained model"""
        # Sample vulnerable code patterns
        vulnerable_patterns = [
            "eval(user_input)",
            "exec(user_input)",
            "os.system(user_input)",
            "subprocess.call(user_input)",
            "sql = f'SELECT * FROM users WHERE id = {user_id}'",
            "cursor.execute('SELECT * FROM users WHERE name = ' + user_name)",
            "document.innerHTML = user_input",
            "element.outerHTML = data",
        ]
        
        # Sample safe code patterns
        safe_patterns = [
            "result = calculate_sum(a, b)",
            "user = User.objects.get(pk=user_id)",
            "data = json.loads(json_string)",
            "response = requests.get(url, timeout=30)",
            "logger.info('Processing request')",
            "return {'status': 'success', 'data': result}",
        ]
        
        # Create training data
        X_train = vulnerable_patterns + safe_patterns
        y_train = [1] * len(vulnerable_patterns) + [0] * len(safe_patterns)
        
        # Train vectorizer and model
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        X_vectorized = self.vectorizer.fit_transform(X_train)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_vectorized, y_train)
        
        self.is_loaded = True
    
    def predict(self, code_content: str) -> Dict[str, Any]:
        """Predict vulnerabilities in code"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        try:
            # Extract features from code
            features = self._extract_features(code_content)
            
            # Vectorize code
            code_vector = self.vectorizer.transform([code_content])
            
            # Predict
            vulnerability_prob = self.model.predict_proba(code_vector)[0][1]
            is_vulnerable = vulnerability_prob > 0.7
            
            # Identify specific vulnerability types
            vulnerabilities = self._identify_vulnerability_types(code_content)
            
            return {
                "is_vulnerable": is_vulnerable,
                "vulnerability_score": float(vulnerability_prob),
                "vulnerabilities": vulnerabilities,
                "confidence": float(vulnerability_prob)
            }
            
        except Exception as e:
            logger.error(f"Vulnerability prediction failed: {e}")
            return {"error": str(e)}
    
    def _extract_features(self, code_content: str) -> Dict[str, int]:
        """Extract security-relevant features from code"""
        features = {}
        
        # Dangerous function calls
        dangerous_functions = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'os\.system\s*\(',
            r'subprocess\.call\s*\(',
            r'subprocess\.run\s*\(',
        ]
        
        for pattern in dangerous_functions:
            features[f"has_{pattern}"] = len(re.findall(pattern, code_content))
        
        # SQL injection patterns
        sql_patterns = [
            r'cursor\.execute\s*\(\s*[\'"].+\+',
            r'\.format\s*\(.+\)',
            r'%.*%',
        ]
        
        for pattern in sql_patterns:
            features[f"sql_{pattern}"] = len(re.findall(pattern, code_content))
        
        return features
    
    def _identify_vulnerability_types(self, code_content: str) -> List[Dict[str, Any]]:
        """Identify specific types of vulnerabilities"""
        vulnerabilities = []
        
        # Check for code injection
        if re.search(r'eval\s*\(|exec\s*\(', code_content):
            vulnerabilities.append({
                "type": "code_injection",
                "severity": "critical",
                "description": "Potential code injection vulnerability detected"
            })
        
        # Check for command injection
        if re.search(r'os\.system\s*\(|subprocess\.call\s*\(', code_content):
            vulnerabilities.append({
                "type": "command_injection", 
                "severity": "high",
                "description": "Potential command injection vulnerability detected"
            })
        
        # Check for SQL injection
        if re.search(r'cursor\.execute\s*\(\s*[\'"].+\+|\.format\s*\(', code_content):
            vulnerabilities.append({
                "type": "sql_injection",
                "severity": "high", 
                "description": "Potential SQL injection vulnerability detected"
            })
        
        return vulnerabilities

class QualityAnalyzer:
    """ML model for analyzing code quality metrics"""
    
    def __init__(self):
        self.complexity_model = None
        self.maintainability_model = None
        self.is_loaded = False
    
    async def load_model(self, model_path: Path):
        """Load quality analysis models"""
        try:
            # For now, use rule-based analysis
            self.is_loaded = True
            logger.info("Quality analyzer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load quality analyzer: {e}")
    
    def analyze(self, code_content: str, file_path: str) -> Dict[str, Any]:
        """Analyze code quality"""
        try:
            # Calculate cyclomatic complexity
            complexity = self._calculate_complexity(code_content)
            
            # Calculate maintainability index
            maintainability = self._calculate_maintainability(code_content)
            
            # Detect code smells
            code_smells = self._detect_code_smells(code_content)
            
            return {
                "complexity": complexity,
                "maintainability_index": maintainability,
                "code_smells": code_smells,
                "lines_of_code": len(code_content.splitlines()),
                "quality_score": self._calculate_quality_score(complexity, maintainability)
            }
            
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_complexity(self, code_content: str) -> int:
        """Calculate cyclomatic complexity"""
        try:
            tree = ast.parse(code_content)
            complexity = 1  # Base complexity
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
                    
            return complexity
            
        except:
            # Fallback: count control structures with regex
            control_structures = len(re.findall(r'\b(if|elif|else|for|while|try|except|with)\b', code_content))
            return max(1, control_structures)
    
    def _calculate_maintainability(self, code_content: str) -> float:
        """Calculate maintainability index (simplified)"""
        lines = code_content.splitlines()
        loc = len([line for line in lines if line.strip()])
        
        if loc == 0:
            return 100.0
        
        # Simplified maintainability calculation
        comment_ratio = len([line for line in lines if line.strip().startswith('#')]) / loc
        avg_line_length = sum(len(line) for line in lines) / loc
        
        # Simple heuristic (higher is better)
        maintainability = 100 - (avg_line_length * 0.5) + (comment_ratio * 10)
        return max(0.0, min(100.0, maintainability))
    
    def _detect_code_smells(self, code_content: str) -> List[Dict[str, Any]]:
        """Detect common code smells"""
        smells = []
        lines = code_content.splitlines()
        
        # Long method detection
        if len(lines) > 50:
            smells.append({
                "type": "long_method",
                "severity": "medium",
                "description": "Method is too long (>50 lines)"
            })
        
        # Long line detection
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                smells.append({
                    "type": "long_line",
                    "severity": "low",
                    "line": i,
                    "description": f"Line {i} is too long ({len(line)} characters)"
                })
        
        # TODO comments
        for i, line in enumerate(lines, 1):
            if 'TODO' in line or 'FIXME' in line:
                smells.append({
                    "type": "todo_comment",
                    "severity": "info",
                    "line": i,
                    "description": "TODO/FIXME comment found"
                })
        
        return smells
    
    def _calculate_quality_score(self, complexity: int, maintainability: float) -> float:
        """Calculate overall quality score"""
        complexity_score = max(0, 100 - (complexity * 5))
        return (complexity_score + maintainability) / 2

class ModelManager:
    """Central manager for all ML models"""
    
    def __init__(self):
        self.vulnerability_detector = VulnerabilityDetector()
        self.quality_analyzer = QualityAnalyzer()
        self.settings = get_settings()
        self.models_loaded = False
    
    async def initialize_models(self):
        """Initialize all ML models"""
        try:
            model_path = Path(self.settings.ML_MODEL_PATH)
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Load models
            await self.vulnerability_detector.load_model(model_path)
            await self.quality_analyzer.load_model(model_path)
            
            self.models_loaded = True
            logger.info("All ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def analyze_code(self, code_content: str, file_path: str, analysis_type: str = "full") -> Dict[str, Any]:
        """Perform comprehensive code analysis"""
        if not self.models_loaded:
            raise RuntimeError("Models not loaded")
        
        results = {}
        
        try:
            if analysis_type in ["security", "full"]:
                # Security analysis
                security_results = self.vulnerability_detector.predict(code_content)
                results["security"] = security_results
            
            if analysis_type in ["quality", "full"]:
                # Quality analysis
                quality_results = self.quality_analyzer.analyze(code_content, file_path)
                results["quality"] = quality_results
            
            # Calculate overall score
            results["overall_score"] = self._calculate_overall_score(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall analysis score"""
        scores = []
        
        if "security" in results and "vulnerability_score" in results["security"]:
            # Invert vulnerability score (lower is better)
            security_score = (1 - results["security"]["vulnerability_score"]) * 100
            scores.append(security_score)
        
        if "quality" in results and "quality_score" in results["quality"]:
            scores.append(results["quality"]["quality_score"])
        
        return sum(scores) / len(scores) if scores else 0.0
    
    async def get_models_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        return {
            "vulnerability_detector": {
                "loaded": self.vulnerability_detector.is_loaded,
                "status": "operational" if self.vulnerability_detector.is_loaded else "not_loaded"
            },
            "quality_analyzer": {
                "loaded": self.quality_analyzer.is_loaded, 
                "status": "operational" if self.quality_analyzer.is_loaded else "not_loaded"
            },
            "overall_status": "operational" if self.models_loaded else "not_ready"
        }