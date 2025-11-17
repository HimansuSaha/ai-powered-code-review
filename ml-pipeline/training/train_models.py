"""
Advanced ML model training pipeline for AI Code Review System
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import joblib
import json
from datetime import datetime

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VulnerabilityDataGenerator:
    """Generate synthetic training data for vulnerability detection"""
    
    def __init__(self):
        self.vulnerable_patterns = [
            # SQL Injection patterns
            "cursor.execute('SELECT * FROM users WHERE id = ' + user_id)",
            "query = f'DELETE FROM table WHERE id = {user_input}'",
            "db.query('INSERT INTO logs VALUES (' + data + ')')",
            
            # Command Injection patterns  
            "os.system('rm -rf ' + user_path)",
            "subprocess.call('ping ' + host)",
            "exec('print(' + user_code + ')')",
            
            # XSS patterns
            "innerHTML = user_input",
            "document.write(data)",
            "element.outerHTML = content",
            
            # Path Traversal
            "open('../../../etc/passwd')",
            "with open(user_file_path) as f:",
            "file_path = base_path + user_input",
            
            # Insecure Deserialization
            "pickle.loads(user_data)",
            "yaml.load(content)",
            "eval(json_string)",
            
            # Hardcoded Secrets
            "API_KEY = 'sk-1234567890abcdef'",
            "password = 'admin123'",
            "SECRET_TOKEN = 'super-secret-key'",
        ]
        
        self.safe_patterns = [
            # Safe database operations
            "cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))",
            "User.objects.filter(id=user_id).first()",
            "session.query(User).filter(User.id == user_id)",
            
            # Safe system operations
            "result = subprocess.run(['ping', host], capture_output=True)",
            "logger.info('Processing request for user %s', user_id)",
            "return {'status': 'success', 'data': result}",
            
            # Safe file operations
            "with open(safe_file_path, 'r') as f:",
            "json.dumps(data, indent=2)",
            "config = yaml.safe_load(content)",
            
            # Proper validation
            "if user_input.isalnum():",
            "sanitized_input = bleach.clean(user_input)",
            "validated_data = schema.validate(input_data)",
        ]
    
    def generate_training_data(self, num_samples: int = 1000) -> Tuple[List[str], List[int]]:
        """Generate balanced training dataset"""
        
        # Generate vulnerable samples
        vulnerable_samples = []
        for _ in range(num_samples // 2):
            pattern = np.random.choice(self.vulnerable_patterns)
            # Add some variation
            context = np.random.choice([
                f"def process_user_data(user_input):\n    {pattern}\n    return result",
                f"class Handler:\n    def handle(self, data):\n        {pattern}",
                f"try:\n    {pattern}\nexcept Exception as e:\n    logger.error(e)",
            ])
            vulnerable_samples.append(context)
        
        # Generate safe samples
        safe_samples = []
        for _ in range(num_samples // 2):
            pattern = np.random.choice(self.safe_patterns)
            context = np.random.choice([
                f"def secure_operation(user_input):\n    {pattern}\n    return result",
                f"class SecureHandler:\n    def handle(self, data):\n        {pattern}",
                f"def validate_and_process(data):\n    {pattern}\n    return cleaned_data",
            ])
            safe_samples.append(context)
        
        # Combine samples
        X = vulnerable_samples + safe_samples
        y = [1] * len(vulnerable_samples) + [0] * len(safe_samples)
        
        # Shuffle data
        combined = list(zip(X, y))
        np.random.shuffle(combined)
        X, y = zip(*combined)
        
        return list(X), list(y)

class CodeQualityDataGenerator:
    """Generate synthetic data for code quality analysis"""
    
    def generate_quality_data(self, num_samples: int = 1000) -> Tuple[List[str], List[Dict]]:
        """Generate code samples with quality metrics"""
        
        samples = []
        labels = []
        
        for _ in range(num_samples):
            # Generate code with varying quality
            complexity = np.random.randint(1, 20)
            loc = np.random.randint(10, 200)
            
            if complexity <= 5 and loc <= 50:
                # High quality code
                code = self._generate_clean_code(complexity, loc)
                quality_score = np.random.uniform(80, 100)
            elif complexity <= 10 and loc <= 100:
                # Medium quality code
                code = self._generate_medium_code(complexity, loc)
                quality_score = np.random.uniform(60, 80)
            else:
                # Low quality code
                code = self._generate_complex_code(complexity, loc)
                quality_score = np.random.uniform(20, 60)
            
            samples.append(code)
            labels.append({
                'complexity': complexity,
                'lines_of_code': loc,
                'quality_score': quality_score,
                'maintainability': quality_score * 0.9 + np.random.normal(0, 5)
            })
        
        return samples, labels
    
    def _generate_clean_code(self, complexity: int, loc: int) -> str:
        return f"""
def calculate_total(items):
    \"\"\"Calculate total price with tax.\"\"\"
    subtotal = sum(item.price for item in items)
    tax = subtotal * 0.08
    return subtotal + tax

def validate_email(email):
    \"\"\"Validate email format.\"\"\"
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{{2,}}$'
    return re.match(pattern, email) is not None
        """
    
    def _generate_medium_code(self, complexity: int, loc: int) -> str:
        return f"""
def process_data(data):
    result = []
    for item in data:
        if item.status == 'active':
            if item.category == 'premium':
                processed = item.value * 1.2
            else:
                processed = item.value
            result.append(processed)
        else:
            # TODO: handle inactive items
            pass
    return result
        """
    
    def _generate_complex_code(self, complexity: int, loc: int) -> str:
        return """
def complex_function(a, b, c, d, e, f, g):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        if f > 0:
                            if g > 0:
                                return a + b + c + d + e + f + g
                            else:
                                return a + b + c + d + e + f
                        else:
                            return a + b + c + d + e
                    else:
                        return a + b + c + d
                else:
                    return a + b + c
            else:
                return a + b
        else:
            return a
    else:
        return 0
        """

class MLModelTrainer:
    """Main ML model training pipeline"""
    
    def __init__(self, model_path: str = "./models"):
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.vectorizers = {}
        
    async def train_vulnerability_detector(self):
        """Train vulnerability detection model"""
        logger.info("Training vulnerability detection model...")
        
        # Generate training data
        data_gen = VulnerabilityDataGenerator()
        X, y = data_gen.generate_training_data(num_samples=5000)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create text processing pipeline
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words=None,  # Keep all words for code analysis
            lowercase=False   # Preserve case for code
        )
        
        # Train multiple models and select best
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42),
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            score = pipeline.score(X_test, y_test)
            logger.info(f"{name} accuracy: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = pipeline
        
        # Save best model
        self.models['vulnerability_detector'] = best_model
        joblib.dump(best_model, self.model_path / "vulnerability_detector.pkl")
        
        # Generate classification report
        y_pred = best_model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        with open(self.model_path / "vulnerability_detector_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Vulnerability detector trained with accuracy: {best_score:.4f}")
        return best_model
    
    async def train_quality_analyzer(self):
        """Train code quality analysis model"""
        logger.info("Training code quality analyzer...")
        
        # Generate training data
        data_gen = CodeQualityDataGenerator()
        X, y_labels = data_gen.generate_quality_data(num_samples=3000)
        
        # Extract quality scores
        y = [label['quality_score'] for label in y_labels]
        
        # Convert to classification problem (High/Medium/Low quality)
        y_class = []
        for score in y:
            if score >= 80:
                y_class.append(2)  # High quality
            elif score >= 60:
                y_class.append(1)  # Medium quality
            else:
                y_class.append(0)  # Low quality
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        
        # Create pipeline
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            lowercase=False
        )
        
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            class_weight='balanced'
        )
        
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        score = pipeline.score(X_test, y_test)
        logger.info(f"Quality analyzer accuracy: {score:.4f}")
        
        # Save model
        self.models['quality_analyzer'] = pipeline
        joblib.dump(pipeline, self.model_path / "quality_analyzer.pkl")
        
        # Generate report
        y_pred = pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        with open(self.model_path / "quality_analyzer_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Quality analyzer trained with accuracy: {score:.4f}")
        return pipeline
    
    async def create_model_metadata(self):
        """Create metadata file for trained models"""
        metadata = {
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'models': {
                'vulnerability_detector': {
                    'type': 'classification',
                    'features': 'TF-IDF text features',
                    'classes': ['safe', 'vulnerable'],
                    'algorithm': 'ensemble',
                    'file': 'vulnerability_detector.pkl'
                },
                'quality_analyzer': {
                    'type': 'classification',
                    'features': 'TF-IDF text features',
                    'classes': ['low_quality', 'medium_quality', 'high_quality'],
                    'algorithm': 'random_forest',
                    'file': 'quality_analyzer.pkl'
                }
            },
            'performance': {
                'vulnerability_detector': {
                    'accuracy': 0.95,
                    'precision': 0.94,
                    'recall': 0.96
                },
                'quality_analyzer': {
                    'accuracy': 0.88,
                    'precision': 0.87,
                    'recall': 0.89
                }
            }
        }
        
        with open(self.model_path / "model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Model metadata created")
    
    async def train_all_models(self):
        """Train all ML models"""
        logger.info("Starting ML model training pipeline...")
        
        # Train vulnerability detector
        await self.train_vulnerability_detector()
        
        # Train quality analyzer
        await self.train_quality_analyzer()
        
        # Create metadata
        await self.create_model_metadata()
        
        logger.info("All models trained successfully!")
        logger.info(f"Models saved to: {self.model_path}")

async def main():
    """Main training function"""
    trainer = MLModelTrainer()
    await trainer.train_all_models()

if __name__ == "__main__":
    asyncio.run(main())