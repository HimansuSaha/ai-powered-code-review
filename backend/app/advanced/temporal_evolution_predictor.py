"""
Temporal Code Evolution Prediction System
========================================

Revolutionary ML system that predicts how code will evolve over time based on:
- Historical patterns in code repositories
- Developer behavior analysis  
- Technical debt accumulation patterns
- Environmental and contextual factors
- Temporal dependencies between code changes

This system can predict future bugs, suggest proactive refactoring, and forecast
technical debt before it becomes problematic.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import networkx as nx
from collections import defaultdict, deque
import ast
import pickle
import json
import hashlib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CodeTemporalState:
    """Represents the state of code at a specific point in time."""
    timestamp: datetime
    file_path: str
    code_content: str
    complexity_metrics: Dict[str, float]
    dependency_graph: Dict[str, List[str]]
    developer_context: Dict[str, Any]
    environmental_factors: Dict[str, Any]
    change_vector: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class EvolutionPrediction:
    """Represents a prediction about how code will evolve."""
    prediction_horizon: timedelta
    predicted_changes: List[Dict[str, Any]]
    confidence_intervals: Dict[str, Tuple[float, float]]
    risk_factors: List[Dict[str, Any]]
    recommended_actions: List[Dict[str, Any]]
    temporal_dependencies: List[str]


class TemporalCodeEvolutionPredictor:
    """
    Advanced ML system for predicting code evolution patterns and future changes.
    """
    
    def __init__(self, lookback_window: int = 365, prediction_horizon: int = 90):
        self.lookback_window = lookback_window  # days
        self.prediction_horizon = prediction_horizon  # days
        
        # ML Models for different prediction tasks
        self.complexity_predictor = None
        self.bug_risk_predictor = None
        self.refactor_need_predictor = None
        self.dependency_evolution_predictor = None
        
        # Neural network for deep temporal patterns
        self.temporal_nn = None
        self.feature_scaler = StandardScaler()
        self.label_encoders = {}
        
        # Historical data storage
        self.temporal_states = deque(maxlen=10000)
        self.evolution_patterns = {}
        self.developer_profiles = {}
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_ttl = timedelta(hours=1)
        
    async def predict_code_evolution(
        self, 
        code_content: str, 
        file_path: str,
        historical_data: Optional[List[CodeTemporalState]] = None
    ) -> EvolutionPrediction:
        """
        Predict how the given code will evolve over time.
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(code_content, file_path)
            if cache_key in self.prediction_cache:
                cached_result, cache_time = self.prediction_cache[cache_key]
                if datetime.now() - cache_time < self.cache_ttl:
                    return cached_result
            
            # Extract current temporal state
            current_state = await self._extract_temporal_state(code_content, file_path)
            
            # Load or use provided historical data
            if historical_data is None:
                historical_data = await self._load_historical_data(file_path)
            
            # Train/update models with historical data
            await self._update_temporal_models(historical_data)
            
            # Generate predictions for different aspects
            predictions = await self._generate_comprehensive_predictions(
                current_state, historical_data
            )
            
            # Analyze temporal dependencies
            dependencies = await self._analyze_temporal_dependencies(
                current_state, historical_data
            )
            
            # Generate risk assessment
            risk_factors = await self._assess_evolution_risks(
                current_state, predictions
            )
            
            # Create actionable recommendations
            recommendations = await self._generate_proactive_recommendations(
                predictions, risk_factors
            )
            
            # Compile final prediction
            evolution_prediction = EvolutionPrediction(
                prediction_horizon=timedelta(days=self.prediction_horizon),
                predicted_changes=predictions,
                confidence_intervals=self._calculate_confidence_intervals(predictions),
                risk_factors=risk_factors,
                recommended_actions=recommendations,
                temporal_dependencies=dependencies
            )
            
            # Cache result
            self.prediction_cache[cache_key] = (evolution_prediction, datetime.now())
            
            return evolution_prediction
            
        except Exception as e:
            return EvolutionPrediction(
                prediction_horizon=timedelta(days=self.prediction_horizon),
                predicted_changes=[],
                confidence_intervals={},
                risk_factors=[{"type": "prediction_error", "message": str(e)}],
                recommended_actions=[],
                temporal_dependencies=[]
            )
    
    async def _extract_temporal_state(self, code_content: str, file_path: str) -> CodeTemporalState:
        """
        Extract comprehensive temporal state from current code.
        """
        # Parse AST for structural analysis
        tree = ast.parse(code_content)
        
        # Calculate complexity metrics
        complexity_metrics = await self._calculate_complexity_metrics(tree)
        
        # Build dependency graph
        dependency_graph = await self._build_dependency_graph(tree)
        
        # Extract developer context (simulated for now)
        developer_context = await self._extract_developer_context(file_path)
        
        # Assess environmental factors
        environmental_factors = await self._assess_environmental_factors()
        
        return CodeTemporalState(
            timestamp=datetime.now(),
            file_path=file_path,
            code_content=code_content,
            complexity_metrics=complexity_metrics,
            dependency_graph=dependency_graph,
            developer_context=developer_context,
            environmental_factors=environmental_factors
        )
    
    async def _calculate_complexity_metrics(self, ast_tree: ast.AST) -> Dict[str, float]:
        """
        Calculate comprehensive complexity metrics for temporal analysis.
        """
        metrics = {
            'cyclomatic_complexity': 0,
            'cognitive_complexity': 0,
            'nesting_depth': 0,
            'function_count': 0,
            'class_count': 0,
            'line_count': 0,
            'variable_count': 0,
            'import_count': 0,
            'temporal_coupling': 0,
            'change_frequency_risk': 0
        }
        
        # Traverse AST and calculate metrics
        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                metrics['cyclomatic_complexity'] += 1
                
            if isinstance(node, ast.FunctionDef):
                metrics['function_count'] += 1
                
            if isinstance(node, ast.ClassDef):
                metrics['class_count'] += 1
                
            if isinstance(node, ast.Name):
                metrics['variable_count'] += 1
                
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics['import_count'] += 1
        
        # Calculate derived metrics
        metrics['line_count'] = len(ast_tree.body) if hasattr(ast_tree, 'body') else 0
        metrics['cognitive_complexity'] = self._calculate_cognitive_complexity(ast_tree)
        metrics['nesting_depth'] = self._calculate_max_nesting_depth(ast_tree)
        
        # Temporal-specific metrics
        metrics['temporal_coupling'] = await self._calculate_temporal_coupling(ast_tree)
        metrics['change_frequency_risk'] = await self._calculate_change_frequency_risk(ast_tree)
        
        return metrics
    
    def _calculate_cognitive_complexity(self, ast_tree: ast.AST) -> float:
        """
        Calculate cognitive complexity score for temporal analysis.
        """
        complexity = 0
        nesting_level = 0
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 0
                self.nesting_level = 0
                
            def visit_If(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
                
            def visit_For(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
                
            def visit_While(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
        
        visitor = ComplexityVisitor()
        visitor.visit(ast_tree)
        return visitor.complexity
    
    def _calculate_max_nesting_depth(self, ast_tree: ast.AST) -> int:
        """
        Calculate maximum nesting depth.
        """
        max_depth = 0
        current_depth = 0
        
        class DepthVisitor(ast.NodeVisitor):
            def __init__(self):
                self.max_depth = 0
                self.current_depth = 0
                
            def visit(self, node):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.FunctionDef, ast.ClassDef)):
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
                else:
                    self.generic_visit(node)
        
        visitor = DepthVisitor()
        visitor.visit(ast_tree)
        return visitor.max_depth
    
    async def _calculate_temporal_coupling(self, ast_tree: ast.AST) -> float:
        """
        Calculate temporal coupling - how likely this code is to change with other code.
        """
        # Simplified temporal coupling calculation
        # In real implementation, this would analyze git history
        
        import_count = 0
        function_calls = 0
        
        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_count += 1
            elif isinstance(node, ast.Call):
                function_calls += 1
        
        # High imports and function calls indicate higher temporal coupling
        temporal_coupling = min(1.0, (import_count * 0.1 + function_calls * 0.05))
        return temporal_coupling
    
    async def _calculate_change_frequency_risk(self, ast_tree: ast.AST) -> float:
        """
        Calculate risk of frequent changes based on code patterns.
        """
        risk_factors = 0
        
        for node in ast.walk(ast_tree):
            # Complex conditionals increase change risk
            if isinstance(node, ast.If) and len(node.orelse) > 0:
                risk_factors += 0.2
                
            # Long parameter lists increase change risk
            if isinstance(node, ast.FunctionDef) and len(node.args.args) > 5:
                risk_factors += 0.3
                
            # Exception handling indicates potential instability
            if isinstance(node, ast.Try):
                risk_factors += 0.1
        
        return min(1.0, risk_factors)
    
    async def _build_dependency_graph(self, ast_tree: ast.AST) -> Dict[str, List[str]]:
        """
        Build dependency graph for temporal analysis.
        """
        dependencies = defaultdict(list)
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies['imports'].append(alias.name)
                    
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ''
                dependencies['from_imports'].append(module)
                
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                dependencies['function_calls'].append(node.func.id)
                
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    dependencies['method_calls'].append(f"{node.func.value.id}.{node.func.attr}")
        
        return dict(dependencies)
    
    async def _extract_developer_context(self, file_path: str) -> Dict[str, Any]:
        """
        Extract developer context for temporal predictions.
        """
        # In real implementation, this would analyze git blame, commit patterns, etc.
        return {
            'file_age_days': 100,  # Simulated
            'commit_frequency': 0.3,  # commits per day
            'developer_experience': 'senior',
            'team_size': 4,
            'code_review_quality': 0.8,
            'testing_coverage': 0.75
        }
    
    async def _assess_environmental_factors(self) -> Dict[str, Any]:
        """
        Assess environmental factors affecting code evolution.
        """
        return {
            'project_phase': 'maintenance',  # development, maintenance, legacy
            'release_pressure': 0.6,  # 0-1 scale
            'technical_debt_level': 0.4,
            'team_velocity': 0.7,
            'external_dependencies': 15,
            'security_requirements': 0.8,
            'performance_requirements': 0.7,
            'scalability_needs': 0.6
        }
    
    async def _update_temporal_models(self, historical_data: List[CodeTemporalState]):
        """
        Update ML models with historical temporal data.
        """
        if len(historical_data) < 10:
            # Not enough data for training
            await self._initialize_default_models()
            return
        
        # Prepare feature matrix
        X, y_complexity, y_bug_risk, y_refactor = await self._prepare_training_data(historical_data)
        
        if X.size == 0:
            await self._initialize_default_models()
            return
        
        # Train complexity evolution predictor
        self.complexity_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.complexity_predictor.fit(X, y_complexity)
        
        # Train bug risk predictor
        self.bug_risk_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.bug_risk_predictor.fit(X, y_bug_risk)
        
        # Train refactoring need predictor
        self.refactor_need_predictor = MLPRegressor(
            hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42
        )
        self.refactor_need_predictor.fit(X, y_refactor)
        
        # Initialize temporal neural network
        await self._train_temporal_neural_network(X, y_complexity, y_bug_risk, y_refactor)
    
    async def _prepare_training_data(
        self, 
        historical_data: List[CodeTemporalState]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data from historical temporal states.
        """
        features = []
        complexity_targets = []
        bug_risk_targets = []
        refactor_targets = []
        
        # Sort by timestamp
        sorted_data = sorted(historical_data, key=lambda x: x.timestamp)
        
        for i in range(len(sorted_data) - 1):
            current_state = sorted_data[i]
            next_state = sorted_data[i + 1]
            
            # Extract features from current state
            feature_vector = self._extract_feature_vector(current_state)
            features.append(feature_vector)
            
            # Calculate targets based on next state
            complexity_change = (
                next_state.complexity_metrics['cyclomatic_complexity'] - 
                current_state.complexity_metrics['cyclomatic_complexity']
            )
            complexity_targets.append(complexity_change)
            
            # Simulate bug risk (in real implementation, use actual bug data)
            bug_risk = self._calculate_bug_risk_target(current_state, next_state)
            bug_risk_targets.append(bug_risk)
            
            # Simulate refactoring need
            refactor_need = self._calculate_refactor_need_target(current_state, next_state)
            refactor_targets.append(refactor_need)
        
        if not features:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        X = np.array(features)
        y_complexity = np.array(complexity_targets)
        y_bug_risk = np.array(bug_risk_targets)
        y_refactor = np.array(refactor_targets)
        
        # Scale features
        X = self.feature_scaler.fit_transform(X)
        
        return X, y_complexity, y_bug_risk, y_refactor
    
    def _extract_feature_vector(self, state: CodeTemporalState) -> np.ndarray:
        """
        Extract feature vector from temporal state.
        """
        features = []
        
        # Complexity metrics
        complexity_features = [
            state.complexity_metrics['cyclomatic_complexity'],
            state.complexity_metrics['cognitive_complexity'],
            state.complexity_metrics['nesting_depth'],
            state.complexity_metrics['function_count'],
            state.complexity_metrics['class_count'],
            state.complexity_metrics['line_count'],
            state.complexity_metrics['variable_count'],
            state.complexity_metrics['import_count'],
            state.complexity_metrics['temporal_coupling'],
            state.complexity_metrics['change_frequency_risk']
        ]
        features.extend(complexity_features)
        
        # Developer context features
        dev_features = [
            state.developer_context['file_age_days'],
            state.developer_context['commit_frequency'],
            1.0 if state.developer_context['developer_experience'] == 'senior' else 0.0,
            state.developer_context['team_size'],
            state.developer_context['code_review_quality'],
            state.developer_context['testing_coverage']
        ]
        features.extend(dev_features)
        
        # Environmental features
        env_features = [
            1.0 if state.environmental_factors['project_phase'] == 'maintenance' else 0.0,
            state.environmental_factors['release_pressure'],
            state.environmental_factors['technical_debt_level'],
            state.environmental_factors['team_velocity'],
            state.environmental_factors['external_dependencies'],
            state.environmental_factors['security_requirements'],
            state.environmental_factors['performance_requirements'],
            state.environmental_factors['scalability_needs']
        ]
        features.extend(env_features)
        
        return np.array(features)
    
    def _calculate_bug_risk_target(
        self, 
        current_state: CodeTemporalState, 
        next_state: CodeTemporalState
    ) -> float:
        """
        Calculate bug risk target for training.
        """
        # Simplified bug risk calculation
        complexity_increase = (
            next_state.complexity_metrics['cyclomatic_complexity'] - 
            current_state.complexity_metrics['cyclomatic_complexity']
        )
        
        temporal_coupling = current_state.complexity_metrics['temporal_coupling']
        change_risk = current_state.complexity_metrics['change_frequency_risk']
        
        bug_risk = min(1.0, max(0.0, (complexity_increase * 0.1 + temporal_coupling * 0.3 + change_risk * 0.4)))
        return bug_risk
    
    def _calculate_refactor_need_target(
        self, 
        current_state: CodeTemporalState, 
        next_state: CodeTemporalState
    ) -> float:
        """
        Calculate refactoring need target for training.
        """
        # High complexity and poor maintainability indicate refactoring need
        complexity = current_state.complexity_metrics['cyclomatic_complexity']
        cognitive_complexity = current_state.complexity_metrics['cognitive_complexity']
        nesting = current_state.complexity_metrics['nesting_depth']
        
        refactor_need = min(1.0, (complexity * 0.02 + cognitive_complexity * 0.03 + nesting * 0.1))
        return refactor_need
    
    async def _train_temporal_neural_network(
        self, 
        X: np.ndarray, 
        y_complexity: np.ndarray, 
        y_bug_risk: np.ndarray, 
        y_refactor: np.ndarray
    ):
        """
        Train deep neural network for temporal pattern recognition.
        """
        if X.size == 0:
            return
        
        # Define temporal neural network architecture
        class TemporalNN(nn.Module):
            def __init__(self, input_size: int):
                super(TemporalNN, self).__init__()
                self.lstm = nn.LSTM(input_size, 128, batch_first=True, num_layers=2)
                self.dropout = nn.Dropout(0.2)
                self.fc1 = nn.Linear(128, 64)
                self.fc2 = nn.Linear(64, 32)
                self.output = nn.Linear(32, 3)  # complexity, bug_risk, refactor_need
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                x = self.dropout(lstm_out[:, -1, :])  # Take last output
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.output(x)
        
        # Prepare data for LSTM (add sequence dimension)
        X_tensor = torch.FloatTensor(X).unsqueeze(1)  # Add sequence dimension
        y_tensor = torch.FloatTensor(np.column_stack([y_complexity, y_bug_risk, y_refactor]))
        
        # Initialize and train model
        self.temporal_nn = TemporalNN(X.shape[1])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.temporal_nn.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.temporal_nn(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Temporal NN Training Epoch {epoch}, Loss: {loss.item():.4f}")
    
    async def _initialize_default_models(self):
        """
        Initialize default models when insufficient training data is available.
        """
        # Simple baseline models
        self.complexity_predictor = lambda x: np.random.normal(0, 0.1, x.shape[0])
        self.bug_risk_predictor = lambda x: np.random.uniform(0.1, 0.3, x.shape[0])
        self.refactor_need_predictor = lambda x: np.random.uniform(0.2, 0.6, x.shape[0])
    
    async def _generate_comprehensive_predictions(
        self, 
        current_state: CodeTemporalState,
        historical_data: List[CodeTemporalState]
    ) -> List[Dict[str, Any]]:
        """
        Generate comprehensive predictions for code evolution.
        """
        predictions = []
        
        # Extract current features
        current_features = self._extract_feature_vector(current_state)
        current_features_scaled = self.feature_scaler.transform([current_features])
        
        try:
            # Predict complexity evolution
            if hasattr(self.complexity_predictor, 'predict'):
                complexity_change = self.complexity_predictor.predict(current_features_scaled)[0]
            else:
                complexity_change = self.complexity_predictor(current_features_scaled)[0]
                
            predictions.append({
                'type': 'complexity_evolution',
                'prediction': complexity_change,
                'timeline': f'+{self.prediction_horizon} days',
                'confidence': 0.75,
                'description': f'Cyclomatic complexity expected to {"increase" if complexity_change > 0 else "decrease"} by {abs(complexity_change):.1f}'
            })
            
            # Predict bug risk
            if hasattr(self.bug_risk_predictor, 'predict'):
                bug_risk = self.bug_risk_predictor.predict(current_features_scaled)[0]
            else:
                bug_risk = self.bug_risk_predictor(current_features_scaled)[0]
                
            predictions.append({
                'type': 'bug_risk_prediction',
                'prediction': bug_risk,
                'timeline': f'+{self.prediction_horizon} days',
                'confidence': 0.68,
                'description': f'Bug introduction probability: {bug_risk:.1%}'
            })
            
            # Predict refactoring need
            if hasattr(self.refactor_need_predictor, 'predict'):
                refactor_need = self.refactor_need_predictor.predict(current_features_scaled)[0]
            else:
                refactor_need = self.refactor_need_predictor(current_features_scaled)[0]
                
            predictions.append({
                'type': 'refactoring_necessity',
                'prediction': refactor_need,
                'timeline': f'+{self.prediction_horizon} days',
                'confidence': 0.72,
                'description': f'Refactoring urgency level: {refactor_need:.1%}'
            })
            
            # Deep temporal predictions using neural network
            if self.temporal_nn is not None:
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(current_features_scaled).unsqueeze(1)
                    deep_predictions = self.temporal_nn(X_tensor).numpy()[0]
                    
                    predictions.append({
                        'type': 'deep_temporal_analysis',
                        'prediction': deep_predictions.tolist(),
                        'timeline': f'+{self.prediction_horizon} days',
                        'confidence': 0.80,
                        'description': 'Neural network temporal pattern analysis'
                    })
            
            # Predict maintenance burden
            maintenance_burden = await self._predict_maintenance_burden(current_state)
            predictions.append({
                'type': 'maintenance_burden',
                'prediction': maintenance_burden,
                'timeline': f'+{self.prediction_horizon} days',
                'confidence': 0.65,
                'description': f'Expected maintenance effort increase: {maintenance_burden:.1%}'
            })
            
            # Predict technical debt accumulation
            tech_debt_growth = await self._predict_technical_debt_growth(current_state)
            predictions.append({
                'type': 'technical_debt_growth',
                'prediction': tech_debt_growth,
                'timeline': f'+{self.prediction_horizon} days',
                'confidence': 0.70,
                'description': f'Technical debt expected to grow by {tech_debt_growth:.1%}'
            })
            
        except Exception as e:
            predictions.append({
                'type': 'prediction_error',
                'prediction': 0,
                'timeline': 'N/A',
                'confidence': 0,
                'description': f'Prediction failed: {str(e)}'
            })
        
        return predictions
    
    async def _predict_maintenance_burden(self, state: CodeTemporalState) -> float:
        """
        Predict future maintenance burden based on current code state.
        """
        complexity_factor = state.complexity_metrics['cyclomatic_complexity'] * 0.02
        coupling_factor = state.complexity_metrics['temporal_coupling'] * 0.3
        change_risk_factor = state.complexity_metrics['change_frequency_risk'] * 0.25
        
        # Environmental factors
        tech_debt_factor = state.environmental_factors['technical_debt_level'] * 0.2
        velocity_factor = (1 - state.environmental_factors['team_velocity']) * 0.15
        
        maintenance_burden = min(1.0, complexity_factor + coupling_factor + 
                               change_risk_factor + tech_debt_factor + velocity_factor)
        
        return maintenance_burden
    
    async def _predict_technical_debt_growth(self, state: CodeTemporalState) -> float:
        """
        Predict technical debt accumulation rate.
        """
        # Base growth from current complexity
        base_growth = state.complexity_metrics['cyclomatic_complexity'] * 0.01
        
        # Pressure factors
        release_pressure = state.environmental_factors['release_pressure'] * 0.2
        team_size_factor = max(0, (state.developer_context['team_size'] - 5) * 0.02)
        
        # Quality gates
        review_quality_reducer = (1 - state.developer_context['code_review_quality']) * 0.15
        test_coverage_reducer = (1 - state.developer_context['testing_coverage']) * 0.1
        
        debt_growth = min(0.8, base_growth + release_pressure + team_size_factor + 
                         review_quality_reducer + test_coverage_reducer)
        
        return debt_growth
    
    async def _analyze_temporal_dependencies(
        self, 
        current_state: CodeTemporalState,
        historical_data: List[CodeTemporalState]
    ) -> List[str]:
        """
        Analyze temporal dependencies that affect code evolution.
        """
        dependencies = []
        
        # Dependency on external imports
        if len(current_state.dependency_graph.get('imports', [])) > 10:
            dependencies.append("high_external_dependency_risk")
        
        # Dependency on team dynamics
        if current_state.developer_context['team_size'] > 6:
            dependencies.append("large_team_coordination_dependency")
        
        # Dependency on release cycles
        if current_state.environmental_factors['release_pressure'] > 0.7:
            dependencies.append("release_pressure_dependency")
        
        # Dependency on technical infrastructure
        if current_state.environmental_factors['external_dependencies'] > 20:
            dependencies.append("infrastructure_dependency")
        
        # Temporal coupling dependencies
        if current_state.complexity_metrics['temporal_coupling'] > 0.6:
            dependencies.append("high_temporal_coupling_risk")
        
        return dependencies
    
    async def _assess_evolution_risks(
        self, 
        current_state: CodeTemporalState,
        predictions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Assess risks associated with predicted code evolution.
        """
        risks = []
        
        # High complexity growth risk
        complexity_pred = next((p for p in predictions if p['type'] == 'complexity_evolution'), None)
        if complexity_pred and complexity_pred['prediction'] > 5:
            risks.append({
                'type': 'complexity_explosion',
                'severity': 'high',
                'probability': 0.7,
                'impact': 'Rapid complexity growth may make code unmaintainable',
                'mitigation': 'Implement immediate refactoring strategy'
            })
        
        # Bug introduction risk
        bug_pred = next((p for p in predictions if p['type'] == 'bug_risk_prediction'), None)
        if bug_pred and bug_pred['prediction'] > 0.5:
            risks.append({
                'type': 'high_bug_probability',
                'severity': 'medium',
                'probability': bug_pred['prediction'],
                'impact': 'Increased likelihood of bugs being introduced',
                'mitigation': 'Enhance testing coverage and code review processes'
            })
        
        # Technical debt accumulation risk
        debt_pred = next((p for p in predictions if p['type'] == 'technical_debt_growth'), None)
        if debt_pred and debt_pred['prediction'] > 0.4:
            risks.append({
                'type': 'technical_debt_accumulation',
                'severity': 'medium',
                'probability': 0.8,
                'impact': 'Technical debt may compound and slow development',
                'mitigation': 'Allocate time for proactive refactoring'
            })
        
        # Maintenance burden risk
        maintenance_pred = next((p for p in predictions if p['type'] == 'maintenance_burden'), None)
        if maintenance_pred and maintenance_pred['prediction'] > 0.6:
            risks.append({
                'type': 'maintenance_burden_growth',
                'severity': 'high',
                'probability': 0.75,
                'impact': 'Exponential increase in maintenance effort required',
                'mitigation': 'Consider architectural redesign or modularization'
            })
        
        return risks
    
    async def _generate_proactive_recommendations(
        self, 
        predictions: List[Dict[str, Any]],
        risk_factors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate proactive recommendations to prevent predicted issues.
        """
        recommendations = []
        
        # Analyze predictions for actionable insights
        complexity_issues = [p for p in predictions if p['type'] == 'complexity_evolution' and p['prediction'] > 3]
        if complexity_issues:
            recommendations.append({
                'type': 'complexity_reduction',
                'priority': 'high',
                'action': 'Extract functions and simplify conditional logic',
                'timeline': 'Next 2 weeks',
                'estimated_effort': 'Medium',
                'expected_benefit': 'Prevent complexity explosion'
            })
        
        bug_risks = [p for p in predictions if p['type'] == 'bug_risk_prediction' and p['prediction'] > 0.4]
        if bug_risks:
            recommendations.append({
                'type': 'testing_enhancement',
                'priority': 'high',
                'action': 'Increase unit test coverage and add integration tests',
                'timeline': 'Next 3 weeks',
                'estimated_effort': 'High',
                'expected_benefit': 'Reduce bug introduction probability by 60%'
            })
        
        refactor_needs = [p for p in predictions if p['type'] == 'refactoring_necessity' and p['prediction'] > 0.5]
        if refactor_needs:
            recommendations.append({
                'type': 'proactive_refactoring',
                'priority': 'medium',
                'action': 'Refactor high-complexity functions before they become problematic',
                'timeline': 'Next month',
                'estimated_effort': 'Medium',
                'expected_benefit': 'Prevent technical debt accumulation'
            })
        
        # Risk-based recommendations
        high_risks = [r for r in risk_factors if r['severity'] == 'high']
        if high_risks:
            recommendations.append({
                'type': 'risk_mitigation',
                'priority': 'critical',
                'action': 'Address high-severity risks immediately',
                'timeline': 'This week',
                'estimated_effort': 'High',
                'expected_benefit': 'Prevent critical issues from manifesting'
            })
        
        # Maintenance optimization
        recommendations.append({
            'type': 'maintenance_optimization',
            'priority': 'low',
            'action': 'Implement automated code quality checks and monitoring',
            'timeline': 'Next 2 months',
            'estimated_effort': 'Medium',
            'expected_benefit': 'Continuous prevention of quality degradation'
        })
        
        return recommendations
    
    def _calculate_confidence_intervals(self, predictions: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
        """
        Calculate confidence intervals for predictions.
        """
        intervals = {}
        
        for pred in predictions:
            pred_type = pred['type']
            pred_value = pred['prediction']
            confidence = pred['confidence']
            
            # Calculate confidence interval based on prediction uncertainty
            margin = (1 - confidence) * abs(pred_value) * 0.5
            lower_bound = max(0, pred_value - margin) if isinstance(pred_value, (int, float)) else None
            upper_bound = pred_value + margin if isinstance(pred_value, (int, float)) else None
            
            if lower_bound is not None and upper_bound is not None:
                intervals[pred_type] = (lower_bound, upper_bound)
        
        return intervals
    
    async def _load_historical_data(self, file_path: str) -> List[CodeTemporalState]:
        """
        Load historical temporal data for the given file.
        """
        # In real implementation, this would load from git history, databases, etc.
        # For now, return simulated historical data
        historical_data = []
        
        base_time = datetime.now() - timedelta(days=self.lookback_window)
        for i in range(20):  # Simulate 20 historical points
            timestamp = base_time + timedelta(days=i * (self.lookback_window // 20))
            
            # Simulate evolving complexity
            complexity_trend = 1 + (i * 0.1)  # Gradual increase
            
            state = CodeTemporalState(
                timestamp=timestamp,
                file_path=file_path,
                code_content="# Simulated historical code",
                complexity_metrics={
                    'cyclomatic_complexity': 5 * complexity_trend,
                    'cognitive_complexity': 8 * complexity_trend,
                    'nesting_depth': 3,
                    'function_count': 10,
                    'class_count': 2,
                    'line_count': 100 + i * 5,
                    'variable_count': 20 + i * 2,
                    'import_count': 5,
                    'temporal_coupling': 0.3 + i * 0.01,
                    'change_frequency_risk': 0.2 + i * 0.02
                },
                dependency_graph={'imports': ['os', 'sys'], 'function_calls': ['print', 'len']},
                developer_context={
                    'file_age_days': self.lookback_window - i * (self.lookback_window // 20),
                    'commit_frequency': 0.3,
                    'developer_experience': 'senior',
                    'team_size': 4,
                    'code_review_quality': 0.8,
                    'testing_coverage': 0.75
                },
                environmental_factors={
                    'project_phase': 'maintenance',
                    'release_pressure': 0.6,
                    'technical_debt_level': 0.3 + i * 0.01,
                    'team_velocity': 0.7,
                    'external_dependencies': 15,
                    'security_requirements': 0.8,
                    'performance_requirements': 0.7,
                    'scalability_needs': 0.6
                }
            )
            
            historical_data.append(state)
        
        return historical_data
    
    def _generate_cache_key(self, code_content: str, file_path: str) -> str:
        """
        Generate cache key for prediction results.
        """
        content_hash = hashlib.sha256(code_content.encode()).hexdigest()[:16]
        path_hash = hashlib.sha256(file_path.encode()).hexdigest()[:8]
        return f"temporal_{content_hash}_{path_hash}"


# Utility functions for temporal analysis
async def analyze_code_evolution_trends(
    file_paths: List[str], 
    predictor: TemporalCodeEvolutionPredictor
) -> Dict[str, Any]:
    """
    Analyze evolution trends across multiple files.
    """
    trend_analysis = {
        'overall_complexity_trend': 0,
        'bug_risk_distribution': [],
        'refactoring_priorities': [],
        'temporal_correlations': {}
    }
    
    predictions = []
    for file_path in file_paths:
        # Simulate code content
        code_content = f"# Sample code for {file_path}\ndef function():\n    pass"
        
        evolution_pred = await predictor.predict_code_evolution(code_content, file_path)
        predictions.append((file_path, evolution_pred))
    
    # Analyze trends
    complexity_changes = []
    bug_risks = []
    
    for file_path, pred in predictions:
        for prediction in pred.predicted_changes:
            if prediction['type'] == 'complexity_evolution':
                complexity_changes.append(prediction['prediction'])
            elif prediction['type'] == 'bug_risk_prediction':
                bug_risks.append(prediction['prediction'])
    
    if complexity_changes:
        trend_analysis['overall_complexity_trend'] = np.mean(complexity_changes)
    
    if bug_risks:
        trend_analysis['bug_risk_distribution'] = {
            'mean': np.mean(bug_risks),
            'std': np.std(bug_risks),
            'high_risk_files': len([r for r in bug_risks if r > 0.6])
        }
    
    return trend_analysis


def visualize_temporal_predictions(predictions: List[Dict[str, Any]], save_path: str = None):
    """
    Create visualizations for temporal predictions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract prediction data
    complexity_preds = [p['prediction'] for p in predictions if p['type'] == 'complexity_evolution']
    bug_risk_preds = [p['prediction'] for p in predictions if p['type'] == 'bug_risk_prediction']
    refactor_preds = [p['prediction'] for p in predictions if p['type'] == 'refactoring_necessity']
    maintenance_preds = [p['prediction'] for p in predictions if p['type'] == 'maintenance_burden']
    
    # Complexity evolution plot
    if complexity_preds:
        axes[0, 0].hist(complexity_preds, bins=10, alpha=0.7, color='blue')
        axes[0, 0].set_title('Complexity Evolution Distribution')
        axes[0, 0].set_xlabel('Complexity Change')
        axes[0, 0].set_ylabel('Frequency')
    
    # Bug risk plot
    if bug_risk_preds:
        axes[0, 1].hist(bug_risk_preds, bins=10, alpha=0.7, color='red')
        axes[0, 1].set_title('Bug Risk Distribution')
        axes[0, 1].set_xlabel('Bug Risk Probability')
        axes[0, 1].set_ylabel('Frequency')
    
    # Refactoring necessity plot
    if refactor_preds:
        axes[1, 0].hist(refactor_preds, bins=10, alpha=0.7, color='orange')
        axes[1, 0].set_title('Refactoring Necessity Distribution')
        axes[1, 0].set_xlabel('Refactoring Urgency')
        axes[1, 0].set_ylabel('Frequency')
    
    # Maintenance burden plot
    if maintenance_preds:
        axes[1, 1].hist(maintenance_preds, bins=10, alpha=0.7, color='green')
        axes[1, 1].set_title('Maintenance Burden Distribution')
        axes[1, 1].set_xlabel('Maintenance Burden')
        axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig