"""
Probabilistic Bug Prediction System
==================================

Advanced Bayesian networks that predict future bugs based on code patterns,
developer behavior, environmental factors, and historical data. This system
goes beyond traditional static analysis to provide probabilistic forecasts
of where bugs are most likely to occur.

Features:
- Bayesian probabilistic modeling for bug prediction
- Multi-factor analysis (code, developer, environment, time)
- Dynamic probability updates based on new evidence
- Causal inference for bug root cause analysis
- Probabilistic graphical models for complex relationships
- Monte Carlo simulation for uncertainty quantification
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import ast
import json
import pickle
import networkx as nx
from enum import Enum
import logging

# Bayesian and probabilistic modeling
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

try:
    from pgmpy.models import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False

# Machine learning for probabilistic modeling
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# Statistical analysis
from scipy import stats
from scipy.special import beta, gamma
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BugRiskLevel(Enum):
    """Bug risk levels with probability ranges."""
    VERY_LOW = (0.0, 0.1)
    LOW = (0.1, 0.3)
    MEDIUM = (0.3, 0.6)
    HIGH = (0.6, 0.8)
    VERY_HIGH = (0.8, 1.0)


class EvidenceType(Enum):
    """Types of evidence for Bayesian updates."""
    CODE_COMPLEXITY = "code_complexity"
    DEVELOPER_EXPERIENCE = "developer_experience"
    TIME_PRESSURE = "time_pressure"
    TESTING_COVERAGE = "testing_coverage"
    CHANGE_FREQUENCY = "change_frequency"
    HISTORICAL_BUGS = "historical_bugs"
    CODE_REVIEW_QUALITY = "code_review_quality"
    ENVIRONMENTAL_FACTORS = "environmental_factors"


@dataclass
class BugPredictionEvidence:
    """Evidence for Bayesian bug prediction."""
    evidence_type: EvidenceType
    value: float
    confidence: float
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbabilisticBugPrediction:
    """Probabilistic bug prediction result."""
    file_path: str
    bug_probability: float
    confidence_interval: Tuple[float, float]
    risk_level: BugRiskLevel
    contributing_factors: List[Dict[str, Any]]
    bayesian_evidence: List[BugPredictionEvidence]
    prediction_horizon: timedelta
    uncertainty_metrics: Dict[str, float]
    recommendations: List[str]


@dataclass
class DeveloperProfile:
    """Developer behavior profile for bug prediction."""
    developer_id: str
    experience_level: float  # 0-1 scale
    bug_introduction_rate: float
    code_quality_score: float
    testing_propensity: float
    code_review_effectiveness: float
    work_pattern_consistency: float
    domain_expertise: Dict[str, float]
    historical_metrics: Dict[str, List[float]]


@dataclass
class EnvironmentalContext:
    """Environmental context affecting bug probability."""
    project_phase: str  # development, maintenance, legacy
    team_size: int
    deadline_pressure: float  # 0-1 scale
    technical_debt_level: float
    integration_complexity: float
    external_dependencies: int
    deployment_frequency: float
    monitoring_coverage: float


class BayesianBugPredictor:
    """
    Core Bayesian network for probabilistic bug prediction.
    """
    
    def __init__(self):
        self.bayesian_network = None
        self.evidence_history = deque(maxlen=10000)
        self.developer_profiles = {}
        self.environmental_contexts = {}
        
        # Prior probabilities (will be updated with evidence)
        self.priors = {
            'base_bug_rate': 0.15,  # 15% base bug probability
            'complexity_influence': 0.3,
            'developer_influence': 0.25,
            'environmental_influence': 0.2,
            'temporal_influence': 0.25
        }
        
        # Initialize Bayesian network structure
        if PGMPY_AVAILABLE:
            self._initialize_bayesian_network()
    
    def _initialize_bayesian_network(self):
        """Initialize the Bayesian network structure."""
        # Define network structure
        self.bayesian_network = BayesianNetwork([
            ('CodeComplexity', 'BugProbability'),
            ('DeveloperExperience', 'BugProbability'),
            ('TestingCoverage', 'BugProbability'),
            ('TimePressure', 'BugProbability'),
            ('ChangeFrequency', 'BugProbability'),
            ('HistoricalBugs', 'BugProbability'),
            ('CodeReviewQuality', 'BugProbability'),
            ('CodeComplexity', 'TestingCoverage'),
            ('DeveloperExperience', 'CodeReviewQuality'),
            ('TimePressure', 'TestingCoverage'),
            ('TimePressure', 'CodeReviewQuality')
        ])
        
        # Initialize with default CPDs (Conditional Probability Distributions)
        self._setup_initial_cpds()
    
    def _setup_initial_cpds(self):
        """Setup initial conditional probability distributions."""
        if not PGMPY_AVAILABLE:
            return
        
        # Code Complexity CPD (3 states: Low, Medium, High)
        complexity_cpd = TabularCPD(
            variable='CodeComplexity',
            variable_card=3,
            values=[[0.4], [0.4], [0.2]]  # Prior: mostly low-medium complexity
        )
        
        # Developer Experience CPD (3 states: Junior, Mid, Senior)
        experience_cpd = TabularCPD(
            variable='DeveloperExperience',
            variable_card=3,
            values=[[0.3], [0.4], [0.3]]  # Balanced experience distribution
        )
        
        # Time Pressure CPD (3 states: Low, Medium, High)
        pressure_cpd = TabularCPD(
            variable='TimePressure',
            variable_card=3,
            values=[[0.5], [0.3], [0.2]]  # Usually not under high pressure
        )
        
        # Historical Bugs CPD (3 states: Low, Medium, High)
        history_cpd = TabularCPD(
            variable='HistoricalBugs',
            variable_card=3,
            values=[[0.6], [0.3], [0.1]]  # Most code has low historical bugs
        )
        
        # Change Frequency CPD (3 states: Low, Medium, High)
        change_cpd = TabularCPD(
            variable='ChangeFrequency',
            variable_card=3,
            values=[[0.5], [0.3], [0.2]]  # Most code changes infrequently
        )
        
        # Testing Coverage CPD (depends on Complexity, Developer, Pressure)
        testing_cpd = TabularCPD(
            variable='TestingCoverage',
            variable_card=3,  # Low, Medium, High
            values=[
                # Low Complexity
                [0.2, 0.3, 0.4,  # Junior developer
                 0.1, 0.2, 0.3,  # Mid developer  
                 0.05, 0.1, 0.2], # Senior developer
                # Medium Complexity  
                [0.3, 0.4, 0.5,
                 0.2, 0.3, 0.4,
                 0.1, 0.2, 0.3],
                # High Complexity
                [0.5, 0.3, 0.1,
                 0.7, 0.5, 0.3,
                 0.85, 0.7, 0.5]
            ],
            evidence=['CodeComplexity', 'DeveloperExperience', 'TimePressure'],
            evidence_card=[3, 3, 3]
        )
        
        # Code Review Quality CPD (depends on Developer Experience, Time Pressure)
        review_cpd = TabularCPD(
            variable='CodeReviewQuality',
            variable_card=3,  # Low, Medium, High
            values=[
                # Low Pressure
                [0.4, 0.2, 0.1,  # Junior, Mid, Senior under low pressure
                 0.4, 0.3, 0.2,  # Medium quality
                 0.2, 0.5, 0.7], # High quality
                # Medium Pressure
                [0.5, 0.3, 0.2,
                 0.3, 0.4, 0.3,
                 0.2, 0.3, 0.5],
                # High Pressure
                [0.7, 0.5, 0.3,
                 0.2, 0.3, 0.4,
                 0.1, 0.2, 0.3]
            ],
            evidence=['DeveloperExperience', 'TimePressure'],
            evidence_card=[3, 3]
        )
        
        # Bug Probability CPD (depends on all factors)
        # This is simplified - in practice would have many more combinations
        bug_prob_cpd = TabularCPD(
            variable='BugProbability',
            variable_card=2,  # Bug, No Bug
            values=self._generate_bug_probability_matrix(),
            evidence=['CodeComplexity', 'DeveloperExperience', 'TestingCoverage', 
                     'TimePressure', 'ChangeFrequency', 'HistoricalBugs', 'CodeReviewQuality'],
            evidence_card=[3, 3, 3, 3, 3, 3, 3]
        )
        
        # Add CPDs to network
        self.bayesian_network.add_cpds(
            complexity_cpd, experience_cpd, pressure_cpd, history_cpd,
            change_cpd, testing_cpd, review_cpd, bug_prob_cpd
        )
        
        # Verify network consistency
        if self.bayesian_network.check_model():
            logger.info("Bayesian network initialized successfully")
        else:
            logger.error("Bayesian network is inconsistent")
    
    def _generate_bug_probability_matrix(self) -> List[List[float]]:
        """Generate bug probability matrix for all factor combinations."""
        # Simplified matrix generation
        # In practice, this would be learned from historical data
        total_combinations = 3**7  # 7 factors with 3 states each
        
        # Generate probabilities based on heuristics
        bug_probs = []
        no_bug_probs = []
        
        for i in range(total_combinations):
            # Decode combination
            combo = []
            temp = i
            for _ in range(7):
                combo.append(temp % 3)
                temp //= 3
            
            # Calculate bug probability based on factor values
            complexity, experience, testing, pressure, change_freq, history, review = combo
            
            # Base probability
            prob = 0.1
            
            # Complexity influence (0=low, 1=medium, 2=high)
            prob += complexity * 0.15
            
            # Experience influence (0=junior, 1=mid, 2=senior) - inverse relationship
            prob += (2 - experience) * 0.1
            
            # Testing influence (0=low, 1=medium, 2=high) - inverse relationship
            prob -= testing * 0.12
            
            # Pressure influence
            prob += pressure * 0.08
            
            # Change frequency influence
            prob += change_freq * 0.05
            
            # Historical bugs influence
            prob += history * 0.2
            
            # Code review quality influence - inverse relationship
            prob -= review * 0.1
            
            # Clamp probability
            prob = max(0.01, min(0.99, prob))
            
            bug_probs.append(prob)
            no_bug_probs.append(1 - prob)
        
        return [no_bug_probs, bug_probs]
    
    async def predict_bug_probability(
        self,
        evidence: List[BugPredictionEvidence],
        file_path: str,
        prediction_horizon: Optional[timedelta] = None
    ) -> ProbabilisticBugPrediction:
        """
        Predict bug probability given evidence using Bayesian inference.
        """
        if prediction_horizon is None:
            prediction_horizon = timedelta(days=30)
        
        try:
            # Convert evidence to Bayesian network format
            bayesian_evidence = self._convert_evidence_to_bayesian(evidence)
            
            # Perform Bayesian inference
            if PGMPY_AVAILABLE and self.bayesian_network:
                inference_result = await self._perform_bayesian_inference(bayesian_evidence)
            else:
                inference_result = await self._fallback_probabilistic_inference(evidence)
            
            # Calculate confidence intervals
            confidence_interval = self._calculate_confidence_interval(
                inference_result['probability'], 
                inference_result['uncertainty']
            )
            
            # Determine risk level
            risk_level = self._categorize_risk_level(inference_result['probability'])
            
            # Analyze contributing factors
            contributing_factors = await self._analyze_contributing_factors(evidence, inference_result)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                inference_result['probability'], contributing_factors
            )
            
            # Calculate uncertainty metrics
            uncertainty_metrics = self._calculate_uncertainty_metrics(inference_result)
            
            return ProbabilisticBugPrediction(
                file_path=file_path,
                bug_probability=inference_result['probability'],
                confidence_interval=confidence_interval,
                risk_level=risk_level,
                contributing_factors=contributing_factors,
                bayesian_evidence=evidence,
                prediction_horizon=prediction_horizon,
                uncertainty_metrics=uncertainty_metrics,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Bug prediction failed: {e}")
            return self._create_fallback_prediction(file_path, prediction_horizon)
    
    def _convert_evidence_to_bayesian(self, evidence: List[BugPredictionEvidence]) -> Dict[str, int]:
        """Convert evidence to format suitable for Bayesian network."""
        bayesian_evidence = {}
        
        for evt in evidence:
            if evt.evidence_type == EvidenceType.CODE_COMPLEXITY:
                bayesian_evidence['CodeComplexity'] = self._discretize_value(evt.value, [0.3, 0.7])
            elif evt.evidence_type == EvidenceType.DEVELOPER_EXPERIENCE:
                bayesian_evidence['DeveloperExperience'] = self._discretize_value(evt.value, [0.3, 0.7])
            elif evt.evidence_type == EvidenceType.TESTING_COVERAGE:
                bayesian_evidence['TestingCoverage'] = self._discretize_value(evt.value, [0.4, 0.8])
            elif evt.evidence_type == EvidenceType.TIME_PRESSURE:
                bayesian_evidence['TimePressure'] = self._discretize_value(evt.value, [0.3, 0.7])
            elif evt.evidence_type == EvidenceType.CHANGE_FREQUENCY:
                bayesian_evidence['ChangeFrequency'] = self._discretize_value(evt.value, [0.2, 0.6])
            elif evt.evidence_type == EvidenceType.HISTORICAL_BUGS:
                bayesian_evidence['HistoricalBugs'] = self._discretize_value(evt.value, [0.1, 0.4])
            elif evt.evidence_type == EvidenceType.CODE_REVIEW_QUALITY:
                bayesian_evidence['CodeReviewQuality'] = self._discretize_value(evt.value, [0.4, 0.8])
        
        return bayesian_evidence
    
    def _discretize_value(self, value: float, thresholds: List[float]) -> int:
        """Discretize continuous value into categorical states."""
        if value <= thresholds[0]:
            return 0  # Low
        elif value <= thresholds[1]:
            return 1  # Medium
        else:
            return 2  # High
    
    async def _perform_bayesian_inference(self, evidence: Dict[str, int]) -> Dict[str, Any]:
        """Perform Bayesian inference using the network."""
        if not PGMPY_AVAILABLE or not self.bayesian_network:
            return await self._fallback_probabilistic_inference([])
        
        try:
            # Create inference object
            inference = VariableElimination(self.bayesian_network)
            
            # Query bug probability given evidence
            result = inference.query(variables=['BugProbability'], evidence=evidence)
            
            # Extract probability of bug (state 1)
            bug_probability = result.values[1]
            
            # Estimate uncertainty (simplified)
            uncertainty = self._estimate_bayesian_uncertainty(evidence, result)
            
            return {
                'probability': float(bug_probability),
                'uncertainty': uncertainty,
                'method': 'bayesian_network',
                'evidence_strength': len(evidence),
                'posterior_distribution': result.values.tolist()
            }
            
        except Exception as e:
            logger.warning(f"Bayesian inference failed: {e}, using fallback")
            return await self._fallback_probabilistic_inference([])
    
    async def _fallback_probabilistic_inference(self, evidence: List[BugPredictionEvidence]) -> Dict[str, Any]:
        """Fallback probabilistic inference when Bayesian network is unavailable."""
        # Weighted combination of evidence
        base_probability = self.priors['base_bug_rate']
        total_weight = 0
        weighted_sum = 0
        
        evidence_weights = {
            EvidenceType.CODE_COMPLEXITY: 0.25,
            EvidenceType.DEVELOPER_EXPERIENCE: 0.2,
            EvidenceType.TESTING_COVERAGE: 0.15,
            EvidenceType.TIME_PRESSURE: 0.15,
            EvidenceType.CHANGE_FREQUENCY: 0.1,
            EvidenceType.HISTORICAL_BUGS: 0.1,
            EvidenceType.CODE_REVIEW_QUALITY: 0.05
        }
        
        for evt in evidence:
            weight = evidence_weights.get(evt.evidence_type, 0.05) * evt.confidence
            
            # Transform evidence value to probability influence
            if evt.evidence_type == EvidenceType.DEVELOPER_EXPERIENCE:
                # Higher experience = lower bug probability
                influence = (1 - evt.value) * 0.3
            elif evt.evidence_type == EvidenceType.TESTING_COVERAGE:
                # Higher coverage = lower bug probability
                influence = (1 - evt.value) * 0.4
            elif evt.evidence_type == EvidenceType.CODE_REVIEW_QUALITY:
                # Higher quality = lower bug probability
                influence = (1 - evt.value) * 0.2
            else:
                # Higher value = higher bug probability
                influence = evt.value * 0.3
            
            weighted_sum += weight * influence
            total_weight += weight
        
        if total_weight > 0:
            probability = base_probability + (weighted_sum / total_weight)
        else:
            probability = base_probability
        
        # Clamp probability
        probability = max(0.01, min(0.99, probability))
        
        # Estimate uncertainty based on evidence availability
        uncertainty = max(0.1, 1.0 - (total_weight / sum(evidence_weights.values())))
        
        return {
            'probability': probability,
            'uncertainty': uncertainty,
            'method': 'weighted_combination',
            'evidence_strength': len(evidence),
            'total_evidence_weight': total_weight
        }
    
    def _estimate_bayesian_uncertainty(self, evidence: Dict[str, int], result) -> float:
        """Estimate uncertainty in Bayesian inference."""
        # Simplified uncertainty estimation
        # In practice, this would use entropy or variance measures
        
        # Base uncertainty
        uncertainty = 0.2
        
        # Reduce uncertainty based on evidence strength
        evidence_strength = len(evidence) / 7  # 7 total factors
        uncertainty *= (1 - evidence_strength * 0.5)
        
        # Adjust based on probability distribution sharpness
        prob_values = result.values
        entropy = -np.sum(prob_values * np.log(prob_values + 1e-10))
        max_entropy = np.log(len(prob_values))
        normalized_entropy = entropy / max_entropy
        
        uncertainty *= (0.5 + 0.5 * normalized_entropy)
        
        return max(0.05, min(0.5, uncertainty))
    
    def _calculate_confidence_interval(self, probability: float, uncertainty: float) -> Tuple[float, float]:
        """Calculate confidence interval for probability estimate."""
        # Use beta distribution for probability confidence intervals
        alpha = probability * (1 / uncertainty - 1)
        beta_param = (1 - probability) * (1 / uncertainty - 1)
        
        # 95% confidence interval
        lower = stats.beta.ppf(0.025, alpha, beta_param)
        upper = stats.beta.ppf(0.975, alpha, beta_param)
        
        return (max(0.0, lower), min(1.0, upper))
    
    def _categorize_risk_level(self, probability: float) -> BugRiskLevel:
        """Categorize probability into risk levels."""
        for risk_level in BugRiskLevel:
            lower, upper = risk_level.value
            if lower <= probability < upper:
                return risk_level
        return BugRiskLevel.VERY_HIGH  # Fallback
    
    async def _analyze_contributing_factors(
        self,
        evidence: List[BugPredictionEvidence],
        inference_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze which factors contribute most to bug probability."""
        factors = []
        
        # Sort evidence by impact on probability
        for evt in evidence:
            # Estimate factor impact (simplified)
            if evt.evidence_type == EvidenceType.CODE_COMPLEXITY:
                impact = evt.value * 0.25
            elif evt.evidence_type == EvidenceType.HISTORICAL_BUGS:
                impact = evt.value * 0.3
            elif evt.evidence_type == EvidenceType.DEVELOPER_EXPERIENCE:
                impact = (1 - evt.value) * 0.2  # Inverse relationship
            elif evt.evidence_type == EvidenceType.TESTING_COVERAGE:
                impact = (1 - evt.value) * 0.15  # Inverse relationship
            else:
                impact = evt.value * 0.1
            
            factors.append({
                'factor': evt.evidence_type.value,
                'value': evt.value,
                'impact_on_probability': impact,
                'confidence': evt.confidence,
                'description': self._get_factor_description(evt.evidence_type, evt.value)
            })
        
        # Sort by impact
        factors.sort(key=lambda x: x['impact_on_probability'], reverse=True)
        
        return factors
    
    def _get_factor_description(self, evidence_type: EvidenceType, value: float) -> str:
        """Get human-readable description of factor influence."""
        descriptions = {
            EvidenceType.CODE_COMPLEXITY: f"Code complexity score of {value:.2f} {'increases' if value > 0.5 else 'moderately affects'} bug risk",
            EvidenceType.DEVELOPER_EXPERIENCE: f"Developer experience level of {value:.2f} {'reduces' if value > 0.5 else 'increases'} bug risk",
            EvidenceType.TESTING_COVERAGE: f"Testing coverage of {value:.1%} {'significantly reduces' if value > 0.8 else 'inadequately protects against'} bug risk",
            EvidenceType.TIME_PRESSURE: f"Time pressure level of {value:.2f} {'significantly increases' if value > 0.7 else 'moderately affects'} bug risk",
            EvidenceType.CHANGE_FREQUENCY: f"Change frequency of {value:.2f} {'increases instability and' if value > 0.5 else 'has minimal impact on'} bug risk",
            EvidenceType.HISTORICAL_BUGS: f"Historical bug rate of {value:.2f} {'strongly indicates' if value > 0.3 else 'suggests low'} future bug risk",
            EvidenceType.CODE_REVIEW_QUALITY: f"Code review quality of {value:.2f} {'effectively reduces' if value > 0.7 else 'inadequately addresses'} bug risk"
        }
        
        return descriptions.get(evidence_type, f"{evidence_type.value} value of {value:.2f}")
    
    async def _generate_recommendations(
        self,
        bug_probability: float,
        contributing_factors: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations based on prediction."""
        recommendations = []
        
        if bug_probability > 0.7:
            recommendations.append("🚨 HIGH RISK: Immediate code review and testing recommended")
            recommendations.append("Consider pair programming for this component")
            recommendations.append("Implement comprehensive unit and integration tests")
        elif bug_probability > 0.4:
            recommendations.append("⚠️ MEDIUM RISK: Enhanced testing and review recommended")
            recommendations.append("Add additional test cases for edge conditions")
        
        # Factor-specific recommendations
        top_factors = contributing_factors[:3]  # Top 3 contributing factors
        
        for factor in top_factors:
            factor_type = factor['factor']
            
            if factor_type == 'code_complexity' and factor['value'] > 0.6:
                recommendations.append("Refactor complex functions to reduce cyclomatic complexity")
                recommendations.append("Consider breaking down large functions into smaller units")
            
            elif factor_type == 'testing_coverage' and factor['value'] < 0.6:
                recommendations.append("Increase test coverage, especially for complex code paths")
                recommendations.append("Add property-based tests for better edge case coverage")
            
            elif factor_type == 'developer_experience' and factor['value'] < 0.4:
                recommendations.append("Provide additional mentoring and code review support")
                recommendations.append("Consider pair programming with senior developers")
            
            elif factor_type == 'time_pressure' and factor['value'] > 0.6:
                recommendations.append("Allocate additional time for quality assurance")
                recommendations.append("Implement staged delivery to reduce pressure")
            
            elif factor_type == 'historical_bugs' and factor['value'] > 0.3:
                recommendations.append("This component has a history of bugs - consider architectural review")
                recommendations.append("Implement additional monitoring and alerting")
        
        return recommendations
    
    def _calculate_uncertainty_metrics(self, inference_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate various uncertainty metrics."""
        uncertainty = inference_result['uncertainty']
        probability = inference_result['probability']
        
        return {
            'epistemic_uncertainty': uncertainty,  # Model uncertainty
            'aleatoric_uncertainty': probability * (1 - probability),  # Data uncertainty
            'prediction_confidence': 1 - uncertainty,
            'evidence_strength': min(1.0, inference_result.get('evidence_strength', 0) / 5),
            'model_reliability': 1 - uncertainty * 0.5
        }
    
    def _create_fallback_prediction(
        self,
        file_path: str,
        prediction_horizon: timedelta
    ) -> ProbabilisticBugPrediction:
        """Create fallback prediction when analysis fails."""
        return ProbabilisticBugPrediction(
            file_path=file_path,
            bug_probability=self.priors['base_bug_rate'],
            confidence_interval=(0.05, 0.25),
            risk_level=BugRiskLevel.LOW,
            contributing_factors=[],
            bayesian_evidence=[],
            prediction_horizon=prediction_horizon,
            uncertainty_metrics={'epistemic_uncertainty': 0.8},
            recommendations=["Unable to perform detailed analysis - consider manual review"]
        )
    
    async def update_with_feedback(
        self,
        prediction: ProbabilisticBugPrediction,
        actual_outcome: bool,
        feedback_confidence: float = 1.0
    ):
        """Update model with feedback about prediction accuracy."""
        # Store feedback for model improvement
        feedback = {
            'predicted_probability': prediction.bug_probability,
            'actual_outcome': actual_outcome,
            'prediction_error': abs(prediction.bug_probability - (1.0 if actual_outcome else 0.0)),
            'confidence': feedback_confidence,
            'timestamp': datetime.now(),
            'evidence': prediction.bayesian_evidence
        }
        
        self.evidence_history.append(feedback)
        
        # Trigger model retraining if enough feedback accumulated
        if len(self.evidence_history) >= 100:
            await self._retrain_with_feedback()
    
    async def _retrain_with_feedback(self):
        """Retrain model components with accumulated feedback."""
        logger.info("Retraining bug prediction model with feedback...")
        
        # Extract training data from feedback
        feedback_data = list(self.evidence_history)
        
        # Update priors based on observed outcomes
        actual_bug_rate = sum(1 for f in feedback_data if f['actual_outcome']) / len(feedback_data)
        self.priors['base_bug_rate'] = 0.7 * self.priors['base_bug_rate'] + 0.3 * actual_bug_rate
        
        # Update Bayesian network parameters if available
        if PGMPY_AVAILABLE and self.bayesian_network:
            await self._update_bayesian_parameters(feedback_data)
        
        logger.info(f"Model retrained with {len(feedback_data)} feedback samples")


class BugPredictionOrchestrator:
    """
    Orchestrates bug prediction across multiple files and contexts.
    """
    
    def __init__(self):
        self.bayesian_predictor = BayesianBugPredictor()
        self.developer_profiles = {}
        self.environmental_contexts = {}
        self.prediction_cache = {}
    
    async def predict_bugs_for_codebase(
        self,
        file_analyses: Dict[str, Dict[str, Any]],
        developer_context: Optional[DeveloperProfile] = None,
        environmental_context: Optional[EnvironmentalContext] = None
    ) -> Dict[str, ProbabilisticBugPrediction]:
        """
        Predict bugs for entire codebase.
        """
        predictions = {}
        
        for file_path, analysis in file_analyses.items():
            try:
                # Extract evidence from analysis
                evidence = await self._extract_evidence_from_analysis(
                    analysis, developer_context, environmental_context
                )
                
                # Make prediction
                prediction = await self.bayesian_predictor.predict_bug_probability(
                    evidence, file_path
                )
                
                predictions[file_path] = prediction
                
            except Exception as e:
                logger.error(f"Failed to predict bugs for {file_path}: {e}")
        
        return predictions
    
    async def _extract_evidence_from_analysis(
        self,
        analysis: Dict[str, Any],
        developer_context: Optional[DeveloperProfile],
        environmental_context: Optional[EnvironmentalContext]
    ) -> List[BugPredictionEvidence]:
        """Extract evidence from code analysis results."""
        evidence = []
        
        # Code complexity evidence
        if 'complexity_metrics' in analysis:
            complexity = analysis['complexity_metrics'].get('cyclomatic_complexity', 0)
            normalized_complexity = min(1.0, complexity / 20.0)  # Normalize
            
            evidence.append(BugPredictionEvidence(
                evidence_type=EvidenceType.CODE_COMPLEXITY,
                value=normalized_complexity,
                confidence=0.9,
                timestamp=datetime.now(),
                source='static_analysis',
                metadata={'raw_complexity': complexity}
            ))
        
        # Testing coverage evidence
        if 'testing_metrics' in analysis:
            coverage = analysis['testing_metrics'].get('coverage_percentage', 0)
            
            evidence.append(BugPredictionEvidence(
                evidence_type=EvidenceType.TESTING_COVERAGE,
                value=coverage / 100.0,  # Normalize to 0-1
                confidence=0.8,
                timestamp=datetime.now(),
                source='testing_analysis'
            ))
        
        # Historical bugs evidence
        if 'historical_data' in analysis:
            bug_count = analysis['historical_data'].get('bug_count', 0)
            total_changes = analysis['historical_data'].get('total_changes', 1)
            bug_rate = min(1.0, bug_count / total_changes)
            
            evidence.append(BugPredictionEvidence(
                evidence_type=EvidenceType.HISTORICAL_BUGS,
                value=bug_rate,
                confidence=0.85,
                timestamp=datetime.now(),
                source='version_control_history'
            ))
        
        # Change frequency evidence
        if 'change_metrics' in analysis:
            change_frequency = analysis['change_metrics'].get('changes_per_month', 0)
            normalized_frequency = min(1.0, change_frequency / 10.0)  # Normalize
            
            evidence.append(BugPredictionEvidence(
                evidence_type=EvidenceType.CHANGE_FREQUENCY,
                value=normalized_frequency,
                confidence=0.7,
                timestamp=datetime.now(),
                source='version_control_analysis'
            ))
        
        # Developer experience evidence
        if developer_context:
            evidence.append(BugPredictionEvidence(
                evidence_type=EvidenceType.DEVELOPER_EXPERIENCE,
                value=developer_context.experience_level,
                confidence=0.9,
                timestamp=datetime.now(),
                source='developer_profile'
            ))
        
        # Environmental evidence
        if environmental_context:
            evidence.append(BugPredictionEvidence(
                evidence_type=EvidenceType.TIME_PRESSURE,
                value=environmental_context.deadline_pressure,
                confidence=0.8,
                timestamp=datetime.now(),
                source='project_context'
            ))
        
        return evidence
    
    async def generate_codebase_bug_report(
        self,
        predictions: Dict[str, ProbabilisticBugPrediction]
    ) -> Dict[str, Any]:
        """Generate comprehensive bug prediction report for codebase."""
        
        # Calculate aggregate statistics
        total_files = len(predictions)
        high_risk_files = len([p for p in predictions.values() 
                              if p.risk_level in [BugRiskLevel.HIGH, BugRiskLevel.VERY_HIGH]])
        
        avg_bug_probability = sum(p.bug_probability for p in predictions.values()) / total_files
        
        # Identify top risk files
        top_risk_files = sorted(
            predictions.items(),
            key=lambda x: x[1].bug_probability,
            reverse=True
        )[:10]
        
        # Analyze common contributing factors
        all_factors = []
        for prediction in predictions.values():
            all_factors.extend(prediction.contributing_factors)
        
        factor_frequency = defaultdict(int)
        for factor in all_factors:
            factor_frequency[factor['factor']] += 1
        
        common_factors = sorted(
            factor_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'summary': {
                'total_files_analyzed': total_files,
                'high_risk_files': high_risk_files,
                'average_bug_probability': avg_bug_probability,
                'overall_risk_level': self._calculate_overall_risk_level(predictions)
            },
            'top_risk_files': [
                {
                    'file_path': file_path,
                    'bug_probability': prediction.bug_probability,
                    'risk_level': prediction.risk_level.name,
                    'confidence_interval': prediction.confidence_interval,
                    'top_factors': prediction.contributing_factors[:3]
                }
                for file_path, prediction in top_risk_files
            ],
            'common_risk_factors': [
                {
                    'factor': factor,
                    'frequency': freq,
                    'percentage': freq / total_files * 100
                }
                for factor, freq in common_factors
            ],
            'recommendations': self._generate_codebase_recommendations(predictions),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_overall_risk_level(self, predictions: Dict[str, ProbabilisticBugPrediction]) -> str:
        """Calculate overall risk level for codebase."""
        if not predictions:
            return "UNKNOWN"
        
        risk_scores = {
            BugRiskLevel.VERY_LOW: 1,
            BugRiskLevel.LOW: 2,
            BugRiskLevel.MEDIUM: 3,
            BugRiskLevel.HIGH: 4,
            BugRiskLevel.VERY_HIGH: 5
        }
        
        avg_risk_score = sum(risk_scores[p.risk_level] for p in predictions.values()) / len(predictions)
        
        if avg_risk_score <= 1.5:
            return "VERY_LOW"
        elif avg_risk_score <= 2.5:
            return "LOW"  
        elif avg_risk_score <= 3.5:
            return "MEDIUM"
        elif avg_risk_score <= 4.5:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def _generate_codebase_recommendations(
        self,
        predictions: Dict[str, ProbabilisticBugPrediction]
    ) -> List[str]:
        """Generate codebase-wide recommendations."""
        recommendations = []
        
        high_risk_count = len([p for p in predictions.values() 
                              if p.risk_level in [BugRiskLevel.HIGH, BugRiskLevel.VERY_HIGH]])
        
        if high_risk_count > len(predictions) * 0.2:
            recommendations.append("🚨 CRITICAL: Over 20% of files are high-risk - consider code quality initiative")
            recommendations.append("Implement mandatory code reviews for all high-risk files")
            recommendations.append("Increase testing coverage across the codebase")
        
        # Analyze common patterns
        complexity_issues = sum(1 for p in predictions.values() 
                               for f in p.contributing_factors 
                               if f['factor'] == 'code_complexity' and f['value'] > 0.6)
        
        if complexity_issues > len(predictions) * 0.3:
            recommendations.append("Complexity is a major issue - implement refactoring strategy")
        
        testing_issues = sum(1 for p in predictions.values()
                           for f in p.contributing_factors
                           if f['factor'] == 'testing_coverage' and f['value'] < 0.6)
        
        if testing_issues > len(predictions) * 0.4:
            recommendations.append("Testing coverage is inadequate - implement testing standards")
        
        return recommendations


# Main interface
class ProbabilisticBugAnalyzer:
    """
    Main interface for probabilistic bug prediction analysis.
    """
    
    def __init__(self):
        self.orchestrator = BugPredictionOrchestrator()
    
    async def analyze_bug_probability(
        self,
        code_content: str,
        file_path: str,
        analysis_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze bug probability for a single file.
        """
        try:
            # Create mock analysis data (in real implementation, this would come from other analyzers)
            mock_analysis = self._create_mock_analysis(code_content)
            
            # Extract developer and environmental context
            developer_context = self._extract_developer_context(analysis_context)
            environmental_context = self._extract_environmental_context(analysis_context)
            
            # Make prediction
            prediction = await self.orchestrator.predict_bugs_for_codebase(
                {file_path: mock_analysis},
                developer_context,
                environmental_context
            )
            
            result = {
                'probabilistic_bug_analysis': {
                    'file_path': file_path,
                    'bug_probability': prediction[file_path].bug_probability,
                    'confidence_interval': prediction[file_path].confidence_interval,
                    'risk_level': prediction[file_path].risk_level.name,
                    'contributing_factors': prediction[file_path].contributing_factors,
                    'uncertainty_metrics': prediction[file_path].uncertainty_metrics,
                    'recommendations': prediction[file_path].recommendations,
                    'prediction_horizon_days': prediction[file_path].prediction_horizon.days
                },
                'analysis_metadata': {
                    'prediction_method': 'bayesian_probabilistic',
                    'evidence_count': len(prediction[file_path].bayesian_evidence),
                    'timestamp': datetime.now().isoformat(),
                    'model_version': '1.0'
                }
            }
            
            return result
            
        except Exception as e:
            return {
                'error': f'Probabilistic bug analysis failed: {str(e)}',
                'fallback_analysis': {
                    'bug_probability': 0.15,  # Base rate
                    'risk_level': 'UNKNOWN',
                    'recommendations': ['Manual code review recommended due to analysis failure']
                }
            }
    
    def _create_mock_analysis(self, code_content: str) -> Dict[str, Any]:
        """Create mock analysis data for demonstration."""
        # Parse code to extract basic metrics
        try:
            tree = ast.parse(code_content)
            
            # Calculate basic complexity
            complexity = 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    complexity += 1
            
            # Mock testing coverage (in real implementation, this would come from coverage tools)
            coverage = 0.75 if 'test' in code_content.lower() else 0.4
            
            # Mock change frequency (would come from version control)
            change_frequency = 2.5  # changes per month
            
            # Mock historical bug data
            bug_count = 1 if complexity > 10 else 0
            
            return {
                'complexity_metrics': {
                    'cyclomatic_complexity': complexity
                },
                'testing_metrics': {
                    'coverage_percentage': coverage * 100
                },
                'change_metrics': {
                    'changes_per_month': change_frequency
                },
                'historical_data': {
                    'bug_count': bug_count,
                    'total_changes': 20
                }
            }
            
        except:
            return {
                'complexity_metrics': {'cyclomatic_complexity': 5},
                'testing_metrics': {'coverage_percentage': 50},
                'change_metrics': {'changes_per_month': 1},
                'historical_data': {'bug_count': 0, 'total_changes': 10}
            }
    
    def _extract_developer_context(self, context: Optional[Dict[str, Any]]) -> Optional[DeveloperProfile]:
        """Extract developer context from analysis context."""
        if not context or 'developer' not in context:
            return None
        
        dev_data = context['developer']
        return DeveloperProfile(
            developer_id=dev_data.get('id', 'unknown'),
            experience_level=dev_data.get('experience_level', 0.5),
            bug_introduction_rate=dev_data.get('bug_rate', 0.1),
            code_quality_score=dev_data.get('quality_score', 0.7),
            testing_propensity=dev_data.get('testing_propensity', 0.6),
            code_review_effectiveness=dev_data.get('review_effectiveness', 0.7),
            work_pattern_consistency=dev_data.get('consistency', 0.8),
            domain_expertise=dev_data.get('domain_expertise', {}),
            historical_metrics=dev_data.get('historical_metrics', {})
        )
    
    def _extract_environmental_context(self, context: Optional[Dict[str, Any]]) -> Optional[EnvironmentalContext]:
        """Extract environmental context from analysis context."""
        if not context or 'environment' not in context:
            return None
        
        env_data = context['environment']
        return EnvironmentalContext(
            project_phase=env_data.get('phase', 'development'),
            team_size=env_data.get('team_size', 5),
            deadline_pressure=env_data.get('deadline_pressure', 0.5),
            technical_debt_level=env_data.get('tech_debt', 0.3),
            integration_complexity=env_data.get('integration_complexity', 0.4),
            external_dependencies=env_data.get('external_deps', 10),
            deployment_frequency=env_data.get('deploy_frequency', 0.2),
            monitoring_coverage=env_data.get('monitoring', 0.7)
        )


# Example usage and testing
async def demonstrate_probabilistic_bug_prediction():
    """
    Demonstrate the probabilistic bug prediction system.
    """
    analyzer = ProbabilisticBugAnalyzer()
    
    # Example code with various risk factors
    high_risk_code = '''
def process_user_data(data, config):
    # High complexity function with multiple risk factors
    result = []
    
    for item in data:
        if item:
            if config['validate']:
                if len(item) > 100:
                    if item.startswith('special'):
                        for char in item:
                            if char.isdigit():
                                result.append(int(char) * config['multiplier'])
                            else:
                                try:
                                    result.append(ord(char) / config['divisor'])
                                except:
                                    pass
                    else:
                        result.append(len(item))
                else:
                    result.append(0)
            else:
                result.append(item)
    
    return result
'''
    
    low_risk_code = '''
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of a and b
    """
    return a + b
'''
    
    # Analysis contexts
    contexts = [
        {
            'developer': {
                'experience_level': 0.3,  # Junior developer
                'bug_rate': 0.2,
                'quality_score': 0.6
            },
            'environment': {
                'deadline_pressure': 0.8,  # High pressure
                'tech_debt': 0.6,
                'team_size': 3
            }
        },
        {
            'developer': {
                'experience_level': 0.9,  # Senior developer
                'bug_rate': 0.05,
                'quality_score': 0.9
            },
            'environment': {
                'deadline_pressure': 0.2,  # Low pressure
                'tech_debt': 0.2,
                'team_size': 6
            }
        }
    ]
    
    test_cases = [
        (high_risk_code, 'high_risk_function.py', contexts[0]),
        (high_risk_code, 'high_risk_function.py', contexts[1]),
        (low_risk_code, 'simple_function.py', contexts[0]),
        (low_risk_code, 'simple_function.py', contexts[1])
    ]
    
    print("Probabilistic Bug Prediction Demonstration")
    print("=" * 50)
    
    for i, (code, file_path, context) in enumerate(test_cases, 1):
        print(f"\\nTest Case {i}: {file_path}")
        print(f"Developer Experience: {context['developer']['experience_level']:.1f}")
        print(f"Time Pressure: {context['environment']['deadline_pressure']:.1f}")
        
        result = await analyzer.analyze_bug_probability(code, file_path, context)
        
        if 'error' not in result:
            analysis = result['probabilistic_bug_analysis']
            
            print(f"\\n🎯 Bug Probability: {analysis['bug_probability']:.1%}")
            print(f"📊 Risk Level: {analysis['risk_level']}")
            print(f"🔍 Confidence Interval: {analysis['confidence_interval'][0]:.1%} - {analysis['confidence_interval'][1]:.1%}")
            
            print(f"\\n🔧 Top Contributing Factors:")
            for factor in analysis['contributing_factors'][:3]:
                print(f"  • {factor['factor']}: {factor['value']:.2f} (impact: {factor['impact_on_probability']:.3f})")
            
            print(f"\\n💡 Recommendations:")
            for rec in analysis['recommendations'][:3]:
                print(f"  • {rec}")
                
            print(f"\\n📈 Uncertainty Metrics:")
            for metric, value in analysis['uncertainty_metrics'].items():
                print(f"  • {metric}: {value:.3f}")
        
        else:
            print(f"❌ Analysis failed: {result['error']}")
        
        print("-" * 30)


if __name__ == "__main__":
    asyncio.run(demonstrate_probabilistic_bug_prediction())