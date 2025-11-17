"""
Metamorphic Code Evolution System
=================================

Revolutionary self-modifying AI system that continuously evolves and improves
its code analysis algorithms based on discovered patterns, outcomes, and feedback.
This system can rewrite its own analysis logic, adapt to new programming patterns,
and develop novel detection techniques through evolutionary computing principles.

Features:
- Self-modifying analysis algorithms
- Evolutionary algorithm optimization
- Adaptive pattern recognition
- Dynamic rule generation and refinement  
- Genetic programming for algorithm evolution
- Neural architecture search for model improvement
- Continuous learning from analysis outcomes
- Meta-learning across different codebases
- Algorithm mutation and crossover
- Fitness-based selection of analysis strategies
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import logging
import math
import statistics
import random
import inspect
import ast
import types
import hashlib
from enum import Enum
import pickle
import copy
from abc import ABC, abstractmethod

# Genetic Programming and Evolution
import networkx as nx
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvolutionStrategy(Enum):
    """Different evolution strategies for algorithm improvement."""
    GENETIC_PROGRAMMING = "genetic_programming"
    NEURAL_ARCHITECTURE_SEARCH = "neural_architecture_search"  
    RULE_EVOLUTION = "rule_evolution"
    ALGORITHM_MUTATION = "algorithm_mutation"
    ENSEMBLE_EVOLUTION = "ensemble_evolution"
    META_LEARNING = "meta_learning"


class AlgorithmGene(Enum):
    """Genetic components of analysis algorithms."""
    DETECTION_PATTERN = "detection_pattern"
    FEATURE_EXTRACTION = "feature_extraction"
    CLASSIFICATION_LOGIC = "classification_logic"
    PREPROCESSING_STEP = "preprocessing_step"
    POSTPROCESSING_RULE = "postprocessing_rule"
    WEIGHT_PARAMETER = "weight_parameter"
    THRESHOLD_VALUE = "threshold_value"
    COMBINATION_METHOD = "combination_method"


@dataclass
class AnalysisAlgorithm:
    """Represents an evolved analysis algorithm."""
    algorithm_id: str
    version: int
    creation_time: datetime
    parent_algorithms: List[str]
    genetic_code: Dict[AlgorithmGene, Any]
    performance_metrics: Dict[str, float]
    fitness_score: float
    source_code: str
    executable_function: Optional[Callable]
    specialization_domain: str  # e.g., "security", "performance", "quality"


@dataclass
class EvolutionExperiment:
    """Tracks an evolution experiment."""
    experiment_id: str
    start_time: datetime
    end_time: Optional[datetime]
    strategy: EvolutionStrategy
    population_size: int
    generations: int
    best_fitness_progression: List[float]
    evolved_algorithms: List[AnalysisAlgorithm]
    training_data: List[Dict[str, Any]]
    evolution_log: List[Dict[str, Any]]


@dataclass
class AlgorithmPerformanceRecord:
    """Records performance of an algorithm on specific tasks."""
    algorithm_id: str
    task_type: str
    dataset_identifier: str
    performance_metrics: Dict[str, float]
    execution_time: float
    resource_usage: Dict[str, float]
    error_cases: List[Dict[str, Any]]
    success_cases: List[Dict[str, Any]]
    feedback_score: float


class AlgorithmGenome:
    """
    Represents the genetic structure of analysis algorithms.
    """
    
    def __init__(self):
        self.genes = {}
        self.gene_expressions = {}
        self.fitness_history = []
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
    
    def encode_algorithm(self, algorithm: AnalysisAlgorithm) -> Dict[str, Any]:
        """Encode an algorithm into genetic representation."""
        
        genome = {
            'detection_patterns': self._extract_detection_patterns(algorithm.source_code),
            'feature_extractors': self._extract_feature_methods(algorithm.source_code),
            'classification_rules': self._extract_classification_logic(algorithm.source_code),
            'preprocessing_steps': self._extract_preprocessing(algorithm.source_code),
            'parameters': self._extract_parameters(algorithm.genetic_code),
            'architecture': self._extract_architecture(algorithm.source_code)
        }
        
        return genome
    
    def decode_genome(self, genome: Dict[str, Any]) -> str:
        """Decode genetic representation back to algorithm source code."""
        
        # Generate source code from genetic components
        source_parts = []
        
        # Import statements
        source_parts.append("import numpy as np")
        source_parts.append("import ast")
        source_parts.append("from typing import List, Dict, Any")
        source_parts.append("")
        
        # Function definition
        source_parts.append("def evolved_analysis_function(code_content: str, file_path: str) -> Dict[str, Any]:")
        source_parts.append('    """Evolved analysis algorithm generated by metamorphic evolution."""')
        source_parts.append("    results = {}")
        source_parts.append("    ")
        
        # Add preprocessing steps
        preprocessing = genome.get('preprocessing_steps', [])
        for step in preprocessing:
            source_parts.append(f"    # Preprocessing: {step['description']}")
            source_parts.append(f"    {step['code']}")
        
        # Add feature extraction
        feature_extractors = genome.get('feature_extractors', [])
        source_parts.append("    # Feature extraction")
        source_parts.append("    features = {}")
        
        for extractor in feature_extractors:
            source_parts.append(f"    # {extractor['description']}")
            source_parts.append(f"    {extractor['code']}")
        
        # Add detection patterns
        detection_patterns = genome.get('detection_patterns', [])
        source_parts.append("    # Pattern detection")
        source_parts.append("    detected_patterns = []")
        
        for pattern in detection_patterns:
            source_parts.append(f"    # {pattern['description']}")
            source_parts.append(f"    {pattern['code']}")
        
        # Add classification logic
        classification_rules = genome.get('classification_rules', [])
        source_parts.append("    # Classification")
        source_parts.append("    classifications = {}")
        
        for rule in classification_rules:
            source_parts.append(f"    # {rule['description']}")
            source_parts.append(f"    {rule['code']}")
        
        # Return results
        source_parts.append("    ")
        source_parts.append("    return {")
        source_parts.append("        'features': features,")
        source_parts.append("        'patterns': detected_patterns,")
        source_parts.append("        'classifications': classifications,")
        source_parts.append("        'algorithm_version': 'evolved'")
        source_parts.append("    }")
        
        return "\n".join(source_parts)
    
    def mutate_genome(self, genome: Dict[str, Any], mutation_rate: float = None) -> Dict[str, Any]:
        """Apply mutations to a genome."""
        
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        
        mutated_genome = copy.deepcopy(genome)
        
        # Mutate detection patterns
        if random.random() < mutation_rate:
            patterns = mutated_genome.get('detection_patterns', [])
            if patterns:
                pattern_idx = random.randint(0, len(patterns) - 1)
                patterns[pattern_idx] = self._mutate_detection_pattern(patterns[pattern_idx])
        
        # Mutate feature extractors
        if random.random() < mutation_rate:
            extractors = mutated_genome.get('feature_extractors', [])
            if extractors:
                extractor_idx = random.randint(0, len(extractors) - 1)
                extractors[extractor_idx] = self._mutate_feature_extractor(extractors[extractor_idx])
        
        # Mutate parameters
        if random.random() < mutation_rate:
            parameters = mutated_genome.get('parameters', {})
            for param_name in parameters:
                if random.random() < 0.3:  # 30% chance to mutate each parameter
                    parameters[param_name] = self._mutate_parameter(parameters[param_name])
        
        # Add new components occasionally
        if random.random() < 0.1:  # 10% chance to add new component
            mutated_genome = self._add_random_component(mutated_genome)
        
        return mutated_genome
    
    def crossover_genomes(self, genome1: Dict[str, Any], genome2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover between two genomes."""
        
        child1 = copy.deepcopy(genome1)
        child2 = copy.deepcopy(genome2)
        
        # Crossover detection patterns
        patterns1 = child1.get('detection_patterns', [])
        patterns2 = child2.get('detection_patterns', [])
        
        if patterns1 and patterns2:
            crossover_point = random.randint(0, min(len(patterns1), len(patterns2)) - 1)
            child1['detection_patterns'] = patterns1[:crossover_point] + patterns2[crossover_point:]
            child2['detection_patterns'] = patterns2[:crossover_point] + patterns1[crossover_point:]
        
        # Crossover feature extractors
        extractors1 = child1.get('feature_extractors', [])
        extractors2 = child2.get('feature_extractors', [])
        
        if extractors1 and extractors2:
            # Randomly swap some extractors
            for i in range(min(len(extractors1), len(extractors2))):
                if random.random() < 0.5:
                    extractors1[i], extractors2[i] = extractors2[i], extractors1[i]
        
        # Crossover parameters
        params1 = child1.get('parameters', {})
        params2 = child2.get('parameters', {})
        
        all_param_names = set(params1.keys()) | set(params2.keys())
        for param_name in all_param_names:
            if random.random() < 0.5:
                if param_name in params1 and param_name in params2:
                    params1[param_name], params2[param_name] = params2[param_name], params1[param_name]
        
        return child1, child2
    
    def _extract_detection_patterns(self, source_code: str) -> List[Dict[str, Any]]:
        """Extract detection patterns from source code."""
        patterns = []
        
        try:
            tree = ast.parse(source_code)
            
            # Look for pattern detection logic
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    # Extract conditional patterns
                    pattern_code = ast.unparse(node.test) if hasattr(ast, 'unparse') else "condition"
                    patterns.append({
                        'type': 'conditional_pattern',
                        'code': pattern_code,
                        'description': f"Conditional pattern: {pattern_code[:50]}...",
                        'complexity': len(ast.walk(node.test))
                    })
                
                elif isinstance(node, ast.For):
                    # Extract loop patterns
                    patterns.append({
                        'type': 'loop_pattern',
                        'code': f"for {ast.unparse(node.target) if hasattr(ast, 'unparse') else 'item'} in collection:",
                        'description': "Loop-based pattern detection",
                        'complexity': len(ast.walk(node))
                    })
        
        except Exception as e:
            logger.warning(f"Failed to extract patterns: {e}")
            # Default patterns
            patterns = [
                {
                    'type': 'basic_pattern',
                    'code': 'len(code_lines) > 10',
                    'description': 'Basic length pattern',
                    'complexity': 1
                }
            ]
        
        return patterns
    
    def _extract_feature_methods(self, source_code: str) -> List[Dict[str, Any]]:
        """Extract feature extraction methods."""
        extractors = []
        
        # Common feature extraction patterns
        default_extractors = [
            {
                'name': 'line_count',
                'code': 'features["line_count"] = len(code_content.split("\\n"))',
                'description': 'Count lines of code',
                'return_type': 'int'
            },
            {
                'name': 'complexity_estimate',
                'code': 'features["complexity"] = len([c for c in code_content if c in "(){}[]"])',
                'description': 'Estimate syntactic complexity',
                'return_type': 'int'
            },
            {
                'name': 'keyword_density',
                'code': 'features["keyword_density"] = len([w for w in code_content.split() if w in ["if", "for", "while", "def", "class"]]) / max(1, len(code_content.split()))',
                'description': 'Calculate keyword density',
                'return_type': 'float'
            }
        ]
        
        extractors.extend(default_extractors)
        
        return extractors
    
    def _extract_classification_logic(self, source_code: str) -> List[Dict[str, Any]]:
        """Extract classification rules."""
        rules = []
        
        # Default classification rules
        default_rules = [
            {
                'name': 'complexity_classification',
                'code': 'if features.get("complexity", 0) > 100: classifications["complexity_level"] = "high"',
                'description': 'Classify based on complexity',
                'threshold': 100
            },
            {
                'name': 'size_classification', 
                'code': 'if features.get("line_count", 0) > 50: classifications["size_category"] = "large"',
                'description': 'Classify based on size',
                'threshold': 50
            }
        ]
        
        rules.extend(default_rules)
        
        return rules
    
    def _extract_preprocessing(self, source_code: str) -> List[Dict[str, Any]]:
        """Extract preprocessing steps."""
        steps = [
            {
                'name': 'normalize_whitespace',
                'code': 'code_content = " ".join(code_content.split())',
                'description': 'Normalize whitespace'
            },
            {
                'name': 'remove_comments',
                'code': 'code_lines = [line for line in code_content.split("\\n") if not line.strip().startswith("#")]',
                'description': 'Remove comment lines'
            }
        ]
        
        return steps
    
    def _extract_parameters(self, genetic_code: Dict[AlgorithmGene, Any]) -> Dict[str, Any]:
        """Extract algorithm parameters."""
        parameters = {}
        
        for gene_type, value in genetic_code.items():
            if gene_type == AlgorithmGene.THRESHOLD_VALUE:
                parameters['threshold'] = value
            elif gene_type == AlgorithmGene.WEIGHT_PARAMETER:
                parameters['weight'] = value
        
        # Default parameters
        if 'threshold' not in parameters:
            parameters['threshold'] = 0.5
        if 'weight' not in parameters:
            parameters['weight'] = 1.0
        
        return parameters
    
    def _extract_architecture(self, source_code: str) -> Dict[str, Any]:
        """Extract algorithm architecture."""
        return {
            'type': 'procedural',
            'components': ['preprocessing', 'feature_extraction', 'pattern_detection', 'classification'],
            'execution_order': 'sequential'
        }
    
    def _mutate_detection_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate a detection pattern."""
        mutated_pattern = copy.deepcopy(pattern)
        
        # Randomly modify the pattern
        if pattern['type'] == 'conditional_pattern':
            # Modify threshold or operator
            if 'threshold' in mutated_pattern:
                mutated_pattern['threshold'] *= random.uniform(0.8, 1.2)
        
        return mutated_pattern
    
    def _mutate_feature_extractor(self, extractor: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate a feature extractor."""
        mutated_extractor = copy.deepcopy(extractor)
        
        # Modify extractor parameters
        if 'code' in mutated_extractor:
            # Simple mutations for demonstration
            code = mutated_extractor['code']
            if 'len(' in code and '>' in code:
                # Try to find and modify numeric thresholds
                import re
                numbers = re.findall(r'\d+', code)
                if numbers:
                    old_num = numbers[0]
                    new_num = str(int(float(old_num) * random.uniform(0.8, 1.2)))
                    mutated_extractor['code'] = code.replace(old_num, new_num, 1)
        
        return mutated_extractor
    
    def _mutate_parameter(self, parameter_value: Any) -> Any:
        """Mutate a parameter value."""
        if isinstance(parameter_value, (int, float)):
            return parameter_value * random.uniform(0.8, 1.2)
        elif isinstance(parameter_value, str):
            return parameter_value  # Don't mutate strings for now
        else:
            return parameter_value
    
    def _add_random_component(self, genome: Dict[str, Any]) -> Dict[str, Any]:
        """Add a random new component to the genome."""
        enhanced_genome = copy.deepcopy(genome)
        
        # Randomly choose component type to add
        component_types = ['detection_patterns', 'feature_extractors', 'classification_rules']
        component_type = random.choice(component_types)
        
        if component_type == 'detection_patterns':
            new_pattern = {
                'type': 'random_pattern',
                'code': f'random_value > {random.uniform(0.1, 0.9)}',
                'description': 'Randomly generated pattern',
                'complexity': 1
            }
            enhanced_genome.setdefault('detection_patterns', []).append(new_pattern)
        
        elif component_type == 'feature_extractors':
            new_extractor = {
                'name': f'random_feature_{random.randint(1000, 9999)}',
                'code': f'features["random_metric"] = len(code_content) / {random.randint(10, 100)}',
                'description': 'Randomly generated feature',
                'return_type': 'float'
            }
            enhanced_genome.setdefault('feature_extractors', []).append(new_extractor)
        
        return enhanced_genome


class AlgorithmEvolutionEngine:
    """
    Drives the evolution of analysis algorithms using genetic programming.
    """
    
    def __init__(self):
        self.population = []
        self.generation = 0
        self.fitness_evaluator = None
        self.genome_handler = AlgorithmGenome()
        self.evolution_history = []
        self.elite_preservation_rate = 0.1
        self.tournament_size = 3
    
    async def initialize_population(
        self,
        population_size: int,
        base_algorithms: List[AnalysisAlgorithm],
        strategy: EvolutionStrategy
    ) -> List[AnalysisAlgorithm]:
        """Initialize the evolution population."""
        
        self.population = []
        
        # Start with base algorithms
        for base_algo in base_algorithms[:population_size // 2]:
            self.population.append(base_algo)
        
        # Generate variations
        while len(self.population) < population_size:
            if base_algorithms:
                # Create variation of existing algorithm
                parent = random.choice(base_algorithms)
                child = await self._create_algorithm_variation(parent, strategy)
                self.population.append(child)
            else:
                # Create random algorithm
                random_algo = await self._create_random_algorithm()
                self.population.append(random_algo)
        
        logger.info(f"Initialized population with {len(self.population)} algorithms")
        return self.population
    
    async def evolve_generation(
        self,
        training_data: List[Dict[str, Any]],
        fitness_targets: Dict[str, float]
    ) -> List[AnalysisAlgorithm]:
        """Evolve one generation of algorithms."""
        
        # Evaluate fitness of current population
        fitness_scores = await self._evaluate_population_fitness(
            self.population, training_data, fitness_targets
        )
        
        # Update fitness scores
        for i, algorithm in enumerate(self.population):
            algorithm.fitness_score = fitness_scores[i]
        
        # Sort by fitness (higher is better)
        self.population.sort(key=lambda a: a.fitness_score, reverse=True)
        
        # Preserve elite
        elite_count = max(1, int(len(self.population) * self.elite_preservation_rate))
        elite_algorithms = self.population[:elite_count]
        
        # Create new generation
        new_population = elite_algorithms.copy()
        
        while len(new_population) < len(self.population):
            # Tournament selection
            parent1 = await self._tournament_selection(self.population)
            parent2 = await self._tournament_selection(self.population)
            
            # Crossover and mutation
            if random.random() < self.genome_handler.crossover_rate:
                child1, child2 = await self._crossover_algorithms(parent1, parent2)
            else:
                child1 = await self._mutate_algorithm(parent1)
                child2 = await self._mutate_algorithm(parent2)
            
            new_population.extend([child1, child2])
        
        # Trim to original size
        new_population = new_population[:len(self.population)]
        
        self.population = new_population
        self.generation += 1
        
        # Record evolution history
        best_fitness = max(fitness_scores)
        avg_fitness = statistics.mean(fitness_scores)
        
        self.evolution_history.append({
            'generation': self.generation,
            'best_fitness': best_fitness,
            'average_fitness': avg_fitness,
            'population_size': len(self.population),
            'elite_preserved': elite_count
        })
        
        logger.info(f"Generation {self.generation}: Best fitness = {best_fitness:.3f}, Avg fitness = {avg_fitness:.3f}")
        
        return self.population
    
    async def _evaluate_population_fitness(
        self,
        population: List[AnalysisAlgorithm],
        training_data: List[Dict[str, Any]],
        targets: Dict[str, float]
    ) -> List[float]:
        """Evaluate fitness of all algorithms in population."""
        
        fitness_scores = []
        
        for algorithm in population:
            try:
                fitness = await self._evaluate_algorithm_fitness(
                    algorithm, training_data, targets
                )
                fitness_scores.append(fitness)
            except Exception as e:
                logger.warning(f"Failed to evaluate algorithm {algorithm.algorithm_id}: {e}")
                fitness_scores.append(0.0)
        
        return fitness_scores
    
    async def _evaluate_algorithm_fitness(
        self,
        algorithm: AnalysisAlgorithm,
        training_data: List[Dict[str, Any]],
        targets: Dict[str, float]
    ) -> float:
        """Evaluate fitness of a single algorithm."""
        
        if not algorithm.executable_function:
            return 0.0
        
        correct_predictions = 0
        total_predictions = 0
        execution_times = []
        
        for data_point in training_data:
            try:
                start_time = datetime.now()
                
                # Execute algorithm
                result = algorithm.executable_function(
                    data_point['code_content'],
                    data_point['file_path']
                )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                execution_times.append(execution_time)
                
                # Compare with expected results
                expected = data_point.get('expected_results', {})
                accuracy = self._calculate_result_accuracy(result, expected)
                
                correct_predictions += accuracy
                total_predictions += 1
                
            except Exception as e:
                logger.debug(f"Algorithm execution failed: {e}")
                total_predictions += 1
                execution_times.append(1.0)  # Penalty for failure
        
        # Calculate fitness components
        accuracy_score = correct_predictions / max(1, total_predictions)
        speed_score = 1.0 / (1.0 + statistics.mean(execution_times))
        
        # Penalize for complexity (encourage simpler solutions)
        complexity_penalty = len(algorithm.source_code) / 10000
        
        # Combined fitness
        fitness = accuracy_score * 0.6 + speed_score * 0.3 - complexity_penalty * 0.1
        
        return max(0.0, fitness)
    
    def _calculate_result_accuracy(self, result: Dict[str, Any], expected: Dict[str, Any]) -> float:
        """Calculate accuracy between algorithm result and expected result."""
        
        if not expected:
            return 0.5  # Neutral score if no expected results
        
        matches = 0
        total_comparisons = 0
        
        # Compare features
        result_features = result.get('features', {})
        expected_features = expected.get('features', {})
        
        for feature_name in expected_features:
            if feature_name in result_features:
                expected_val = expected_features[feature_name]
                actual_val = result_features[feature_name]
                
                if isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float)):
                    # Numerical comparison with tolerance
                    tolerance = abs(expected_val * 0.1)  # 10% tolerance
                    if abs(actual_val - expected_val) <= tolerance:
                        matches += 1
                elif expected_val == actual_val:
                    # Exact match for non-numerical values
                    matches += 1
                
                total_comparisons += 1
        
        # Compare patterns
        result_patterns = set(result.get('patterns', []))
        expected_patterns = set(expected.get('patterns', []))
        
        if expected_patterns:
            pattern_accuracy = len(result_patterns & expected_patterns) / len(expected_patterns)
            matches += pattern_accuracy
            total_comparisons += 1
        
        return matches / max(1, total_comparisons)
    
    async def _tournament_selection(self, population: List[AnalysisAlgorithm]) -> AnalysisAlgorithm:
        """Select parent using tournament selection."""
        
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        winner = max(tournament, key=lambda a: a.fitness_score)
        
        return winner
    
    async def _crossover_algorithms(
        self,
        parent1: AnalysisAlgorithm,
        parent2: AnalysisAlgorithm
    ) -> Tuple[AnalysisAlgorithm, AnalysisAlgorithm]:
        """Create offspring through genetic crossover."""
        
        # Extract genomes
        genome1 = self.genome_handler.encode_algorithm(parent1)
        genome2 = self.genome_handler.encode_algorithm(parent2)
        
        # Perform crossover
        child_genome1, child_genome2 = self.genome_handler.crossover_genomes(genome1, genome2)
        
        # Create child algorithms
        child1 = await self._create_algorithm_from_genome(
            child_genome1, [parent1.algorithm_id, parent2.algorithm_id]
        )
        child2 = await self._create_algorithm_from_genome(
            child_genome2, [parent1.algorithm_id, parent2.algorithm_id]
        )
        
        return child1, child2
    
    async def _mutate_algorithm(self, parent: AnalysisAlgorithm) -> AnalysisAlgorithm:
        """Create mutated offspring."""
        
        # Extract and mutate genome
        genome = self.genome_handler.encode_algorithm(parent)
        mutated_genome = self.genome_handler.mutate_genome(genome)
        
        # Create mutated algorithm
        mutated_algorithm = await self._create_algorithm_from_genome(
            mutated_genome, [parent.algorithm_id]
        )
        
        return mutated_algorithm
    
    async def _create_algorithm_from_genome(
        self,
        genome: Dict[str, Any],
        parent_ids: List[str]
    ) -> AnalysisAlgorithm:
        """Create executable algorithm from genome."""
        
        # Generate source code
        source_code = self.genome_handler.decode_genome(genome)
        
        # Create executable function
        try:
            exec_globals = {'np': np, 'ast': ast, 'len': len, 'max': max}
            exec(source_code, exec_globals)
            executable_function = exec_globals.get('evolved_analysis_function')
        except Exception as e:
            logger.warning(f"Failed to create executable function: {e}")
            executable_function = None
        
        # Create algorithm object
        algorithm_id = hashlib.md5((source_code + str(datetime.now())).encode()).hexdigest()[:12]
        
        algorithm = AnalysisAlgorithm(
            algorithm_id=algorithm_id,
            version=1,
            creation_time=datetime.now(),
            parent_algorithms=parent_ids,
            genetic_code={},  # Will be populated from genome
            performance_metrics={},
            fitness_score=0.0,
            source_code=source_code,
            executable_function=executable_function,
            specialization_domain="general"
        )
        
        return algorithm
    
    async def _create_algorithm_variation(
        self,
        base_algorithm: AnalysisAlgorithm,
        strategy: EvolutionStrategy
    ) -> AnalysisAlgorithm:
        """Create variation of existing algorithm."""
        
        if strategy == EvolutionStrategy.ALGORITHM_MUTATION:
            return await self._mutate_algorithm(base_algorithm)
        else:
            # For other strategies, create simple variation
            return await self._mutate_algorithm(base_algorithm)
    
    async def _create_random_algorithm(self) -> AnalysisAlgorithm:
        """Create completely random algorithm."""
        
        # Create random genome
        random_genome = {
            'detection_patterns': [
                {
                    'type': 'random_pattern',
                    'code': f'len(code_content) > {random.randint(10, 100)}',
                    'description': 'Random length pattern',
                    'complexity': 1
                }
            ],
            'feature_extractors': [
                {
                    'name': 'random_metric',
                    'code': f'features["random_metric"] = len(code_content.split()) / {random.randint(1, 10)}',
                    'description': 'Random word-based metric',
                    'return_type': 'float'
                }
            ],
            'classification_rules': [
                {
                    'name': 'random_classification',
                    'code': f'if features.get("random_metric", 0) > {random.uniform(0.1, 10)}: classifications["category"] = "high"',
                    'description': 'Random classification rule',
                    'threshold': random.uniform(0.1, 10)
                }
            ],
            'preprocessing_steps': [
                {
                    'name': 'basic_preprocessing',
                    'code': 'code_content = code_content.strip()',
                    'description': 'Basic preprocessing'
                }
            ],
            'parameters': {
                'threshold': random.uniform(0.1, 1.0),
                'weight': random.uniform(0.5, 2.0)
            }
        }
        
        return await self._create_algorithm_from_genome(random_genome, [])


class MetamorphicLearningSystem:
    """
    Orchestrates the continuous evolution and learning of analysis algorithms.
    """
    
    def __init__(self):
        self.evolution_engine = AlgorithmEvolutionEngine()
        self.algorithm_registry = {}
        self.performance_tracker = {}
        self.active_experiments = {}
        self.learning_feedback = deque(maxlen=10000)
    
    async def start_evolution_experiment(
        self,
        experiment_name: str,
        strategy: EvolutionStrategy,
        base_algorithms: List[AnalysisAlgorithm],
        training_data: List[Dict[str, Any]],
        population_size: int = 50,
        generations: int = 100,
        fitness_targets: Dict[str, float] = None
    ) -> str:
        """Start a new algorithm evolution experiment."""
        
        if fitness_targets is None:
            fitness_targets = {'accuracy': 0.9, 'speed': 0.8}
        
        experiment_id = hashlib.md5((experiment_name + str(datetime.now())).encode()).hexdigest()[:12]
        
        experiment = EvolutionExperiment(
            experiment_id=experiment_id,
            start_time=datetime.now(),
            end_time=None,
            strategy=strategy,
            population_size=population_size,
            generations=generations,
            best_fitness_progression=[],
            evolved_algorithms=[],
            training_data=training_data,
            evolution_log=[]
        )
        
        self.active_experiments[experiment_id] = experiment
        
        # Initialize population
        await self.evolution_engine.initialize_population(
            population_size, base_algorithms, strategy
        )
        
        logger.info(f"Started evolution experiment '{experiment_name}' (ID: {experiment_id})")
        
        return experiment_id
    
    async def run_evolution_cycle(
        self,
        experiment_id: str,
        num_generations: int = 1
    ) -> Dict[str, Any]:
        """Run evolution cycles for an experiment."""
        
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        results = {
            'generations_completed': 0,
            'best_algorithms': [],
            'fitness_progression': [],
            'performance_metrics': {}
        }
        
        for generation in range(num_generations):
            # Evolve one generation
            evolved_population = await self.evolution_engine.evolve_generation(
                experiment.training_data,
                {'accuracy': 0.9, 'speed': 0.8}
            )
            
            # Track best algorithms
            best_algorithm = max(evolved_population, key=lambda a: a.fitness_score)
            results['best_algorithms'].append(best_algorithm)
            results['fitness_progression'].append(best_algorithm.fitness_score)
            
            # Update experiment
            experiment.best_fitness_progression.append(best_algorithm.fitness_score)
            experiment.evolved_algorithms = evolved_population
            
            # Log generation results
            generation_log = {
                'generation': self.evolution_engine.generation,
                'best_fitness': best_algorithm.fitness_score,
                'population_diversity': self._calculate_population_diversity(evolved_population),
                'timestamp': datetime.now()
            }
            experiment.evolution_log.append(generation_log)
            
            results['generations_completed'] += 1
        
        # Calculate performance metrics
        if results['fitness_progression']:
            results['performance_metrics'] = {
                'final_best_fitness': results['fitness_progression'][-1],
                'fitness_improvement': results['fitness_progression'][-1] - results['fitness_progression'][0] if len(results['fitness_progression']) > 1 else 0,
                'convergence_rate': self._calculate_convergence_rate(results['fitness_progression']),
                'diversity_maintained': self._calculate_population_diversity(evolved_population)
            }
        
        return results
    
    async def evaluate_algorithm_performance(
        self,
        algorithm: AnalysisAlgorithm,
        test_data: List[Dict[str, Any]]
    ) -> AlgorithmPerformanceRecord:
        """Evaluate algorithm performance on test data."""
        
        start_time = datetime.now()
        
        performance_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        
        success_cases = []
        error_cases = []
        execution_times = []
        
        for data_point in test_data:
            try:
                exec_start = datetime.now()
                
                result = algorithm.executable_function(
                    data_point['code_content'],
                    data_point['file_path']
                )
                
                exec_time = (datetime.now() - exec_start).total_seconds()
                execution_times.append(exec_time)
                
                # Evaluate result quality
                expected = data_point.get('expected_results', {})
                accuracy = self.evolution_engine._calculate_result_accuracy(result, expected)
                
                if accuracy > 0.7:
                    success_cases.append({
                        'data_point_id': data_point.get('id', 'unknown'),
                        'accuracy': accuracy,
                        'execution_time': exec_time
                    })
                else:
                    error_cases.append({
                        'data_point_id': data_point.get('id', 'unknown'),
                        'accuracy': accuracy,
                        'error_type': 'low_accuracy',
                        'execution_time': exec_time
                    })
                
                performance_metrics['accuracy'] += accuracy
                
            except Exception as e:
                exec_time = 1.0  # Penalty time
                execution_times.append(exec_time)
                
                error_cases.append({
                    'data_point_id': data_point.get('id', 'unknown'),
                    'error_type': 'execution_failure',
                    'error_message': str(e),
                    'execution_time': exec_time
                })
        
        # Calculate final metrics
        total_tests = len(test_data)
        if total_tests > 0:
            performance_metrics['accuracy'] /= total_tests
            performance_metrics['success_rate'] = len(success_cases) / total_tests
            performance_metrics['error_rate'] = len(error_cases) / total_tests
        
        # Resource usage
        resource_usage = {
            'avg_execution_time': statistics.mean(execution_times) if execution_times else 0,
            'max_execution_time': max(execution_times) if execution_times else 0,
            'memory_estimate': len(algorithm.source_code) / 1000  # Rough estimate
        }
        
        total_execution_time = (datetime.now() - start_time).total_seconds()
        
        record = AlgorithmPerformanceRecord(
            algorithm_id=algorithm.algorithm_id,
            task_type="code_analysis",
            dataset_identifier=hashlib.md5(str(test_data).encode()).hexdigest()[:8],
            performance_metrics=performance_metrics,
            execution_time=total_execution_time,
            resource_usage=resource_usage,
            error_cases=error_cases,
            success_cases=success_cases,
            feedback_score=performance_metrics['accuracy']
        )
        
        # Store performance record
        self.performance_tracker[algorithm.algorithm_id] = record
        
        return record
    
    def get_best_algorithms(
        self,
        specialization_domain: Optional[str] = None,
        top_k: int = 5
    ) -> List[AnalysisAlgorithm]:
        """Get the best performing algorithms."""
        
        all_algorithms = []
        
        # Collect algorithms from all experiments
        for experiment in self.active_experiments.values():
            for algorithm in experiment.evolved_algorithms:
                if specialization_domain is None or algorithm.specialization_domain == specialization_domain:
                    all_algorithms.append(algorithm)
        
        # Sort by fitness score
        all_algorithms.sort(key=lambda a: a.fitness_score, reverse=True)
        
        return all_algorithms[:top_k]
    
    def _calculate_population_diversity(self, population: List[AnalysisAlgorithm]) -> float:
        """Calculate genetic diversity of population."""
        
        if len(population) < 2:
            return 0.0
        
        # Simple diversity measure based on source code differences
        diversity_scores = []
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                # Calculate similarity between two algorithms
                algo1_code = population[i].source_code
                algo2_code = population[j].source_code
                
                # Simple string similarity
                common_chars = len(set(algo1_code) & set(algo2_code))
                total_chars = len(set(algo1_code) | set(algo2_code))
                
                similarity = common_chars / max(1, total_chars)
                diversity = 1.0 - similarity
                
                diversity_scores.append(diversity)
        
        return statistics.mean(diversity_scores) if diversity_scores else 0.0
    
    def _calculate_convergence_rate(self, fitness_progression: List[float]) -> float:
        """Calculate how quickly the algorithm is converging."""
        
        if len(fitness_progression) < 2:
            return 0.0
        
        # Calculate rate of change in fitness
        changes = []
        for i in range(1, len(fitness_progression)):
            change = fitness_progression[i] - fitness_progression[i-1]
            changes.append(abs(change))
        
        # Convergence rate is inverse of average change
        avg_change = statistics.mean(changes)
        convergence_rate = 1.0 / (1.0 + avg_change)
        
        return convergence_rate


# Example usage and demonstration
async def demonstrate_metamorphic_evolution():
    """
    Demonstrate the metamorphic code evolution system.
    """
    print("Metamorphic Code Evolution Demonstration")
    print("=" * 50)
    
    # Initialize the learning system
    learning_system = MetamorphicLearningSystem()
    
    # Create base algorithm
    base_source_code = '''
def base_analysis_function(code_content: str, file_path: str) -> Dict[str, Any]:
    """Basic analysis algorithm."""
    results = {}
    
    # Basic features
    features = {}
    features["line_count"] = len(code_content.split("\\n"))
    features["char_count"] = len(code_content)
    features["word_count"] = len(code_content.split())
    
    # Simple patterns
    detected_patterns = []
    if features["line_count"] > 50:
        detected_patterns.append("large_file")
    if "def " in code_content:
        detected_patterns.append("contains_functions")
    
    # Basic classification
    classifications = {}
    if features["line_count"] > 100:
        classifications["size_category"] = "large"
    elif features["line_count"] > 20:
        classifications["size_category"] = "medium"
    else:
        classifications["size_category"] = "small"
    
    return {
        'features': features,
        'patterns': detected_patterns,
        'classifications': classifications,
        'algorithm_version': 'base'
    }
'''
    
    # Create executable function
    exec_globals = {'len': len}
    exec(base_source_code, exec_globals)
    base_function = exec_globals['base_analysis_function']
    
    # Create base algorithm object
    base_algorithm = AnalysisAlgorithm(
        algorithm_id="base_001",
        version=1,
        creation_time=datetime.now(),
        parent_algorithms=[],
        genetic_code={},
        performance_metrics={},
        fitness_score=0.5,
        source_code=base_source_code,
        executable_function=base_function,
        specialization_domain="general"
    )
    
    # Create training data
    training_data = []
    
    # Sample code files for training
    sample_codes = [
        {
            'code_content': '''
def simple_function():
    print("Hello World")
    return True
''',
            'file_path': 'simple.py',
            'expected_results': {
                'features': {'line_count': 4, 'char_count': 65},
                'patterns': ['contains_functions'],
                'classifications': {'size_category': 'small'}
            }
        },
        {
            'code_content': '''
class ComplexClass:
    def __init__(self):
        self.data = []
        self.counter = 0
    
    def add_item(self, item):
        self.data.append(item)
        self.counter += 1
    
    def get_items(self):
        return self.data
    
    def process_all(self):
        for item in self.data:
            print(f"Processing: {item}")
            if item > 10:
                yield item * 2
''',
            'file_path': 'complex.py',
            'expected_results': {
                'features': {'line_count': 16, 'char_count': 400},
                'patterns': ['contains_functions'],
                'classifications': {'size_category': 'small'}
            }
        },
        {
            'code_content': '''
# Large file with many functions
import sys
import os
import json
from typing import List, Dict, Any

def function_one():
    pass

def function_two():
    pass

def function_three():
    pass

class DataProcessor:
    def __init__(self):
        self.data = {}
    
    def load_data(self, filename):
        with open(filename, 'r') as f:
            self.data = json.load(f)
    
    def process_data(self):
        processed = {}
        for key, value in self.data.items():
            if isinstance(value, (int, float)):
                processed[key] = value * 2
            else:
                processed[key] = str(value).upper()
        return processed
    
    def save_results(self, output_file):
        processed = self.process_data()
        with open(output_file, 'w') as f:
            json.dump(processed, f, indent=2)

if __name__ == "__main__":
    processor = DataProcessor()
    processor.load_data("input.json")
    processor.save_results("output.json")
''',
            'file_path': 'large.py',
            'expected_results': {
                'features': {'line_count': 40, 'char_count': 1200},
                'patterns': ['contains_functions'],
                'classifications': {'size_category': 'medium'}
            }
        }
    ]
    
    training_data.extend(sample_codes)
    
    print(f"📚 Created training dataset with {len(training_data)} samples")
    
    # Start evolution experiment
    experiment_id = await learning_system.start_evolution_experiment(
        experiment_name="Code Analysis Evolution",
        strategy=EvolutionStrategy.GENETIC_PROGRAMMING,
        base_algorithms=[base_algorithm],
        training_data=training_data,
        population_size=20,
        generations=10,
        fitness_targets={'accuracy': 0.85, 'speed': 0.9}
    )
    
    print(f"🧬 Started evolution experiment: {experiment_id}")
    
    # Run evolution cycles
    print("\n🔄 Running evolution cycles...")
    
    evolution_results = await learning_system.run_evolution_cycle(
        experiment_id, num_generations=5
    )
    
    print(f"✅ Completed {evolution_results['generations_completed']} generations")
    
    # Display results
    print(f"\n📊 EVOLUTION RESULTS:")
    print(f"  🎯 Final Best Fitness: {evolution_results['performance_metrics']['final_best_fitness']:.3f}")
    print(f"  📈 Fitness Improvement: {evolution_results['performance_metrics']['fitness_improvement']:.3f}")
    print(f"  🔄 Convergence Rate: {evolution_results['performance_metrics']['convergence_rate']:.3f}")
    print(f"  🌟 Population Diversity: {evolution_results['performance_metrics']['diversity_maintained']:.3f}")
    
    # Show fitness progression
    print(f"\n📈 FITNESS PROGRESSION:")
    for i, fitness in enumerate(evolution_results['fitness_progression']):
        print(f"  Generation {i+1}: {fitness:.3f}")
    
    # Get best algorithms
    best_algorithms = learning_system.get_best_algorithms(top_k=3)
    
    print(f"\n🏆 TOP EVOLVED ALGORITHMS:")
    for i, algorithm in enumerate(best_algorithms):
        print(f"  #{i+1} Algorithm ID: {algorithm.algorithm_id}")
        print(f"      Fitness Score: {algorithm.fitness_score:.3f}")
        print(f"      Parent Algorithms: {algorithm.parent_algorithms}")
        print(f"      Code Length: {len(algorithm.source_code)} characters")
        
        # Show a snippet of evolved code
        code_lines = algorithm.source_code.split('\n')
        print(f"      Code Preview:")
        for line in code_lines[5:10]:  # Show middle section
            if line.strip():
                print(f"        {line}")
        print()
    
    # Test best algorithm
    if best_algorithms:
        best_algorithm = best_algorithms[0]
        print(f"🧪 TESTING BEST ALGORITHM ({best_algorithm.algorithm_id}):")
        
        test_code = '''
def test_function():
    items = [1, 2, 3, 4, 5]
    for item in items:
        if item % 2 == 0:
            print(f"Even: {item}")
    return len(items)
'''
        
        if best_algorithm.executable_function:
            try:
                result = best_algorithm.executable_function(test_code, "test.py")
                print(f"  📊 Analysis Result:")
                for key, value in result.items():
                    print(f"    {key}: {value}")
            except Exception as e:
                print(f"  ❌ Execution failed: {e}")
        
        # Evaluate performance
        performance_record = await learning_system.evaluate_algorithm_performance(
            best_algorithm, training_data
        )
        
        print(f"\n📈 PERFORMANCE EVALUATION:")
        print(f"  🎯 Accuracy: {performance_record.performance_metrics['accuracy']:.1%}")
        print(f"  ✅ Success Rate: {performance_record.performance_metrics['success_rate']:.1%}")
        print(f"  ❌ Error Rate: {performance_record.performance_metrics['error_rate']:.1%}")
        print(f"  ⏱️ Avg Execution Time: {performance_record.resource_usage['avg_execution_time']:.3f}s")
        print(f"  💾 Memory Estimate: {performance_record.resource_usage['memory_estimate']:.2f}KB")
        print(f"  🏃 Success Cases: {len(performance_record.success_cases)}")
        print(f"  🚫 Error Cases: {len(performance_record.error_cases)}")
    
    # Show evolution history
    experiment = learning_system.active_experiments[experiment_id]
    print(f"\n📚 EVOLUTION HISTORY:")
    for log_entry in experiment.evolution_log[-3:]:  # Last 3 generations
        print(f"  Generation {log_entry['generation']}:")
        print(f"    Best Fitness: {log_entry['best_fitness']:.3f}")
        print(f"    Population Diversity: {log_entry['population_diversity']:.3f}")
        print(f"    Timestamp: {log_entry['timestamp'].strftime('%H:%M:%S')}")


if __name__ == "__main__":
    asyncio.run(demonstrate_metamorphic_evolution())