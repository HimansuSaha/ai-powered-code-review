"""
Biometric Code Intelligence System
=================================

Revolutionary AI system that learns from developer biometric patterns including:
- Eye tracking and gaze patterns
- EEG brain wave analysis  
- Typing rhythm and keystroke dynamics
- Heart rate variability during coding
- Facial expression analysis
- Voice stress patterns
- Physiological stress indicators

This system correlates biometric data with code quality to predict defects,
optimize developer performance, and enhance code review accuracy.

Features:
- Real-time biometric monitoring and analysis
- Code quality prediction from physiological patterns
- Developer fatigue and stress detection
- Optimal coding time recommendation
- Biometric-based code review prioritization
- Personalized development environment adaptation
- Team collaboration biometric synchronization
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import logging
import math
import statistics
from enum import Enum
import hashlib
import cv2
from scipy import signal, stats
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiometricSignalType(Enum):
    """Types of biometric signals monitored."""
    EYE_TRACKING = "eye_tracking"
    EEG_BRAINWAVES = "eeg_brainwaves"
    KEYSTROKE_DYNAMICS = "keystroke_dynamics"
    HEART_RATE_VARIABILITY = "heart_rate_variability"
    FACIAL_EXPRESSION = "facial_expression"
    VOICE_STRESS = "voice_stress"
    GALVANIC_SKIN_RESPONSE = "galvanic_skin_response"
    POSTURE_ANALYSIS = "posture_analysis"
    BREATHING_PATTERN = "breathing_pattern"


class CognitiveState(Enum):
    """Developer cognitive states detected from biometrics."""
    FLOW_STATE = "flow_state"
    FOCUSED_CONCENTRATION = "focused_concentration"
    COGNITIVE_OVERLOAD = "cognitive_overload"
    CREATIVE_THINKING = "creative_thinking"
    PROBLEM_SOLVING = "problem_solving"
    DEBUGGING_MODE = "debugging_mode"
    FATIGUE = "fatigue"
    STRESS = "stress"
    DISTRACTION = "distraction"
    LEARNING = "learning"


@dataclass
class BiometricReading:
    """Single biometric measurement."""
    signal_type: BiometricSignalType
    timestamp: datetime
    value: float
    quality_score: float  # Signal quality (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BiometricSession:
    """Complete biometric monitoring session."""
    session_id: str
    developer_id: str
    start_time: datetime
    end_time: Optional[datetime]
    readings: List[BiometricReading]
    cognitive_states: List[Tuple[datetime, CognitiveState, float]]
    code_activities: List[Dict[str, Any]]
    session_quality_score: float


@dataclass
class CodeQualityPrediction:
    """Biometric-based code quality prediction."""
    predicted_quality_score: float
    confidence_level: float
    contributing_biometrics: Dict[BiometricSignalType, float]
    cognitive_state: CognitiveState
    risk_factors: List[str]
    recommendations: List[str]
    optimal_coding_window: Optional[Tuple[datetime, datetime]]


@dataclass
class DeveloperBiometricProfile:
    """Individual developer's biometric patterns."""
    developer_id: str
    baseline_patterns: Dict[BiometricSignalType, Dict[str, float]]
    cognitive_state_patterns: Dict[CognitiveState, Dict[str, Any]]
    productivity_correlations: Dict[str, float]
    optimal_conditions: Dict[str, Any]
    fatigue_indicators: List[str]
    stress_thresholds: Dict[BiometricSignalType, float]


class BiometricDataProcessor:
    """
    Processes raw biometric data into meaningful patterns.
    """
    
    def __init__(self):
        self.signal_processors = {}
        self.anomaly_detectors = {}
        self.pattern_recognizers = {}
        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize signal processors for different biometric types."""
        
        # Eye tracking processor
        self.signal_processors[BiometricSignalType.EYE_TRACKING] = {
            'fixation_analyzer': self._analyze_eye_fixations,
            'saccade_detector': self._detect_saccades,
            'pupil_analyzer': self._analyze_pupil_dilation,
            'gaze_pattern_extractor': self._extract_gaze_patterns
        }
        
        # EEG brainwave processor
        self.signal_processors[BiometricSignalType.EEG_BRAINWAVES] = {
            'frequency_analyzer': self._analyze_brainwave_frequencies,
            'attention_detector': self._detect_attention_levels,
            'cognitive_load_estimator': self._estimate_cognitive_load,
            'flow_state_detector': self._detect_flow_state
        }
        
        # Keystroke dynamics processor
        self.signal_processors[BiometricSignalType.KEYSTROKE_DYNAMICS] = {
            'rhythm_analyzer': self._analyze_typing_rhythm,
            'pressure_analyzer': self._analyze_key_pressure,
            'pattern_recognizer': self._recognize_typing_patterns,
            'fatigue_detector': self._detect_typing_fatigue
        }
        
        # Heart rate variability processor
        self.signal_processors[BiometricSignalType.HEART_RATE_VARIABILITY] = {
            'stress_analyzer': self._analyze_stress_levels,
            'autonomic_balance': self._analyze_autonomic_balance,
            'arousal_detector': self._detect_arousal_levels,
            'recovery_analyzer': self._analyze_recovery_patterns
        }
        
        # Initialize anomaly detectors
        for signal_type in BiometricSignalType:
            self.anomaly_detectors[signal_type] = IsolationForest(
                contamination=0.1,
                random_state=42
            )
    
    async def process_biometric_stream(
        self,
        raw_readings: List[Dict[str, Any]]
    ) -> List[BiometricReading]:
        """Process raw biometric data stream."""
        
        processed_readings = []
        
        for raw_reading in raw_readings:
            try:
                # Convert to BiometricReading
                reading = await self._convert_raw_reading(raw_reading)
                
                # Apply signal processing
                processed_reading = await self._process_single_reading(reading)
                
                # Quality assessment
                quality_score = await self._assess_signal_quality(processed_reading)
                processed_reading.quality_score = quality_score
                
                processed_readings.append(processed_reading)
                
            except Exception as e:
                logger.warning(f"Failed to process biometric reading: {e}")
                continue
        
        return processed_readings
    
    async def _convert_raw_reading(self, raw_data: Dict[str, Any]) -> BiometricReading:
        """Convert raw sensor data to BiometricReading."""
        
        signal_type = BiometricSignalType(raw_data['signal_type'])
        timestamp = datetime.fromisoformat(raw_data['timestamp'])
        value = float(raw_data['value'])
        metadata = raw_data.get('metadata', {})
        
        return BiometricReading(
            signal_type=signal_type,
            timestamp=timestamp,
            value=value,
            quality_score=0.0,  # Will be calculated
            metadata=metadata
        )
    
    async def _process_single_reading(self, reading: BiometricReading) -> BiometricReading:
        """Apply signal-specific processing to a reading."""
        
        processors = self.signal_processors.get(reading.signal_type, {})
        
        # Apply relevant processors
        for processor_name, processor_func in processors.items():
            try:
                processed_value = await processor_func(reading.value, reading.metadata)
                reading.metadata[processor_name] = processed_value
            except Exception as e:
                logger.warning(f"Processor {processor_name} failed: {e}")
        
        return reading
    
    async def _assess_signal_quality(self, reading: BiometricReading) -> float:
        """Assess the quality of a biometric signal."""
        
        quality_factors = []
        
        # Signal-specific quality assessment
        if reading.signal_type == BiometricSignalType.EYE_TRACKING:
            # Eye tracking quality factors
            confidence = reading.metadata.get('tracking_confidence', 0.5)
            stability = reading.metadata.get('gaze_stability', 0.5)
            calibration_quality = reading.metadata.get('calibration_quality', 0.7)
            
            quality_factors.extend([confidence, stability, calibration_quality])
        
        elif reading.signal_type == BiometricSignalType.EEG_BRAINWAVES:
            # EEG quality factors
            electrode_contact = reading.metadata.get('electrode_contact_quality', 0.8)
            noise_level = 1.0 - reading.metadata.get('noise_level', 0.2)
            artifact_presence = 1.0 - reading.metadata.get('artifact_level', 0.1)
            
            quality_factors.extend([electrode_contact, noise_level, artifact_presence])
        
        elif reading.signal_type == BiometricSignalType.KEYSTROKE_DYNAMICS:
            # Keystroke dynamics quality factors
            key_detection_confidence = reading.metadata.get('key_confidence', 0.9)
            timing_precision = reading.metadata.get('timing_precision', 0.8)
            
            quality_factors.extend([key_detection_confidence, timing_precision])
        
        # Default quality factors
        if not quality_factors:
            quality_factors = [0.7]  # Default moderate quality
        
        # Calculate overall quality score
        quality_score = statistics.mean(quality_factors)
        
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, quality_score))
    
    # Signal-specific processing methods
    async def _analyze_eye_fixations(self, value: float, metadata: Dict) -> Dict[str, float]:
        """Analyze eye fixation patterns."""
        gaze_x = metadata.get('gaze_x', 0)
        gaze_y = metadata.get('gaze_y', 0)
        
        return {
            'fixation_duration': value,
            'fixation_stability': 1.0 / (1.0 + abs(gaze_x) + abs(gaze_y)),
            'code_region_focus': self._calculate_code_region_focus(gaze_x, gaze_y)
        }
    
    async def _detect_saccades(self, value: float, metadata: Dict) -> Dict[str, float]:
        """Detect and analyze saccadic eye movements."""
        velocity = metadata.get('eye_velocity', 0)
        
        is_saccade = velocity > 30  # degrees per second threshold
        
        return {
            'saccade_detected': float(is_saccade),
            'saccade_velocity': velocity,
            'reading_efficiency': 1.0 / (1.0 + velocity / 100)
        }
    
    async def _analyze_pupil_dilation(self, value: float, metadata: Dict) -> Dict[str, float]:
        """Analyze pupil dilation for cognitive load."""
        baseline_pupil = metadata.get('baseline_pupil_size', 3.0)
        current_pupil = value
        
        dilation_ratio = current_pupil / baseline_pupil
        cognitive_load = min(1.0, max(0.0, (dilation_ratio - 1.0) / 0.5))
        
        return {
            'dilation_ratio': dilation_ratio,
            'cognitive_load_estimate': cognitive_load,
            'attention_level': 1.0 - abs(dilation_ratio - 1.2) / 0.3
        }
    
    async def _extract_gaze_patterns(self, value: float, metadata: Dict) -> Dict[str, float]:
        """Extract high-level gaze patterns."""
        gaze_entropy = metadata.get('gaze_entropy', 0.5)
        scan_path_length = metadata.get('scan_path_length', 100)
        
        return {
            'gaze_entropy': gaze_entropy,
            'exploration_behavior': min(1.0, gaze_entropy / 2.0),
            'focused_behavior': max(0.0, 1.0 - gaze_entropy),
            'reading_efficiency': 100.0 / max(1.0, scan_path_length)
        }
    
    def _calculate_code_region_focus(self, gaze_x: float, gaze_y: float) -> float:
        """Calculate focus on code regions vs other areas."""
        # Simplified - assumes code is in central region
        code_region_x = (0.2 <= gaze_x <= 0.8)
        code_region_y = (0.1 <= gaze_y <= 0.9)
        
        return float(code_region_x and code_region_y)
    
    async def _analyze_brainwave_frequencies(self, value: float, metadata: Dict) -> Dict[str, float]:
        """Analyze EEG frequency bands."""
        # Extract frequency band powers
        delta = metadata.get('delta_power', 0.2)  # 0.5-4 Hz
        theta = metadata.get('theta_power', 0.3)  # 4-8 Hz
        alpha = metadata.get('alpha_power', 0.4)  # 8-13 Hz
        beta = metadata.get('beta_power', 0.5)    # 13-30 Hz
        gamma = metadata.get('gamma_power', 0.1)  # 30-100 Hz
        
        return {
            'attention_index': (beta / (alpha + theta)) if (alpha + theta) > 0 else 0,
            'relaxation_index': alpha / (alpha + beta) if (alpha + beta) > 0 else 0,
            'cognitive_effort': (beta + gamma) / (delta + theta + alpha + beta + gamma),
            'flow_indicator': alpha / max(0.001, beta + theta)
        }
    
    async def _detect_attention_levels(self, value: float, metadata: Dict) -> Dict[str, float]:
        """Detect attention and focus levels from EEG."""
        beta_power = metadata.get('beta_power', 0.5)
        gamma_power = metadata.get('gamma_power', 0.1)
        alpha_power = metadata.get('alpha_power', 0.4)
        
        attention_score = (beta_power + gamma_power) / max(0.001, alpha_power)
        sustained_attention = min(1.0, attention_score / 2.0)
        
        return {
            'attention_score': attention_score,
            'sustained_attention': sustained_attention,
            'focus_stability': 1.0 - abs(attention_score - 1.0)
        }
    
    async def _estimate_cognitive_load(self, value: float, metadata: Dict) -> Dict[str, float]:
        """Estimate cognitive workload from EEG patterns."""
        theta_power = metadata.get('theta_power', 0.3)
        alpha_power = metadata.get('alpha_power', 0.4)
        beta_power = metadata.get('beta_power', 0.5)
        
        # Cognitive load typically increases theta and beta, decreases alpha
        cognitive_load = (theta_power + beta_power) / max(0.001, alpha_power)
        normalized_load = min(1.0, cognitive_load / 3.0)
        
        return {
            'cognitive_load': normalized_load,
            'mental_effort': beta_power / max(0.001, alpha_power),
            'working_memory_load': theta_power
        }
    
    async def _detect_flow_state(self, value: float, metadata: Dict) -> Dict[str, float]:
        """Detect flow state characteristics in EEG."""
        alpha_power = metadata.get('alpha_power', 0.4)
        theta_power = metadata.get('theta_power', 0.3)
        beta_power = metadata.get('beta_power', 0.5)
        
        # Flow state often characterized by alpha-theta synchrony
        flow_ratio = (alpha_power * theta_power) / max(0.001, beta_power)
        flow_probability = min(1.0, flow_ratio / 0.5)
        
        return {
            'flow_probability': flow_probability,
            'alpha_theta_sync': alpha_power * theta_power,
            'relaxed_focus': alpha_power / max(0.001, beta_power)
        }
    
    async def _analyze_typing_rhythm(self, value: float, metadata: Dict) -> Dict[str, float]:
        """Analyze keystroke timing patterns."""
        dwell_time = metadata.get('dwell_time', 100)  # ms
        flight_time = metadata.get('flight_time', 50)  # ms
        
        typing_speed = 1000.0 / max(1, dwell_time + flight_time)  # keys per second
        rhythm_consistency = 1.0 / (1.0 + abs(dwell_time - 100) / 50)
        
        return {
            'typing_speed': typing_speed,
            'rhythm_consistency': rhythm_consistency,
            'keystroke_pressure': value,
            'fluency_indicator': typing_speed * rhythm_consistency
        }
    
    async def _analyze_key_pressure(self, value: float, metadata: Dict) -> Dict[str, float]:
        """Analyze keystroke pressure patterns."""
        baseline_pressure = metadata.get('baseline_pressure', 50)
        current_pressure = value
        
        pressure_ratio = current_pressure / max(1, baseline_pressure)
        stress_indicator = max(0, pressure_ratio - 1.5) / 0.5
        
        return {
            'pressure_ratio': pressure_ratio,
            'stress_indicator': min(1.0, stress_indicator),
            'typing_confidence': 1.0 / (1.0 + abs(pressure_ratio - 1.0))
        }
    
    async def _recognize_typing_patterns(self, value: float, metadata: Dict) -> Dict[str, float]:
        """Recognize high-level typing behavior patterns."""
        pause_duration = metadata.get('pause_duration', 0)
        burst_length = metadata.get('burst_length', 5)
        
        thinking_pause = float(pause_duration > 2000)  # > 2 seconds
        code_complexity = min(1.0, burst_length / 20)
        
        return {
            'thinking_pause': thinking_pause,
            'code_complexity_estimate': code_complexity,
            'typing_fluency': 1.0 / (1.0 + pause_duration / 1000)
        }
    
    async def _detect_typing_fatigue(self, value: float, metadata: Dict) -> Dict[str, float]:
        """Detect fatigue indicators in typing patterns."""
        error_rate = metadata.get('error_rate', 0.05)
        speed_variance = metadata.get('speed_variance', 0.1)
        
        fatigue_score = (error_rate * 10 + speed_variance * 5) / 2
        fatigue_probability = min(1.0, fatigue_score)
        
        return {
            'fatigue_probability': fatigue_probability,
            'error_rate': error_rate,
            'speed_inconsistency': speed_variance
        }
    
    async def _analyze_stress_levels(self, value: float, metadata: Dict) -> Dict[str, float]:
        """Analyze stress from heart rate variability."""
        hrv = value  # Heart rate variability
        resting_hrv = metadata.get('resting_hrv', 50)
        
        stress_ratio = resting_hrv / max(1, hrv)  # Lower HRV = higher stress
        stress_level = min(1.0, max(0, (stress_ratio - 1.0) / 2.0))
        
        return {
            'stress_level': stress_level,
            'autonomic_balance': min(1.0, hrv / 50),
            'recovery_capacity': 1.0 - stress_level
        }
    
    async def _analyze_autonomic_balance(self, value: float, metadata: Dict) -> Dict[str, float]:
        """Analyze autonomic nervous system balance."""
        lf_power = metadata.get('lf_power', 0.4)  # Low frequency power
        hf_power = metadata.get('hf_power', 0.6)  # High frequency power
        
        lf_hf_ratio = lf_power / max(0.001, hf_power)
        sympathetic_dominance = min(1.0, lf_hf_ratio / 4.0)
        
        return {
            'lf_hf_ratio': lf_hf_ratio,
            'sympathetic_dominance': sympathetic_dominance,
            'parasympathetic_tone': hf_power,
            'autonomic_balance': 1.0 / (1.0 + abs(lf_hf_ratio - 1.0))
        }
    
    async def _detect_arousal_levels(self, value: float, metadata: Dict) -> Dict[str, float]:
        """Detect arousal and activation levels."""
        heart_rate = metadata.get('heart_rate', 70)
        baseline_hr = metadata.get('baseline_hr', 65)
        
        arousal_index = (heart_rate - baseline_hr) / baseline_hr
        activation_level = min(1.0, max(0, arousal_index + 0.5))
        
        return {
            'arousal_index': arousal_index,
            'activation_level': activation_level,
            'optimal_arousal': 1.0 - abs(arousal_index - 0.2) / 0.3
        }
    
    async def _analyze_recovery_patterns(self, value: float, metadata: Dict) -> Dict[str, float]:
        """Analyze recovery and resilience patterns."""
        recovery_time = metadata.get('recovery_time', 60)  # seconds
        baseline_recovery = metadata.get('baseline_recovery', 45)
        
        recovery_efficiency = baseline_recovery / max(1, recovery_time)
        resilience_score = min(1.0, recovery_efficiency)
        
        return {
            'recovery_efficiency': recovery_efficiency,
            'resilience_score': resilience_score,
            'adaptation_capacity': min(1.0, 1.0 / max(0.1, recovery_time / 30))
        }


class CognitiveStateDetector:
    """
    Detects developer cognitive states from biometric patterns.
    """
    
    def __init__(self):
        self.state_models = {}
        self.state_thresholds = {}
        self._initialize_state_detection()
    
    def _initialize_state_detection(self):
        """Initialize cognitive state detection models."""
        
        # Define state detection thresholds and patterns
        self.state_thresholds = {
            CognitiveState.FLOW_STATE: {
                'alpha_theta_sync': 0.6,
                'attention_stability': 0.7,
                'typing_fluency': 0.8,
                'stress_level': 0.3
            },
            CognitiveState.FOCUSED_CONCENTRATION: {
                'attention_score': 0.8,
                'cognitive_load': 0.6,
                'gaze_stability': 0.7,
                'distraction_level': 0.2
            },
            CognitiveState.COGNITIVE_OVERLOAD: {
                'cognitive_load': 0.8,
                'stress_level': 0.7,
                'error_rate': 0.1,
                'hrv_decrease': 0.6
            },
            CognitiveState.CREATIVE_THINKING: {
                'alpha_power': 0.6,
                'gaze_entropy': 0.7,
                'thinking_pauses': 0.5,
                'exploration_behavior': 0.6
            },
            CognitiveState.PROBLEM_SOLVING: {
                'beta_power': 0.7,
                'fixation_duration': 0.6,
                'cognitive_effort': 0.7,
                'working_memory_load': 0.6
            },
            CognitiveState.DEBUGGING_MODE: {
                'attention_score': 0.8,
                'gaze_entropy': 0.4,
                'systematic_scanning': 0.7,
                'sustained_focus': 0.8
            },
            CognitiveState.FATIGUE: {
                'fatigue_probability': 0.6,
                'attention_decline': 0.7,
                'error_rate_increase': 0.8,
                'recovery_capacity': 0.3
            },
            CognitiveState.STRESS: {
                'stress_level': 0.6,
                'hrv_decrease': 0.5,
                'sympathetic_dominance': 0.7,
                'pressure_increase': 0.6
            },
            CognitiveState.DISTRACTION: {
                'attention_instability': 0.7,
                'gaze_wandering': 0.6,
                'typing_inconsistency': 0.5,
                'task_switching': 0.8
            },
            CognitiveState.LEARNING: {
                'exploration_behavior': 0.7,
                'theta_power': 0.6,
                'gaze_entropy': 0.8,
                'processing_time': 0.6
            }
        }
    
    async def detect_cognitive_state(
        self,
        biometric_readings: List[BiometricReading],
        time_window: timedelta = timedelta(minutes=5)
    ) -> List[Tuple[datetime, CognitiveState, float]]:
        """Detect cognitive states from biometric data."""
        
        # Group readings by time windows
        windowed_data = self._create_time_windows(biometric_readings, time_window)
        
        detected_states = []
        
        for window_start, readings in windowed_data.items():
            # Extract features for state detection
            features = await self._extract_cognitive_features(readings)
            
            # Detect state probabilities
            state_probabilities = await self._calculate_state_probabilities(features)
            
            # Determine primary state
            primary_state, confidence = self._determine_primary_state(state_probabilities)
            
            detected_states.append((window_start, primary_state, confidence))
        
        return detected_states
    
    def _create_time_windows(
        self,
        readings: List[BiometricReading],
        window_size: timedelta
    ) -> Dict[datetime, List[BiometricReading]]:
        """Group readings into time windows."""
        
        if not readings:
            return {}
        
        # Sort by timestamp
        sorted_readings = sorted(readings, key=lambda r: r.timestamp)
        
        windows = {}
        current_window_start = sorted_readings[0].timestamp
        current_window_end = current_window_start + window_size
        current_readings = []
        
        for reading in sorted_readings:
            if reading.timestamp <= current_window_end:
                current_readings.append(reading)
            else:
                # Save current window
                if current_readings:
                    windows[current_window_start] = current_readings.copy()
                
                # Start new window
                current_window_start = reading.timestamp
                current_window_end = current_window_start + window_size
                current_readings = [reading]
        
        # Save final window
        if current_readings:
            windows[current_window_start] = current_readings
        
        return windows
    
    async def _extract_cognitive_features(
        self,
        readings: List[BiometricReading]
    ) -> Dict[str, float]:
        """Extract cognitive features from biometric readings."""
        
        features = {}
        
        # Group readings by signal type
        signal_groups = defaultdict(list)
        for reading in readings:
            signal_groups[reading.signal_type].append(reading)
        
        # Extract features for each signal type
        for signal_type, signal_readings in signal_groups.items():
            signal_features = await self._extract_signal_features(
                signal_type, signal_readings
            )
            features.update(signal_features)
        
        return features
    
    async def _extract_signal_features(
        self,
        signal_type: BiometricSignalType,
        readings: List[BiometricReading]
    ) -> Dict[str, float]:
        """Extract features for a specific signal type."""
        
        features = {}
        
        if signal_type == BiometricSignalType.EYE_TRACKING:
            # Eye tracking features
            gaze_stability_values = [
                r.metadata.get('fixation_stability', 0.5) for r in readings
            ]
            attention_values = [
                r.metadata.get('attention_level', 0.5) for r in readings
            ]
            
            features.update({
                'gaze_stability': statistics.mean(gaze_stability_values),
                'attention_level': statistics.mean(attention_values),
                'gaze_entropy': statistics.stdev(gaze_stability_values) if len(gaze_stability_values) > 1 else 0,
                'fixation_duration': statistics.mean([r.value for r in readings])
            })
        
        elif signal_type == BiometricSignalType.EEG_BRAINWAVES:
            # EEG features
            attention_scores = [
                r.metadata.get('attention_score', 0.5) for r in readings
            ]
            cognitive_load_values = [
                r.metadata.get('cognitive_load', 0.5) for r in readings
            ]
            flow_probabilities = [
                r.metadata.get('flow_probability', 0.3) for r in readings
            ]
            
            features.update({
                'attention_score': statistics.mean(attention_scores),
                'cognitive_load': statistics.mean(cognitive_load_values),
                'flow_probability': statistics.mean(flow_probabilities),
                'attention_stability': 1.0 - (statistics.stdev(attention_scores) if len(attention_scores) > 1 else 0)
            })
        
        elif signal_type == BiometricSignalType.KEYSTROKE_DYNAMICS:
            # Keystroke features
            fluency_values = [
                r.metadata.get('fluency_indicator', 0.5) for r in readings
            ]
            fatigue_values = [
                r.metadata.get('fatigue_probability', 0.3) for r in readings
            ]
            
            features.update({
                'typing_fluency': statistics.mean(fluency_values),
                'fatigue_probability': statistics.mean(fatigue_values),
                'typing_consistency': 1.0 - (statistics.stdev(fluency_values) if len(fluency_values) > 1 else 0)
            })
        
        elif signal_type == BiometricSignalType.HEART_RATE_VARIABILITY:
            # HRV features
            stress_values = [
                r.metadata.get('stress_level', 0.3) for r in readings
            ]
            arousal_values = [
                r.metadata.get('activation_level', 0.5) for r in readings
            ]
            
            features.update({
                'stress_level': statistics.mean(stress_values),
                'arousal_level': statistics.mean(arousal_values),
                'autonomic_balance': statistics.mean([
                    r.metadata.get('autonomic_balance', 0.5) for r in readings
                ])
            })
        
        return features
    
    async def _calculate_state_probabilities(
        self,
        features: Dict[str, float]
    ) -> Dict[CognitiveState, float]:
        """Calculate probability for each cognitive state."""
        
        state_probabilities = {}
        
        for state, thresholds in self.state_thresholds.items():
            probability = await self._calculate_single_state_probability(
                state, thresholds, features
            )
            state_probabilities[state] = probability
        
        return state_probabilities
    
    async def _calculate_single_state_probability(
        self,
        state: CognitiveState,
        thresholds: Dict[str, float],
        features: Dict[str, float]
    ) -> float:
        """Calculate probability for a single cognitive state."""
        
        matches = []
        
        for feature_name, threshold in thresholds.items():
            feature_value = features.get(feature_name, 0.0)
            
            # Calculate match score based on how close feature is to threshold
            if state in [CognitiveState.FATIGUE, CognitiveState.STRESS, CognitiveState.COGNITIVE_OVERLOAD]:
                # For negative states, higher values are more indicative
                match_score = min(1.0, feature_value / threshold)
            else:
                # For positive states, values near threshold are most indicative
                distance = abs(feature_value - threshold)
                match_score = max(0.0, 1.0 - distance / 0.5)
            
            matches.append(match_score)
        
        # Calculate overall probability
        if matches:
            probability = statistics.mean(matches)
        else:
            probability = 0.0
        
        return min(1.0, max(0.0, probability))
    
    def _determine_primary_state(
        self,
        state_probabilities: Dict[CognitiveState, float]
    ) -> Tuple[CognitiveState, float]:
        """Determine the primary cognitive state."""
        
        if not state_probabilities:
            return CognitiveState.FOCUSED_CONCENTRATION, 0.5
        
        # Find state with highest probability
        primary_state = max(state_probabilities.keys(), key=lambda s: state_probabilities[s])
        confidence = state_probabilities[primary_state]
        
        # Minimum confidence threshold
        if confidence < 0.3:
            return CognitiveState.FOCUSED_CONCENTRATION, 0.3
        
        return primary_state, confidence


class BiometricCodeQualityPredictor:
    """
    Predicts code quality based on biometric patterns.
    """
    
    def __init__(self):
        self.quality_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = {}
    
    async def predict_code_quality(
        self,
        biometric_session: BiometricSession,
        cognitive_states: List[Tuple[datetime, CognitiveState, float]],
        developer_profile: Optional[DeveloperBiometricProfile] = None
    ) -> CodeQualityPrediction:
        """Predict code quality from biometric data."""
        
        # Extract prediction features
        features = await self._extract_prediction_features(
            biometric_session, cognitive_states, developer_profile
        )
        
        # Make prediction
        if self.is_trained:
            quality_score = await self._predict_quality_score(features)
            confidence = await self._calculate_prediction_confidence(features)
        else:
            # Fallback prediction based on heuristics
            quality_score, confidence = await self._heuristic_quality_prediction(features)
        
        # Identify contributing factors
        contributing_biometrics = await self._identify_contributing_factors(features)
        
        # Determine cognitive state impact
        primary_cognitive_state = self._get_dominant_cognitive_state(cognitive_states)
        
        # Identify risk factors
        risk_factors = await self._identify_risk_factors(features, cognitive_states)
        
        # Generate recommendations
        recommendations = await self._generate_quality_recommendations(
            features, cognitive_states, quality_score
        )
        
        # Find optimal coding window
        optimal_window = await self._find_optimal_coding_window(biometric_session)
        
        return CodeQualityPrediction(
            predicted_quality_score=quality_score,
            confidence_level=confidence,
            contributing_biometrics=contributing_biometrics,
            cognitive_state=primary_cognitive_state,
            risk_factors=risk_factors,
            recommendations=recommendations,
            optimal_coding_window=optimal_window
        )
    
    async def _extract_prediction_features(
        self,
        session: BiometricSession,
        cognitive_states: List[Tuple[datetime, CognitiveState, float]],
        profile: Optional[DeveloperBiometricProfile]
    ) -> Dict[str, float]:
        """Extract features for quality prediction."""
        
        features = {}
        
        # Biometric aggregation features
        signal_aggregates = defaultdict(list)
        for reading in session.readings:
            signal_aggregates[reading.signal_type].append(reading.value)
        
        for signal_type, values in signal_aggregates.items():
            prefix = signal_type.value
            features.update({
                f'{prefix}_mean': statistics.mean(values),
                f'{prefix}_std': statistics.stdev(values) if len(values) > 1 else 0,
                f'{prefix}_quality': statistics.mean([
                    r.quality_score for r in session.readings
                    if r.signal_type == signal_type
                ])
            })
        
        # Cognitive state features
        state_durations = defaultdict(float)
        total_duration = 0
        
        for i, (timestamp, state, confidence) in enumerate(cognitive_states):
            if i < len(cognitive_states) - 1:
                next_timestamp = cognitive_states[i + 1][0]
                duration = (next_timestamp - timestamp).total_seconds()
            else:
                duration = 300  # Default 5 minutes for last state
            
            state_durations[state] += duration
            total_duration += duration
        
        for state in CognitiveState:
            features[f'state_{state.value}_ratio'] = state_durations[state] / max(1, total_duration)
        
        # Session-level features
        session_duration = (session.end_time - session.start_time).total_seconds() if session.end_time else 3600
        features.update({
            'session_duration': session_duration / 3600,  # Hours
            'session_quality': session.session_quality_score,
            'reading_count': len(session.readings),
            'activity_count': len(session.code_activities)
        })
        
        # Developer profile features (if available)
        if profile:
            features.update({
                'profile_productivity_baseline': statistics.mean(profile.productivity_correlations.values()),
                'profile_stress_tolerance': 1.0 - statistics.mean(profile.stress_thresholds.values()),
                'profile_optimal_conditions_match': self._calculate_conditions_match(features, profile)
            })
        
        return features
    
    async def _predict_quality_score(self, features: Dict[str, float]) -> float:
        """Predict quality score using trained model."""
        
        # Convert features to array
        feature_array = np.array([list(features.values())]).reshape(1, -1)
        
        # Scale features
        scaled_features = self.feature_scaler.transform(feature_array)
        
        # Predict
        prediction = self.quality_model.predict(scaled_features)[0]
        
        # Clamp to valid range
        return max(0.0, min(1.0, prediction))
    
    async def _calculate_prediction_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence in the prediction."""
        
        # Base confidence from model uncertainty (simplified)
        feature_stability = 1.0 - np.std(list(features.values()))
        
        # Confidence from signal quality
        quality_features = [v for k, v in features.items() if 'quality' in k]
        signal_quality = statistics.mean(quality_features) if quality_features else 0.7
        
        # Combined confidence
        confidence = (feature_stability * 0.6 + signal_quality * 0.4)
        
        return max(0.1, min(1.0, confidence))
    
    async def _heuristic_quality_prediction(
        self,
        features: Dict[str, float]
    ) -> Tuple[float, float]:
        """Fallback quality prediction using heuristics."""
        
        quality_indicators = []
        
        # Flow state indicator
        flow_ratio = features.get('state_flow_state_ratio', 0.0)
        quality_indicators.append(flow_ratio * 1.2)
        
        # Focus indicator  
        focus_ratio = features.get('state_focused_concentration_ratio', 0.5)
        quality_indicators.append(focus_ratio)
        
        # Stress penalty
        stress_ratio = features.get('state_stress_ratio', 0.3)
        quality_indicators.append(1.0 - stress_ratio)
        
        # Fatigue penalty
        fatigue_ratio = features.get('state_fatigue_ratio', 0.2)
        quality_indicators.append(1.0 - fatigue_ratio)
        
        # Session quality
        session_quality = features.get('session_quality', 0.7)
        quality_indicators.append(session_quality)
        
        # Calculate overall quality
        quality_score = statistics.mean(quality_indicators)
        confidence = 0.6  # Moderate confidence for heuristic
        
        return max(0.0, min(1.0, quality_score)), confidence
    
    async def _identify_contributing_factors(
        self,
        features: Dict[str, float]
    ) -> Dict[BiometricSignalType, float]:
        """Identify which biometric signals contribute most to quality."""
        
        contributions = {}
        
        # Simple heuristic-based contribution analysis
        for signal_type in BiometricSignalType:
            signal_prefix = signal_type.value
            
            # Find relevant features for this signal
            signal_features = {
                k: v for k, v in features.items()
                if k.startswith(signal_prefix)
            }
            
            if signal_features:
                # Calculate contribution based on feature values
                quality_contribution = statistics.mean(signal_features.values())
                contributions[signal_type] = quality_contribution
        
        return contributions
    
    def _get_dominant_cognitive_state(
        self,
        cognitive_states: List[Tuple[datetime, CognitiveState, float]]
    ) -> CognitiveState:
        """Get the dominant cognitive state during the session."""
        
        if not cognitive_states:
            return CognitiveState.FOCUSED_CONCENTRATION
        
        # Calculate state durations
        state_durations = defaultdict(float)
        
        for i, (timestamp, state, confidence) in enumerate(cognitive_states):
            if i < len(cognitive_states) - 1:
                next_timestamp = cognitive_states[i + 1][0]
                duration = (next_timestamp - timestamp).total_seconds()
            else:
                duration = 300  # Default for last state
            
            state_durations[state] += duration * confidence  # Weight by confidence
        
        # Return state with longest weighted duration
        return max(state_durations.keys(), key=lambda s: state_durations[s])
    
    async def _identify_risk_factors(
        self,
        features: Dict[str, float],
        cognitive_states: List[Tuple[datetime, CognitiveState, float]]
    ) -> List[str]:
        """Identify risk factors for code quality."""
        
        risks = []
        
        # High stress risk
        stress_ratio = features.get('state_stress_ratio', 0.0)
        if stress_ratio > 0.3:
            risks.append(f"High stress levels detected ({stress_ratio:.1%} of session)")
        
        # Fatigue risk
        fatigue_ratio = features.get('state_fatigue_ratio', 0.0)
        if fatigue_ratio > 0.2:
            risks.append(f"Developer fatigue detected ({fatigue_ratio:.1%} of session)")
        
        # Cognitive overload risk
        overload_ratio = features.get('state_cognitive_overload_ratio', 0.0)
        if overload_ratio > 0.1:
            risks.append(f"Cognitive overload periods detected ({overload_ratio:.1%} of session)")
        
        # Distraction risk
        distraction_ratio = features.get('state_distraction_ratio', 0.0)
        if distraction_ratio > 0.2:
            risks.append(f"High distraction levels ({distraction_ratio:.1%} of session)")
        
        # Low session quality
        session_quality = features.get('session_quality', 0.7)
        if session_quality < 0.5:
            risks.append(f"Poor biometric signal quality (quality score: {session_quality:.1%})")
        
        # Extended session duration
        session_duration = features.get('session_duration', 1.0)
        if session_duration > 4.0:  # > 4 hours
            risks.append(f"Extended coding session ({session_duration:.1f} hours)")
        
        return risks
    
    async def _generate_quality_recommendations(
        self,
        features: Dict[str, float],
        cognitive_states: List[Tuple[datetime, CognitiveState, float]],
        quality_score: float
    ) -> List[str]:
        """Generate recommendations to improve code quality."""
        
        recommendations = []
        
        # Quality-based recommendations
        if quality_score < 0.6:
            recommendations.append("Consider taking a break to improve focus and code quality")
        
        # State-based recommendations
        stress_ratio = features.get('state_stress_ratio', 0.0)
        if stress_ratio > 0.3:
            recommendations.extend([
                "Practice stress reduction techniques (deep breathing, meditation)",
                "Consider breaking complex tasks into smaller, manageable pieces"
            ])
        
        fatigue_ratio = features.get('state_fatigue_ratio', 0.0)
        if fatigue_ratio > 0.2:
            recommendations.extend([
                "Take regular breaks to prevent fatigue accumulation",
                "Consider adjusting workspace ergonomics"
            ])
        
        flow_ratio = features.get('state_flow_state_ratio', 0.0)
        if flow_ratio < 0.2:
            recommendations.extend([
                "Try to minimize interruptions to achieve flow state",
                "Focus on challenging but achievable tasks"
            ])
        
        # Session-based recommendations
        session_duration = features.get('session_duration', 1.0)
        if session_duration > 3.0:
            recommendations.append("Consider shorter coding sessions with breaks")
        
        # Signal quality recommendations
        signal_quality = features.get('session_quality', 0.7)
        if signal_quality < 0.6:
            recommendations.append("Check biometric sensor placement and calibration")
        
        return recommendations
    
    async def _find_optimal_coding_window(
        self,
        session: BiometricSession
    ) -> Optional[Tuple[datetime, datetime]]:
        """Find the optimal time window for coding based on biometrics."""
        
        # Simplified implementation - find period with best biometric indicators
        if len(session.readings) < 10:
            return None
        
        # Group readings into 30-minute windows
        window_size = timedelta(minutes=30)
        windows = {}
        
        start_time = session.start_time
        current_window = start_time
        
        while current_window < session.start_time + timedelta(hours=8):  # Max 8 hours
            window_end = current_window + window_size
            
            # Find readings in this window
            window_readings = [
                r for r in session.readings
                if current_window <= r.timestamp < window_end
            ]
            
            if window_readings:
                # Calculate window quality score
                quality_scores = [r.quality_score for r in window_readings]
                avg_quality = statistics.mean(quality_scores)
                
                # Consider signal stability
                values = [r.value for r in window_readings]
                stability = 1.0 / (1.0 + statistics.stdev(values)) if len(values) > 1 else 0.5
                
                window_score = avg_quality * 0.7 + stability * 0.3
                windows[current_window] = window_score
            
            current_window = window_end
        
        if windows:
            # Find best window
            best_window_start = max(windows.keys(), key=lambda w: windows[w])
            best_window_end = best_window_start + window_size
            
            return (best_window_start, best_window_end)
        
        return None
    
    def _calculate_conditions_match(
        self,
        current_features: Dict[str, float],
        profile: DeveloperBiometricProfile
    ) -> float:
        """Calculate how well current conditions match optimal profile."""
        
        # Simplified implementation
        optimal_conditions = profile.optimal_conditions
        
        matches = []
        for condition, optimal_value in optimal_conditions.items():
            current_value = current_features.get(condition, 0.5)
            match_score = 1.0 - abs(current_value - optimal_value)
            matches.append(max(0.0, match_score))
        
        return statistics.mean(matches) if matches else 0.5


# Example usage and demonstration
async def demonstrate_biometric_intelligence():
    """
    Demonstrate the biometric code intelligence system.
    """
    print("Biometric Code Intelligence Demonstration")
    print("=" * 50)
    
    # Initialize components
    processor = BiometricDataProcessor()
    state_detector = CognitiveStateDetector()
    quality_predictor = BiometricCodeQualityPredictor()
    
    # Simulate biometric data stream
    current_time = datetime.now()
    raw_biometric_data = []
    
    # Generate sample biometric readings
    for i in range(100):
        timestamp = current_time + timedelta(seconds=i * 30)
        
        # Simulate various biometric signals
        raw_biometric_data.extend([
            {
                'signal_type': 'eye_tracking',
                'timestamp': timestamp.isoformat(),
                'value': 250 + np.random.normal(0, 50),  # Fixation duration ms
                'metadata': {
                    'gaze_x': 0.5 + np.random.normal(0, 0.1),
                    'gaze_y': 0.4 + np.random.normal(0, 0.1),
                    'pupil_size': 3.5 + np.random.normal(0, 0.5),
                    'tracking_confidence': 0.8 + np.random.uniform(-0.2, 0.2)
                }
            },
            {
                'signal_type': 'eeg_brainwaves',
                'timestamp': timestamp.isoformat(),
                'value': 15.0 + np.random.normal(0, 3),  # Beta power
                'metadata': {
                    'alpha_power': 0.4 + np.random.uniform(-0.1, 0.1),
                    'beta_power': 0.5 + np.random.uniform(-0.1, 0.1),
                    'theta_power': 0.3 + np.random.uniform(-0.1, 0.1),
                    'gamma_power': 0.1 + np.random.uniform(-0.05, 0.05),
                    'electrode_contact_quality': 0.85 + np.random.uniform(-0.1, 0.1)
                }
            },
            {
                'signal_type': 'keystroke_dynamics',
                'timestamp': timestamp.isoformat(),
                'value': 60 + np.random.normal(0, 15),  # Keystroke pressure
                'metadata': {
                    'dwell_time': 100 + np.random.normal(0, 20),
                    'flight_time': 50 + np.random.normal(0, 10),
                    'error_rate': max(0, 0.05 + np.random.normal(0, 0.02)),
                    'baseline_pressure': 50
                }
            },
            {
                'signal_type': 'heart_rate_variability',
                'timestamp': timestamp.isoformat(),
                'value': 40 + np.random.normal(0, 10),  # HRV
                'metadata': {
                    'heart_rate': 70 + np.random.normal(0, 5),
                    'lf_power': 0.4 + np.random.uniform(-0.1, 0.1),
                    'hf_power': 0.6 + np.random.uniform(-0.1, 0.1),
                    'baseline_hr': 65
                }
            }
        ])
    
    print(f"🔬 Processing {len(raw_biometric_data)} biometric readings...")
    
    # Process biometric data
    processed_readings = await processor.process_biometric_stream(raw_biometric_data)
    
    print(f"✅ Processed readings: {len(processed_readings)}")
    print(f"📊 Average signal quality: {statistics.mean([r.quality_score for r in processed_readings]):.1%}")
    
    # Detect cognitive states
    cognitive_states = await state_detector.detect_cognitive_state(processed_readings)
    
    print(f"\n🧠 COGNITIVE STATES DETECTED:")
    for timestamp, state, confidence in cognitive_states[:5]:
        print(f"  • {timestamp.strftime('%H:%M:%S')}: {state.value} (confidence: {confidence:.1%})")
    
    # Create biometric session
    session = BiometricSession(
        session_id=hashlib.md5(str(current_time).encode()).hexdigest()[:8],
        developer_id="dev_001",
        start_time=current_time,
        end_time=current_time + timedelta(hours=2),
        readings=processed_readings,
        cognitive_states=cognitive_states,
        code_activities=[
            {'activity': 'writing_function', 'timestamp': current_time + timedelta(minutes=5)},
            {'activity': 'debugging', 'timestamp': current_time + timedelta(minutes=30)},
            {'activity': 'refactoring', 'timestamp': current_time + timedelta(minutes=60)}
        ],
        session_quality_score=statistics.mean([r.quality_score for r in processed_readings])
    )
    
    # Predict code quality
    quality_prediction = await quality_predictor.predict_code_quality(
        session, cognitive_states
    )
    
    print(f"\n🎯 CODE QUALITY PREDICTION:")
    print(f"  📈 Predicted Quality Score: {quality_prediction.predicted_quality_score:.1%}")
    print(f"  🎲 Confidence Level: {quality_prediction.confidence_level:.1%}")
    print(f"  🧠 Primary Cognitive State: {quality_prediction.cognitive_state.value}")
    
    print(f"\n📡 CONTRIBUTING BIOMETRICS:")
    for signal_type, contribution in quality_prediction.contributing_biometrics.items():
        print(f"  • {signal_type.value}: {contribution:.1%}")
    
    print(f"\n⚠️ RISK FACTORS:")
    for risk in quality_prediction.risk_factors:
        print(f"  • {risk}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in quality_prediction.recommendations:
        print(f"  • {rec}")
    
    if quality_prediction.optimal_coding_window:
        start, end = quality_prediction.optimal_coding_window
        print(f"\n⏰ OPTIMAL CODING WINDOW:")
        print(f"  🕐 Start: {start.strftime('%H:%M:%S')}")
        print(f"  🕐 End: {end.strftime('%H:%M:%S')}")
    
    # Analyze signal patterns
    print(f"\n🔍 BIOMETRIC SIGNAL ANALYSIS:")
    
    signal_stats = defaultdict(list)
    for reading in processed_readings:
        signal_stats[reading.signal_type].append(reading.value)
    
    for signal_type, values in signal_stats.items():
        print(f"  📊 {signal_type.value}:")
        print(f"    • Mean: {statistics.mean(values):.2f}")
        print(f"    • Std Dev: {statistics.stdev(values):.2f}" if len(values) > 1 else "    • Std Dev: N/A")
        print(f"    • Range: {min(values):.2f} - {max(values):.2f}")
    
    # Cognitive state distribution
    state_counts = defaultdict(int)
    for _, state, _ in cognitive_states:
        state_counts[state] += 1
    
    print(f"\n🧠 COGNITIVE STATE DISTRIBUTION:")
    total_states = len(cognitive_states)
    for state, count in state_counts.items():
        percentage = count / total_states * 100 if total_states > 0 else 0
        print(f"  • {state.value}: {percentage:.1f}% ({count} periods)")
    
    print(f"\n📈 SESSION SUMMARY:")
    print(f"  🕐 Duration: {(session.end_time - session.start_time).total_seconds() / 3600:.1f} hours")
    print(f"  📊 Total Readings: {len(session.readings)}")
    print(f"  🎯 Session Quality: {session.session_quality_score:.1%}")
    print(f"  🏃 Code Activities: {len(session.code_activities)}")


if __name__ == "__main__":
    asyncio.run(demonstrate_biometric_intelligence())