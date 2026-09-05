"""
SMART WOUND AI - Utility Functions
Feature engineering, predictions, and risk assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class WoundFeatureEngine:
    """
    Real-time temporal feature engineering from sensor readings
    """
    
    def __init__(self, biomarkers: List[str], window_size: int = 5):
        self.biomarkers = biomarkers
        self.window_size = window_size
    
    def create_features_from_readings(self, readings: List[Dict]) -> Optional[pd.DataFrame]:
        """
        Convert a list of sensor readings into the 24-feature temporal representation
        
        Args:
            readings: List of dicts with biomarker values
            
        Returns:
            DataFrame with 24 features, or None if insufficient readings
        """
        if len(readings) < 2:
            return None
        
        df = pd.DataFrame(readings)
        features_dict = {}
        
        for biomarker in self.biomarkers:
            if biomarker not in df.columns:
                continue
            
            values = df[biomarker].astype(float).values
            
            # Latest value
            features_dict[f"{biomarker}_last"] = values[-1]
            
            # Mean over window
            features_dict[f"{biomarker}_mean"] = np.mean(values)
            
            # Velocity (rate of change)
            features_dict[f"{biomarker}_velocity"] = (values[-1] - values[0]) / max(len(values) - 1, 1)
            
            # Variability (standard deviation)
            features_dict[f"{biomarker}_variability"] = np.std(values)
        
        return pd.DataFrame([features_dict])
    
    def validate_readings(self, readings: List[Dict]) -> Tuple[bool, str]:
        """Validate sensor reading format and values"""
        if not readings:
            return False, "No readings provided"
        
        if len(readings) < 2:
            return False, f"Need at least 2 readings, got {len(readings)}"
        
        # Check biomarker presence
        required = set(self.biomarkers)
        for reading in readings:
            provided = set(reading.keys())
            if not required.issubset(provided):
                missing = required - provided
                return False, f"Missing biomarkers: {missing}"
        
        # Check value ranges (alert only, not fatal)
        warnings = []
        ranges = {
            "pH": (6.0, 8.0),
            "temperature": (35, 40),
            "lactate": (0, 10),
            "protease": (0, 200),
            "glucose": (50, 300),
            "oxygen": (0, 100)
        }
        
        latest = readings[-1]
        for biomarker, (min_v, max_v) in ranges.items():
            if biomarker in latest:
                val = latest[biomarker]
                if val < min_v or val > max_v:
                    warnings.append(f"{biomarker}: {val} (expected {min_v}-{max_v})")
        
        return True, "\n".join(warnings) if warnings else "OK"


class RiskAssessment:
    """
    Convert infection probability to clinical risk levels and recommendations
    """
    
    # Prototype thresholds - NOT clinical validation
    THRESHOLDS = {
        "LOW": (0.0, 0.30),
        "MONITOR": (0.30, 0.60),
        "HIGH": (0.60, 1.0)
    }
    
    ACTIONS = {
        "LOW": {
            "risk_level": "LOW RISK",
            "emoji": "✅",
            "color": "green",
            "action": "Routine monitoring",
            "description": "No urgent alert. Continue standard wound care."
        },
        "MONITOR": {
            "risk_level": "MONITOR",
            "emoji": "⚠️",
            "color": "orange",
            "action": "Increase monitoring frequency",
            "description": "Clinical review recommended. Consider prophylactic measures."
        },
        "HIGH": {
            "risk_level": "HIGH RISK",
            "emoji": "🚨",
            "color": "red",
            "action": "Prompt clinical assessment",
            "description": "High-risk warning. Immediate clinical review advised."
        }
    }
    
    @classmethod
    def assess_risk(cls, probability: float) -> Dict:
        """
        Assess infection risk from model probability
        
        Args:
            probability: Model prediction probability [0, 1]
            
        Returns:
            Risk assessment dict with level, action, and details
        """
        risk_level = None
        for level, (min_p, max_p) in cls.THRESHOLDS.items():
            if min_p <= probability < max_p:
                risk_level = level
                break
        
        if risk_level is None:
            risk_level = "HIGH" if probability >= 0.60 else "MONITOR"
        
        action = cls.ACTIONS[risk_level]
        
        return {
            "probability": probability,
            "probability_pct": f"{probability:.1%}",
            "risk_level": action["risk_level"],
            "emoji": action["emoji"],
            "color": action["color"],
            "action": action["action"],
            "description": action["description"],
            "threshold_low": cls.THRESHOLDS[risk_level][0],
            "threshold_high": cls.THRESHOLDS[risk_level][1]
        }
    
    @classmethod
    def get_risk_details(cls, probability: float) -> Dict:
        """Get detailed risk assessment with clinical context"""
        assessment = cls.assess_risk(probability)
        
        # Risk progression context
        risk_distance = {
    "LOW RISK": max(0, 0.30 - probability),
    "MONITOR": max(0, probability - 0.45),
    "MEDIUM": max(0, probability - 0.45),
    "HIGH RISK": max(0, probability - 0.60)
}[assessment["risk_level"]]
        
        # Get trending context
        if probability < 0.15:
            trend = "Excellent - very low risk"
        elif probability < 0.30:
            trend = "Good - low risk"
        elif probability < 0.45:
            trend = "Moderate - monitor closely"
        elif probability < 0.60:
            trend = "Elevated - clinical review recommended"
        else:
            trend = "High - prompt assessment needed"
        
        assessment["trend"] = trend
        assessment["distance_to_threshold"] = risk_distance
        
        return assessment


class SensorSimulator:
    """
    Simulate realistic sensor readings for testing and demo
    """
    
    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
    
    def generate_normal_reading(self) -> Dict:
        """Generate healthy wound biomarkers"""
        return {
            "pH": self.rng.normal(6.9, 0.15),
            "temperature": self.rng.normal(36.8, 0.2),
            "lactate": self.rng.normal(2.2, 0.3),
            "protease": self.rng.normal(40, 8),
            "glucose": self.rng.normal(110, 15),
            "oxygen": self.rng.normal(75, 6)
        }
    
    def generate_elevated_risk_reading(self) -> Dict:
        """Generate biomarkers suggesting elevated infection risk"""
        return {
            "pH": self.rng.normal(6.55, 0.15),
            "temperature": self.rng.normal(37.4, 0.25),
            "lactate": self.rng.normal(3.4, 0.4),
            "protease": self.rng.normal(55, 10),
            "glucose": self.rng.normal(125, 20),
            "oxygen": self.rng.normal(64, 8)
        }
    
    def generate_deteriorating_trend(self, initial_reading: Dict, steps: int = 5) -> List[Dict]:
        """
        Generate a series of readings showing progressive deterioration
        (simulating infection development)
        """
        readings = [initial_reading.copy()]
        
        for _ in range(steps - 1):
            prev = readings[-1]
            deterioration = self.rng.uniform(0.8, 1.2)
            
            next_reading = {
                "pH": prev["pH"] - self.rng.normal(0.05, 0.03) * deterioration,
                "temperature": prev["temperature"] + self.rng.normal(0.1, 0.05) * deterioration,
                "lactate": prev["lactate"] + self.rng.normal(0.2, 0.1) * deterioration,
                "protease": prev["protease"] + self.rng.normal(5, 2) * deterioration,
                "glucose": prev["glucose"] + self.rng.normal(3, 2) * deterioration,
                "oxygen": prev["oxygen"] - self.rng.normal(2, 1) * deterioration
            }
            readings.append(next_reading)
        
        return readings


class TemporalAnalyzer:
    """
    Analyze temporal trends in biomarker readings
    """
    
    @staticmethod
    def calculate_trends(readings: List[Dict], biomarkers: List[str]) -> Dict:
        """Calculate velocity and variability for each biomarker"""
        if len(readings) < 2:
            return {}
        
        df = pd.DataFrame(readings)
        trends = {}
        
        for biomarker in biomarkers:
            if biomarker not in df.columns:
                continue
            
            values = df[biomarker].astype(float).values
            
            # Velocity (rate of change)
            velocity = (values[-1] - values[0]) / (len(values) - 1)
            
            # Variability
            variability = np.std(values)
            
            # Direction
            if velocity > 0.01:
                direction = "increasing"
            elif velocity < -0.01:
                direction = "decreasing"
            else:
                direction = "stable"
            
            trends[biomarker] = {
                "latest": float(values[-1]),
                "mean": float(np.mean(values)),
                "velocity": float(velocity),
                "variability": float(variability),
                "direction": direction,
                "change_pct": float((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
            }
        
        return trends
    
    @staticmethod
    def identify_concerning_trends(readings: List[Dict], biomarkers: List[str]) -> List[str]:
        """Identify biomarkers showing concerning patterns"""
        if len(readings) < 2:
            return []
        
        trends = TemporalAnalyzer.calculate_trends(readings, biomarkers)
        concerning = []
        
        # Define concerning patterns
        concern_rules = {
            "pH": lambda v: v["velocity"] < -0.02,  # Decreasing pH
            "temperature": lambda v: v["velocity"] > 0.08,  # Rising temperature
            "lactate": lambda v: v["velocity"] > 0.15,  # Rising lactate
            "protease": lambda v: v["velocity"] > 3,  # Rising protease
            "glucose": lambda v: v["velocity"] > 3,  # Rising glucose
            "oxygen": lambda v: v["velocity"] < -1.5  # Falling oxygen
        }
        
        for biomarker, trend_data in trends.items():
            if biomarker in concern_rules:
                if concern_rules[biomarker](trend_data):
                    concerning.append(biomarker)
        
        return concerning


# Utility functions
def format_probability_display(probability: float) -> str:
    """Format probability as percentage with appropriate precision"""
    return f"{probability:.1%}"


def get_model_metadata() -> Dict:
    """Get standard model metadata for display"""
    return {
        "name": "SMART WOUND AI",
        "version": "1.0-prototype",
        "biomarkers_count": 6,
        "features_count": 24,
        "model_type": "Random Forest",
        "window_size": 5,
        "note": "Simulation-based research prototype. Not clinically validated."
    }


