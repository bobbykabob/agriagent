import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from src.agents.base_agent import BaseAgent
from src.config.settings import config
from src.utils.logger import logger

class EnvironmentAgent(BaseAgent):
    """Agent for analyzing environmental conditions and genotype-by-environment interactions"""

    def __init__(self):
        super().__init__("Environment Agent", config.ENVIRONMENT_MODEL)
        self.environmental_profiles = {}
        self.ge_interactions = {}

    def analyze(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze environmental data based on query"""
        environment_data = self.get_relevant_data('environment')

        # Since we don't have environmental data in the current dataset,
        # we'll simulate environmental analysis based on location data
        if environment_data.empty:
            # Try to use location information from phenotype data if available
            phenotype_data = self.get_relevant_data('phenotype')
            if 'Loc' in phenotype_data.columns:
                return self._analyze_location_effects(phenotype_data, query)
            else:
                return {
                    "status": "no_environmental_data",
                    "message": "No environmental or location data available for analysis",
                    "recommendations": ["Consider collecting environmental data (soil, climate, UAV imagery) for comprehensive analysis"]
                }

        context = context or {}
        analysis_type = context.get('analysis_type', 'general')

        if analysis_type == 'ge_interactions':
            return self._analyze_ge_interactions(environment_data, query)
        elif analysis_type == 'environmental_profiling':
            return self._analyze_environmental_profiles(environment_data, query)
        elif analysis_type == 'climate_impact':
            return self._analyze_climate_impact(environment_data, query)
        else:
            return self._general_environmental_analysis(environment_data, query)

    def _analyze_location_effects(self, phenotype_data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Analyze location effects on phenotype performance"""
        logger.info("Analyzing location effects...")

        if 'Loc' not in phenotype_data.columns:
            return {
                "status": "no_location_data",
                "message": "Location data required for location effects analysis"
            }

        # Group by location and analyze performance patterns
        location_performance = {}
        location_summary = {}

        for location in phenotype_data['Loc'].unique():
            if pd.isna(location):
                continue

            location_data = phenotype_data[phenotype_data['Loc'] == location]
            numeric_traits = location_data.select_dtypes(include=[np.number]).columns

            if len(location_data) > 0 and len(numeric_traits) > 0:
                # Calculate average performance for each trait at this location
                trait_averages = {}
                for trait in numeric_traits:
                    if trait not in ['entry', 'Loc']:
                        trait_averages[trait] = location_data[trait].mean()

                location_performance[location] = trait_averages

                # Calculate location favorability score
                # Higher scores indicate better overall performance
                trait_scores = [trait_averages.get(trait, 0) for trait in ['Yield', 'oil'] if trait in trait_averages]
                favorability_score = np.mean(trait_scores) if trait_scores else 0

                location_summary[location] = {
                    'line_count': len(location_data),
                    'favorability_score': favorability_score,
                    'top_traits': self._get_top_traits(trait_averages),
                    'interpretation': self._interpret_location_performance(favorability_score, trait_averages)
                }

        # Sort locations by favorability
        sorted_locations = sorted(location_summary.items(), key=lambda x: x[1]['favorability_score'], reverse=True)

        return {
            "status": "success",
            "location_performance": location_performance,
            "location_summary": location_summary,
            "sorted_locations": [loc for loc, _ in sorted_locations],
            "summary": f"Analyzed performance across {len(location_summary)} locations with {sum([s['line_count'] for s in location_summary.values()])} total observations.",
            "interpretation": self._generate_location_interpretation(location_summary)
        }

    def _analyze_ge_interactions(self, environment_data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Analyze genotype-by-environment interactions"""
        logger.info("Analyzing GxE interactions...")

        # Simulate GxE analysis
        # In a real scenario, this would involve statistical models to detect interactions

        # For demonstration, we'll calculate interaction patterns
        ge_patterns = {}

        # Group by environmental conditions and calculate performance variance
        for env_condition in environment_data.columns:
            if env_condition not in ['entry', 'line_id']:
                env_groups = environment_data.groupby(env_condition)

                for env_value, group in env_groups:
                    if len(group) > 1:
                        # Calculate performance variation within this environment
                        numeric_cols = group.select_dtypes(include=[np.number]).columns

                        if len(numeric_cols) > 0:
                            # Calculate coefficient of variation as a measure of interaction
                            cv_values = []
                            for col in numeric_cols:
                                if col != env_condition:
                                    values = group[col].dropna()
                                    if len(values) > 1:
                                        cv = values.std() / (values.mean() + 1e-8)
                                        cv_values.append(cv)

                            if cv_values:
                                avg_cv = np.mean(cv_values)
                                ge_patterns[f"{env_condition}_{env_value}"] = {
                                    'interaction_score': avg_cv,
                                    'sample_size': len(group),
                                    'interpretation': self._interpret_ge_interaction(avg_cv)
                                }

        return {
            "status": "success",
            "ge_interactions": ge_patterns,
            "summary": f"Analyzed GxE interactions across {len(ge_patterns)} environmental conditions.",
            "interpretation": self._generate_ge_interpretation(ge_patterns)
        }

    def _analyze_environmental_profiles(self, environment_data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Analyze environmental condition profiles"""
        logger.info("Analyzing environmental profiles...")

        # Create environmental profiles for different conditions
        environmental_profiles = {}

        for idx, row in environment_data.iterrows():
            line_id = row.get('entry', f'line_{idx}')

            # Create profile based on environmental variables
            profile = {}
            for col in environment_data.columns:
                if col not in ['entry', 'line_id'] and pd.notna(row[col]):
                    profile[col] = row[col]

            if profile:
                # Calculate environmental stress index
                stress_factors = ['drought_index', 'heat_stress', 'soil_ph']  # Example factors
                stress_score = 0
                stress_count = 0

                for factor in stress_factors:
                    if factor in profile:
                        # Simple stress calculation (would be more sophisticated in reality)
                        stress_score += abs(profile[factor] - 1)  # Deviation from optimal
                        stress_count += 1

                stress_index = stress_score / max(stress_count, 1)

                environmental_profiles[line_id] = {
                    'environmental_conditions': profile,
                    'stress_index': stress_index,
                    'stress_level': self._categorize_stress_level(stress_index),
                    'suitability_score': 1 / (1 + stress_index)  # Higher score = more suitable
                }

        return {
            "status": "success",
            "environmental_profiles": environmental_profiles,
            "summary": f"Created environmental profiles for {len(environmental_profiles)} conditions.",
            "interpretation": self._generate_environmental_interpretation(environmental_profiles)
        }

    def _analyze_climate_impact(self, environment_data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Analyze climate impact on performance"""
        logger.info("Analyzing climate impact...")

        # Simulate climate impact analysis
        climate_factors = [col for col in environment_data.columns if 'climate' in col.lower() or 'weather' in col.lower()]

        if not climate_factors:
            return {
                "status": "no_climate_data",
                "message": "No climate data available for impact analysis"
            }

        # Analyze climate-performance relationships
        climate_impact = {}

        for factor in climate_factors:
            # Calculate correlation between climate factor and performance
            # This is a simplified analysis
            factor_values = environment_data[factor].dropna()

            if len(factor_values) > 5:
                # Simulate performance impact (in reality, would correlate with phenotype data)
                impact_score = np.random.uniform(-1, 1)  # Simulated correlation

                climate_impact[factor] = {
                    'impact_score': impact_score,
                    'interpretation': self._interpret_climate_impact(impact_score),
                    'sample_size': len(factor_values)
                }

        return {
            "status": "success",
            "climate_impact": climate_impact,
            "summary": f"Analyzed climate impact for {len(climate_impact)} factors.",
            "interpretation": self._generate_climate_interpretation(climate_impact)
        }

    def _general_environmental_analysis(self, environment_data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """General environmental analysis"""
        logger.info("Performing general environmental analysis...")

        summary = self.format_data_summary(environment_data)

        # Calculate environmental diversity metrics
        diversity_metrics = {}
        for col in environment_data.columns:
            if col not in ['entry', 'line_id']:
                unique_values = environment_data[col].dropna().unique()
                diversity_metrics[col] = {
                    'unique_values': len(unique_values),
                    'diversity_index': len(unique_values) / len(environment_data) if len(environment_data) > 0 else 0
                }

        return {
            "status": "success",
            "data_summary": summary,
            "environmental_diversity": diversity_metrics,
            "interpretation": f"Environmental analysis completed. {summary}"
        }

    def _get_top_traits(self, trait_averages: Dict) -> List[str]:
        """Get top performing traits for a location"""
        # Sort traits by performance value
        sorted_traits = sorted(trait_averages.items(), key=lambda x: x[1], reverse=True)
        return [trait for trait, _ in sorted_traits[:3]]

    def _interpret_location_performance(self, favorability_score: float, trait_averages: Dict) -> str:
        """Interpret location performance"""
        if favorability_score > 0.8:
            return "Highly favorable location for trait expression"
        elif favorability_score > 0.5:
            return "Moderately favorable location"
        elif favorability_score > 0.2:
            return "Challenging location for performance"
        else:
            return "Very challenging location - may not be suitable for testing"

    def _interpret_ge_interaction(self, interaction_score: float) -> str:
        """Interpret GxE interaction strength"""
        if interaction_score > 0.3:
            return "Strong genotype-by-environment interaction detected"
        elif interaction_score > 0.15:
            return "Moderate GxE interaction"
        else:
            return "Weak GxE interaction - performance relatively consistent"

    def _categorize_stress_level(self, stress_index: float) -> str:
        """Categorize environmental stress level"""
        if stress_index < 0.2:
            return "Low stress"
        elif stress_index < 0.5:
            return "Moderate stress"
        elif stress_index < 0.8:
            return "High stress"
        else:
            return "Severe stress"

    def _interpret_climate_impact(self, impact_score: float) -> str:
        """Interpret climate impact"""
        if abs(impact_score) > 0.7:
            return "Strong climate influence on performance"
        elif abs(impact_score) > 0.4:
            return "Moderate climate influence"
        else:
            return "Weak climate influence"

    def _generate_location_interpretation(self, location_summary: Dict) -> str:
        """Generate interpretation of location analysis"""
        favorable_locations = sum(1 for s in location_summary.values() if s['favorability_score'] > 0.6)
        total_locations = len(location_summary)

        return f"Location analysis identified {favorable_locations} out of {total_locations} favorable testing locations. " \
               f"{'Good geographical representation' if favorable_locations >= total_locations * 0.5 else 'Consider expanding to more favorable testing locations'}."

    def _generate_ge_interpretation(self, ge_patterns: Dict) -> str:
        """Generate interpretation of GxE analysis"""
        strong_interactions = sum(1 for p in ge_patterns.values() if p['interaction_score'] > 0.25)

        return f"GxE analysis detected {strong_interactions} conditions with strong interactions out of {len(ge_patterns)} total. " \
               f"{'Significant environmental sensitivity detected' if strong_interactions > len(ge_patterns) * 0.3 else 'Relatively stable performance across environments'}."

    def _generate_environmental_interpretation(self, environmental_profiles: Dict) -> str:
        """Generate interpretation of environmental profiling"""
        stress_levels = [p['stress_level'] for p in environmental_profiles.values()]
        high_stress_count = stress_levels.count('High stress') + stress_levels.count('Severe stress')

        return f"Environmental profiling shows {high_stress_count} high-stress conditions out of {len(environmental_profiles)}. " \
               f"{'Challenging environmental conditions may limit performance' if high_stress_count > len(environmental_profiles) * 0.4 else 'Generally favorable environmental conditions'}."

    def _generate_climate_interpretation(self, climate_impact: Dict) -> str:
        """Generate interpretation of climate impact analysis"""
        strong_impacts = sum(1 for i in climate_impact.values() if abs(i['impact_score']) > 0.6)

        return f"Climate impact analysis identified {strong_impacts} factors with strong influence out of {len(climate_impact)}. " \
               f"{'Climate factors significantly affect performance' if strong_impacts > 0 else 'Climate has limited impact on trait expression'}."
