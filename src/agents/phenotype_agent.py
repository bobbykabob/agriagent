import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler
from src.agents.base_agent import BaseAgent
from src.config.settings import config
from src.utils.logger import logger

class PhenotypeAgent(BaseAgent):
    """Agent for analyzing phenotype data and trait-level observations"""

    def __init__(self):
        super().__init__("Phenotype Agent", config.PHENOTYPE_MODEL)
        self.trait_analyses = {}
        self.performance_scores = {}

    def analyze(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze phenotype data based on query"""
        phenotype_data = self.get_relevant_data('phenotype')

        if phenotype_data.empty:
            return {
                "status": "no_data",
                "message": "No phenotype data available for analysis",
                "recommendations": []
            }

        context = context or {}
        analysis_type = context.get('analysis_type', 'general')

        if analysis_type == 'trait_correlation':
            return self._analyze_trait_correlations(phenotype_data, query)
        elif analysis_type == 'performance_ranking':
            return self._analyze_performance_ranking(phenotype_data, query, context)
        elif analysis_type == 'stability_analysis':
            return self._analyze_stability(phenotype_data, query)
        elif analysis_type == 'breeding_value':
            return self._estimate_breeding_values(phenotype_data, query)
        else:
            return self._general_phenotype_analysis(phenotype_data, query)

    def _analyze_trait_correlations(self, phenotype_data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Analyze correlations between different traits"""
        logger.info("Analyzing trait correlations...")

        # Select numeric traits for correlation analysis
        numeric_traits = phenotype_data.select_dtypes(include=[np.number]).columns

        if len(numeric_traits) < 2:
            return {
                "status": "insufficient_traits",
                "message": "Need at least 2 numeric traits for correlation analysis",
                "correlations": {}
            }

        # Calculate correlation matrix
        correlation_matrix = phenotype_data[numeric_traits].corr()

        # Find significant correlations
        significant_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.3:  # Consider correlations above 0.3 as significant
                    significant_correlations.append({
                        'trait1': correlation_matrix.columns[i],
                        'trait2': correlation_matrix.columns[j],
                        'correlation': corr_val,
                        'interpretation': self._interpret_correlation(corr_val)
                    })

        significant_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

        return {
            "status": "success",
            "correlation_matrix": correlation_matrix.to_dict(),
            "significant_correlations": significant_correlations,
            "summary": f"Analyzed correlations between {len(numeric_traits)} traits. Found {len(significant_correlations)} significant correlations (|r| > 0.3).",
            "interpretation": self._generate_correlation_interpretation(significant_correlations)
        }

    def _analyze_performance_ranking(self, phenotype_data: pd.DataFrame, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rank lines based on phenotype performance"""
        logger.info("Analyzing performance ranking...")

        target_traits = context.get('target_traits', config.TARGET_TRAITS)
        available_traits = [trait for trait in target_traits if trait in phenotype_data.columns]

        if not available_traits:
            return {
                "status": "no_target_traits",
                "message": f"None of the target traits {target_traits} found in phenotype data",
                "rankings": {}
            }

        # Calculate performance scores for each line
        performance_scores = {}

        for idx, row in phenotype_data.iterrows():
            line_id = row.get('entry', f'line_{idx}')

            # Calculate composite performance score
            trait_scores = []
            for trait in available_traits:
                if trait in row.index and pd.notna(row[trait]):
                    # For yield and oil content, higher is better
                    if trait in ['Yield', 'oil']:
                        trait_scores.append(row[trait])
                    # For plant height, moderate values might be preferred
                    elif trait == 'Plant Height':
                        # Assume moderate height is preferred (avoid extremes)
                        height_score = 1 - abs(row[trait] - 100) / 100  # Assume 100cm is ideal
                        trait_scores.append(max(0, height_score))
                    # For breeder scores, higher is better
                    elif 'Score' in trait:
                        trait_scores.append(row[trait])
                    else:
                        trait_scores.append(row[trait])

            if trait_scores:
                # Calculate composite score (average of standardized scores)
                composite_score = np.mean(trait_scores)
            else:
                composite_score = 0.0

            performance_scores[line_id] = {
                'composite_score': composite_score,
                'trait_scores': {trait: row.get(trait, None) for trait in available_traits},
                'rank_category': self._categorize_performance(composite_score)
            }

        # Sort by performance score
        sorted_lines = sorted(performance_scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)

        rankings = {}
        for rank, (line_id, scores) in enumerate(sorted_lines, 1):
            rankings[line_id] = {
                'rank': rank,
                'composite_score': scores['composite_score'],
                'trait_scores': scores['trait_scores'],
                'category': scores['rank_category']
            }

        # Top performers
        top_performers = [line_id for line_id, _ in sorted_lines[:int(len(sorted_lines) * 0.1)]]

        return {
            "status": "success",
            "rankings": rankings,
            "top_performers": top_performers,
            "summary": f"Ranked {len(rankings)} lines based on {len(available_traits)} traits. Top {len(top_performers)} performers identified.",
            "interpretation": self._generate_ranking_interpretation(rankings, available_traits)
        }

    def _analyze_stability(self, phenotype_data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Analyze phenotypic stability across environments"""
        logger.info("Analyzing phenotypic stability...")

        # For stability analysis, we need data from multiple locations or years
        location_col = 'Loc' if 'Loc' in phenotype_data.columns else None

        if location_col is None:
            return {
                "status": "no_location_data",
                "message": "Location data required for stability analysis",
                "stability_scores": {}
            }

        # Group by location and calculate stability metrics
        stability_scores = {}

        for idx, row in phenotype_data.iterrows():
            line_id = row.get('entry', f'line_{idx}')

            # Calculate coefficient of variation across locations (if multiple locations per line)
            line_data = phenotype_data[phenotype_data['entry'] == row['entry']]
            numeric_traits = line_data.select_dtypes(include=[np.number]).columns

            if len(line_data) > 1 and len(numeric_traits) > 0:
                # Calculate stability as inverse of coefficient of variation
                cv_scores = []
                for trait in numeric_traits:
                    if trait != 'entry' and trait != location_col:
                        trait_values = line_data[trait].dropna()
                        if len(trait_values) > 1:
                            cv = trait_values.std() / (trait_values.mean() + 1e-8)
                            cv_scores.append(cv)

                if cv_scores:
                    avg_cv = np.mean(cv_scores)
                    stability_score = 1 / (1 + avg_cv)  # Higher score = more stable
                else:
                    stability_score = 0.0
            else:
                stability_score = 0.0

            stability_scores[line_id] = {
                'stability_score': stability_score,
                'locations_count': len(line_data),
                'interpretation': self._interpret_stability_score(stability_score)
            }

        # Sort by stability
        sorted_lines = sorted(stability_scores.items(), key=lambda x: x[1]['stability_score'], reverse=True)
        most_stable = [line_id for line_id, _ in sorted_lines[:10]]

        return {
            "status": "success",
            "stability_scores": stability_scores,
            "most_stable_lines": most_stable,
            "summary": f"Analyzed stability for {len(stability_scores)} lines across locations.",
            "interpretation": self._generate_stability_interpretation(stability_scores)
        }

    def _estimate_breeding_values(self, phenotype_data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Estimate breeding values for selection"""
        logger.info("Estimating breeding values...")

        # Simple breeding value estimation based on trait performance
        numeric_traits = phenotype_data.select_dtypes(include=[np.number]).columns
        breeding_values = {}

        for idx, row in phenotype_data.iterrows():
            line_id = row.get('entry', f'line_{idx}')

            # Calculate breeding value as average standardized trait performance
            trait_values = []
            for trait in numeric_traits:
                if trait != 'entry' and pd.notna(row[trait]):
                    # Standardize trait value
                    trait_mean = phenotype_data[trait].mean()
                    trait_std = phenotype_data[trait].std()

                    if trait_std > 0:
                        standardized_value = (row[trait] - trait_mean) / trait_std
                        trait_values.append(standardized_value)

            if trait_values:
                breeding_value = np.mean(trait_values)
            else:
                breeding_value = 0.0

            breeding_values[line_id] = {
                'breeding_value': breeding_value,
                'trait_contributions': {trait: row.get(trait, None) for trait in numeric_traits if trait != 'entry'},
                'selection_priority': self._get_selection_priority(breeding_value)
            }

        # Sort by breeding value
        sorted_lines = sorted(breeding_values.items(), key=lambda x: x[1]['breeding_value'], reverse=True)
        high_value_lines = [line_id for line_id, _ in sorted_lines[:int(len(sorted_lines) * 0.2)]]  # Top 20%

        return {
            "status": "success",
            "breeding_values": breeding_values,
            "high_value_lines": high_value_lines,
            "summary": f"Estimated breeding values for {len(breeding_values)} lines across {len(numeric_traits)} traits.",
            "interpretation": self._generate_breeding_value_interpretation(breeding_values)
        }

    def _general_phenotype_analysis(self, phenotype_data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """General phenotype analysis"""
        logger.info("Performing general phenotype analysis...")

        summary = self.format_data_summary(phenotype_data)

        # Calculate basic statistics for key traits
        numeric_traits = phenotype_data.select_dtypes(include=[np.number]).columns
        trait_stats = {}

        for trait in numeric_traits:
            if trait != 'entry':
                trait_stats[trait] = {
                    'mean': phenotype_data[trait].mean(),
                    'std': phenotype_data[trait].std(),
                    'min': phenotype_data[trait].min(),
                    'max': phenotype_data[trait].max(),
                    'cv': phenotype_data[trait].std() / (phenotype_data[trait].mean() + 1e-8) if phenotype_data[trait].mean() > 0 else 0
                }

        return {
            "status": "success",
            "data_summary": summary,
            "trait_statistics": trait_stats,
            "interpretation": f"General analysis completed for {len(phenotype_data)} lines with {len(numeric_traits)} measured traits. {summary}"
        }

    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation strength"""
        abs_corr = abs(correlation)
        if abs_corr > 0.7:
            return "Strong correlation"
        elif abs_corr > 0.5:
            return "Moderate correlation"
        elif abs_corr > 0.3:
            return "Weak correlation"
        else:
            return "Very weak correlation"

    def _categorize_performance(self, score: float) -> str:
        """Categorize performance level"""
        if score > 0.8:
            return "Elite"
        elif score > 0.5:
            return "Good"
        elif score > 0.2:
            return "Average"
        else:
            return "Poor"

    def _interpret_stability_score(self, score: float) -> str:
        """Interpret stability score"""
        if score > 0.8:
            return "Highly stable across environments"
        elif score > 0.6:
            return "Moderately stable"
        elif score > 0.4:
            return "Somewhat variable"
        else:
            return "Highly variable across environments"

    def _get_selection_priority(self, breeding_value: float) -> str:
        """Get selection priority based on breeding value"""
        if breeding_value > 1.5:
            return "High priority for advancement"
        elif breeding_value > 0.5:
            return "Medium priority for advancement"
        elif breeding_value > -0.5:
            return "Low priority, monitor performance"
        else:
            return "Consider elimination from program"

    def _generate_correlation_interpretation(self, correlations: List[Dict]) -> str:
        """Generate interpretation of correlation analysis"""
        if not correlations:
            return "No significant correlations found between traits."

        strong_correlations = [c for c in correlations if abs(c['correlation']) > 0.7]
        return f"Found {len(correlations)} significant trait correlations. " \
               f"{len(strong_correlations)} are strong (|r| > 0.7). " \
               f"This suggests {'tight linkage between traits' if len(strong_correlations) > 3 else 'relatively independent trait expression'}."

    def _generate_ranking_interpretation(self, rankings: Dict, traits: List[str]) -> str:
        """Generate interpretation of performance ranking"""
        elite_count = sum(1 for r in rankings.values() if r['category'] == 'Elite')
        return f"Performance ranking identified {elite_count} elite lines out of {len(rankings)} total lines. " \
               f"Selection should prioritize {'yield and quality traits' if 'Yield' in traits else 'key agronomic traits'} for maximum impact."

    def _generate_stability_interpretation(self, stability_scores: Dict) -> str:
        """Generate interpretation of stability analysis"""
        stable_count = sum(1 for s in stability_scores.values() if s['stability_score'] > 0.7)
        return f"Stability analysis identified {stable_count} highly stable lines. " \
               f"{'Good environmental adaptation detected' if stable_count > len(stability_scores) * 0.2 else 'Consider improving environmental stability'} in the breeding program."

    def _generate_breeding_value_interpretation(self, breeding_values: Dict) -> str:
        """Generate interpretation of breeding value estimation"""
        high_priority_count = sum(1 for bv in breeding_values.values() if bv['breeding_value'] > 1.0)
        return f"Breeding value analysis identified {high_priority_count} high-priority lines for advancement. " \
               f"Focus selection efforts on {'top performers' if high_priority_count > 0 else 'improving overall population performance'}."
