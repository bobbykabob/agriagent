import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from src.agents.base_agent import BaseAgent
from src.config.settings import config
from src.utils.logger import logger

class ControllerAgent(BaseAgent):
    """Controller Agent that fuses reasoning from domain agents to make advancement decisions"""

    def __init__(self):
        super().__init__("Controller Agent", config.CONTROLLER_MODEL)
        self.agent_analyses = {}
        self.advancement_decisions = {}

    def analyze(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make integrated advancement decisions based on all agent analyses"""
        if not self.agent_analyses:
            return {
                "status": "no_analyses",
                "message": "No agent analyses available for decision making",
                "decisions": {}
            }

        context = context or {}
        decision_type = context.get('decision_type', 'advancement')

        if decision_type == 'advancement':
            return self._make_advancement_decisions(query, context)
        elif decision_type == 'prioritization':
            return self._prioritize_lines(query, context)
        elif decision_type == 'risk_assessment':
            return self._assess_risks(query, context)
        else:
            return self._general_decision_making(query, context)

    def set_agent_analyses(self, agent_analyses: Dict[str, Dict[str, Any]]):
        """Set analyses from all domain agents"""
        self.agent_analyses = agent_analyses

    def _make_advancement_decisions(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make line advancement decisions based on integrated analysis"""
        logger.info("Making integrated advancement decisions...")

        # Extract key insights from each agent
        genotype_insights = self.agent_analyses.get('genotype', {})
        phenotype_insights = self.agent_analyses.get('phenotype', {})
        environment_insights = self.agent_analyses.get('environment', {})

        # Get candidate lines from each agent
        genotype_candidates = genotype_insights.get('recommended_lines', [])
        phenotype_candidates = phenotype_insights.get('top_performers', [])
        environment_candidates = environment_insights.get('most_stable_lines', [])

        # Combine candidate lists
        all_candidates = set(genotype_candidates + phenotype_candidates + environment_candidates)

        # Default weights for integration
        default_weights = {'genotype': 0.3, 'phenotype': 0.5, 'environment': 0.2}

        # Calculate integrated scores for each candidate
        integrated_scores = {}

        for line_id in all_candidates:
            # Multi-criteria scoring
            genotype_score = self._calculate_agent_score(genotype_insights, line_id, 'genotype')
            phenotype_score = self._calculate_agent_score(phenotype_insights, line_id, 'phenotype')
            environment_score = self._calculate_agent_score(environment_insights, line_id, 'environment')

            # Weighted integration (can be adjusted based on breeding program priorities)
            integrated_score = (
                genotype_score * default_weights['genotype'] +
                phenotype_score * default_weights['phenotype'] +
                environment_score * default_weights['environment']
            )

            # Risk adjustment based on stability and consistency
            risk_factor = self._calculate_risk_factor(genotype_insights, phenotype_insights, environment_insights, line_id)
            adjusted_score = integrated_score * (1 - risk_factor * 0.1)  # Reduce score for high-risk lines

            integrated_scores[line_id] = {
                'integrated_score': integrated_score,
                'adjusted_score': adjusted_score,
                'agent_scores': {
                    'genotype': genotype_score,
                    'phenotype': phenotype_score,
                    'environment': environment_score
                },
                'risk_factor': risk_factor,
                'decision': self._make_single_advancement_decision(adjusted_score, risk_factor)
            }

        # Sort by adjusted score
        sorted_lines = sorted(integrated_scores.items(), key=lambda x: x[1]['adjusted_score'], reverse=True)

        # Apply advancement threshold
        advancement_threshold = context.get('advancement_threshold', config.ADVANCEMENT_THRESHOLD)
        top_percentage = context.get('top_percentage', config.TOP_LINES_PERCENTAGE)

        # Determine advancement cutoff
        n_advance = max(int(len(sorted_lines) * top_percentage), 1)
        cutoff_score = sorted_lines[n_advance - 1][1]['adjusted_score'] if n_advance <= len(sorted_lines) else 0

        final_decisions = {}
        for line_id, scores in sorted_lines:
            should_advance = scores['adjusted_score'] >= max(cutoff_score, advancement_threshold)
            scores['final_decision'] = 'advance' if should_advance else 'do_not_advance'

            if should_advance:
                scores['priority_rank'] = len([s for s in final_decisions.values() if s['final_decision'] == 'advance']) + 1

            final_decisions[line_id] = scores

        # Generate summary statistics
        advanced_lines = [line_id for line_id, decision in final_decisions.items() if decision['final_decision'] == 'advance']
        not_advanced_lines = [line_id for line_id, decision in final_decisions.items() if decision['final_decision'] == 'do_not_advance']

        return {
            "status": "success",
            "advancement_decisions": final_decisions,
            "advanced_lines": advanced_lines,
            "not_advanced_lines": not_advanced_lines,
            "summary": f"Evaluated {len(final_decisions)} candidate lines. Recommended {len(advanced_lines)} for advancement based on integrated analysis.",
            "interpretation": self._generate_advancement_interpretation(final_decisions, advanced_lines),
            "methodology": {
                "weights": default_weights,
                "threshold": advancement_threshold,
                "top_percentage": top_percentage
            }
        }

    def _prioritize_lines(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prioritize lines for different breeding activities"""
        logger.info("Prioritizing lines for breeding activities...")

        # Similar to advancement decisions but with different priorities
        genotype_insights = self.agent_analyses.get('genotype', {})
        phenotype_insights = self.agent_analyses.get('phenotype', {})
        environment_insights = self.agent_analyses.get('environment', {})

        # Get all unique lines
        all_lines = set()
        for insights in [genotype_insights, phenotype_insights, environment_insights]:
            for key in ['recommended_lines', 'top_performers', 'most_stable_lines']:
                if key in insights:
                    all_lines.update(insights[key])

        # Calculate priority scores with different weightings for different activities
        activity_priorities = {
            'yield_improvement': {'genotype': 0.2, 'phenotype': 0.7, 'environment': 0.1},
            'stress_tolerance': {'genotype': 0.3, 'phenotype': 0.3, 'environment': 0.4},
            'quality_improvement': {'genotype': 0.4, 'phenotype': 0.5, 'environment': 0.1},
            'broad_adaptation': {'genotype': 0.3, 'phenotype': 0.3, 'environment': 0.4}
        }

        prioritization_results = {}

        for activity, weights in activity_priorities.items():
            activity_scores = {}

            for line_id in all_lines:
                genotype_score = self._calculate_agent_score(genotype_insights, line_id, 'genotype')
                phenotype_score = self._calculate_agent_score(phenotype_insights, line_id, 'phenotype')
                environment_score = self._calculate_agent_score(environment_insights, line_id, 'environment')

                priority_score = (
                    genotype_score * weights['genotype'] +
                    phenotype_score * weights['phenotype'] +
                    environment_score * weights['environment']
                )

                activity_scores[line_id] = {
                    'priority_score': priority_score,
                    'suitability': self._categorize_suitability(priority_score)
                }

            # Sort and select top lines for each activity
            sorted_lines = sorted(activity_scores.items(), key=lambda x: x[1]['priority_score'], reverse=True)
            top_lines = [line_id for line_id, _ in sorted_lines[:5]]  # Top 5 for each activity

            prioritization_results[activity] = {
                'top_lines': top_lines,
                'scores': {line_id: scores for line_id, scores in sorted_lines[:10]},  # Top 10 details
                'summary': f"Selected {len(top_lines)} top lines for {activity.replace('_', ' ')}"
            }

        return {
            "status": "success",
            "prioritization_results": prioritization_results,
            "summary": f"Prioritized lines for {len(activity_priorities)} different breeding activities.",
            "interpretation": self._generate_prioritization_interpretation(prioritization_results)
        }

    def _assess_risks(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks associated with line advancement"""
        logger.info("Assessing advancement risks...")

        risk_assessments = {}

        for line_id in self._get_all_candidate_lines():
            # Calculate different risk factors
            genetic_risk = self._calculate_genetic_risk(line_id)
            phenotypic_risk = self._calculate_phenotypic_risk(line_id)
            environmental_risk = self._calculate_environmental_risk(line_id)

            # Overall risk score
            overall_risk = (genetic_risk + phenotypic_risk + environmental_risk) / 3

            risk_assessments[line_id] = {
                'overall_risk': overall_risk,
                'risk_factors': {
                    'genetic': genetic_risk,
                    'phenotypic': phenotypic_risk,
                    'environmental': environmental_risk
                },
                'risk_level': self._categorize_risk_level(overall_risk),
                'mitigation_strategies': self._get_mitigation_strategies(genetic_risk, phenotypic_risk, environmental_risk)
            }

        # Sort by risk level
        high_risk_lines = [line_id for line_id, risk in risk_assessments.items() if risk['overall_risk'] > 0.7]
        medium_risk_lines = [line_id for line_id, risk in risk_assessments.items() if 0.4 < risk['overall_risk'] <= 0.7]
        low_risk_lines = [line_id for line_id, risk in risk_assessments.items() if risk['overall_risk'] <= 0.4]

        return {
            "status": "success",
            "risk_assessments": risk_assessments,
            "risk_summary": {
                'high_risk': high_risk_lines,
                'medium_risk': medium_risk_lines,
                'low_risk': low_risk_lines
            },
            "summary": f"Assessed risks for {len(risk_assessments)} lines. {len(high_risk_lines)} high-risk, {len(medium_risk_lines)} medium-risk, {len(low_risk_lines)} low-risk lines identified.",
            "interpretation": self._generate_risk_interpretation(risk_assessments)
        }

    def _general_decision_making(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """General decision making based on integrated analysis"""
        logger.info("Performing general decision making...")

        # Provide overall program recommendations
        recommendations = []

        # Analyze agent consensus
        genotype_count = len(self.agent_analyses.get('genotype', {}).get('recommended_lines', []))
        phenotype_count = len(self.agent_analyses.get('phenotype', {}).get('top_performers', []))
        environment_count = len(self.agent_analyses.get('environment', {}).get('most_stable_lines', []))

        # Generate program-level insights
        if genotype_count > phenotype_count:
            recommendations.append("Genetic diversity appears to be a strength - consider leveraging this for future breeding")
        elif phenotype_count > genotype_count:
            recommendations.append("Phenotypic performance is strong - focus on maintaining these gains")

        if environment_count < 10:  # Arbitrary threshold
            recommendations.append("Consider expanding environmental testing for better stability assessment")

        return {
            "status": "success",
            "program_insights": {
                'genotype_focus': genotype_count,
                'phenotype_focus': phenotype_count,
                'environment_focus': environment_count
            },
            "recommendations": recommendations,
            "summary": "General program assessment completed based on integrated agent analyses.",
            "interpretation": f"Program analysis shows {'balanced approach' if abs(genotype_count - phenotype_count) < 5 else 'opportunity for optimization'} across breeding priorities."
        }

    def _calculate_agent_score(self, agent_insights: Dict, line_id: str, agent_type: str) -> float:
        """Calculate normalized score for a line from specific agent"""
        if agent_type == 'genotype':
            selection_scores = agent_insights.get('selection_scores', {})
            return selection_scores.get(line_id, {}).get('selection_score', 0.0)
        elif agent_type == 'phenotype':
            rankings = agent_insights.get('rankings', {})
            line_ranking = rankings.get(line_id, {})
            # Convert rank to score (lower rank = higher score)
            if line_ranking:
                rank = line_ranking.get('rank', len(rankings))
                return 1 / (1 + rank / len(rankings))  # Normalize to 0-1
            return 0.0
        elif agent_type == 'environment':
            stability_scores = agent_insights.get('stability_scores', {})
            return stability_scores.get(line_id, {}).get('stability_score', 0.0)
        return 0.0

    def _calculate_risk_factor(self, genotype_insights: Dict, phenotype_insights: Dict, environment_insights: Dict, line_id: str) -> float:
        """Calculate overall risk factor for a line"""
        risk_factors = []

        # Genetic risk - low diversity or high relatedness
        if 'diversity_analysis' in genotype_insights:
            diversity_score = genotype_insights['diversity_analysis'].get(line_id, {}).get('diversity_score', 0.5)
            genetic_risk = 1 - diversity_score  # Lower diversity = higher risk
            risk_factors.append(genetic_risk)

        # Phenotypic risk - poor performance or high variability
        if 'rankings' in phenotype_insights:
            line_ranking = phenotype_insights['rankings'].get(line_id, {})
            if line_ranking:
                rank = line_ranking.get('rank', 1000)
                phenotypic_risk = rank / 1000  # Higher rank = higher risk
                risk_factors.append(phenotypic_risk)

        # Environmental risk - poor stability
        if 'stability_scores' in environment_insights:
            stability_score = environment_insights['stability_scores'].get(line_id, {}).get('stability_score', 0.5)
            environmental_risk = 1 - stability_score  # Lower stability = higher risk
            risk_factors.append(environmental_risk)

        return np.mean(risk_factors) if risk_factors else 0.5

    def _make_single_advancement_decision(self, score: float, risk: float) -> str:
        """Make advancement decision for a single line"""
        if score > 0.8 and risk < 0.3:
            return "definitely_advance"
        elif score > 0.6 and risk < 0.5:
            return "likely_advance"
        elif score > 0.4:
            return "consider_advance"
        elif risk > 0.7:
            return "high_risk_do_not_advance"
        else:
            return "do_not_advance"

    def _get_all_candidate_lines(self) -> List[str]:
        """Get all unique candidate lines from all agents"""
        all_lines = set()
        for agent_type in ['genotype', 'phenotype', 'environment']:
            if agent_type in self.agent_analyses:
                insights = self.agent_analyses[agent_type]
                for key in ['recommended_lines', 'top_performers', 'most_stable_lines']:
                    if key in insights:
                        all_lines.update(insights[key])
        return list(all_lines)

    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize risk level"""
        if risk_score > 0.7:
            return "High Risk"
        elif risk_score > 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"

    def _categorize_suitability(self, score: float) -> str:
        """Categorize suitability for breeding activity"""
        if score > 0.8:
            return "Highly Suitable"
        elif score > 0.6:
            return "Suitable"
        elif score > 0.4:
            return "Moderately Suitable"
        else:
            return "Not Recommended"

    def _get_mitigation_strategies(self, genetic_risk: float, phenotypic_risk: float, environmental_risk: float) -> List[str]:
        """Get mitigation strategies for high-risk factors"""
        strategies = []

        if genetic_risk > 0.6:
            strategies.append("Increase marker coverage for better genetic characterization")
        if phenotypic_risk > 0.6:
            strategies.append("Expand multi-location testing to improve performance assessment")
        if environmental_risk > 0.6:
            strategies.append("Include stress tolerance evaluations in breeding pipeline")

        return strategies if strategies else ["Continue standard evaluation procedures"]

    def _generate_advancement_interpretation(self, decisions: Dict, advanced_lines: List[str]) -> str:
        """Generate interpretation of advancement decisions"""
        if not advanced_lines:
            return "No lines meet advancement criteria based on current analysis."

        avg_score = np.mean([decisions[line]['adjusted_score'] for line in advanced_lines])
        return f"Advanced {len(advanced_lines)} lines with average integrated score of {avg_score:.3f}. " \
               f"Decision criteria balanced genetic potential, phenotypic performance, and environmental stability."

    def _generate_prioritization_interpretation(self, prioritization_results: Dict) -> str:
        """Generate interpretation of prioritization results"""
        return f"Prioritization analysis identified optimal lines for {len(prioritization_results)} breeding objectives. " \
               f"Consider allocating resources based on program priorities and market demands."

    def _generate_risk_interpretation(self, risk_assessments: Dict) -> str:
        """Generate interpretation of risk assessment"""
        risk_levels = [risk['risk_level'] for risk in risk_assessments.values()]
        high_risk_count = risk_levels.count("High Risk")

        return f"Risk assessment identified {high_risk_count} high-risk lines out of {len(risk_assessments)}. " \
               f"{'Careful evaluation recommended' if high_risk_count > len(risk_assessments) * 0.2 else 'Risk profile appears manageable'} for the breeding program."
