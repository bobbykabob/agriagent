import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from src.agents.base_agent import BaseAgent
from src.config.settings import config
from src.utils.logger import logger

class GenotypeAgent(BaseAgent):
    """Agent for analyzing genotype data, SNP markers, haplotype diversity, and kinship matrices"""

    def __init__(self):
        super().__init__("Genotype Agent", config.GENOTYPE_MODEL)
        self.genetic_diversity_scores = {}
        self.kinship_matrix = None

    def analyze(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze genotype data based on query"""
        genotype_data = self.get_relevant_data('genotype')

        if genotype_data.empty:
            return {
                "status": "no_data",
                "message": "No genotype data available for analysis",
                "recommendations": []
            }

        context = context or {}
        analysis_type = context.get('analysis_type', 'general')

        if analysis_type == 'diversity':
            return self._analyze_genetic_diversity(genotype_data, query)
        elif analysis_type == 'kinship':
            return self._analyze_kinship(genotype_data, query)
        elif analysis_type == 'selection':
            return self._analyze_genetic_selection(genotype_data, query, context)
        else:
            return self._general_genotype_analysis(genotype_data, query)

    def _analyze_genetic_diversity(self, genotype_data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Analyze genetic diversity and haplotype patterns"""
        logger.info("Analyzing genetic diversity...")

        # Calculate genetic diversity metrics
        diversity_scores = {}

        # For each line, calculate diversity based on available genetic markers
        for idx, row in genotype_data.iterrows():
            line_id = row.get('entry', f'line_{idx}')

            # Calculate diversity score based on marker variability
            # In a real scenario, this would analyze actual SNP markers
            numeric_markers = row.select_dtypes(include=[np.number]).dropna()

            if len(numeric_markers) > 0:
                # Simple diversity metric - coefficient of variation across markers
                diversity_score = numeric_markers.std() / (numeric_markers.mean() + 1e-8)
            else:
                diversity_score = 0.0

            diversity_scores[line_id] = {
                'diversity_score': diversity_score,
                'marker_count': len(numeric_markers),
                'interpretation': self._interpret_diversity_score(diversity_score)
            }

        self.genetic_diversity_scores = diversity_scores

        # Generate top diverse lines recommendation
        sorted_lines = sorted(diversity_scores.items(), key=lambda x: x[1]['diversity_score'], reverse=True)
        top_lines = [line_id for line_id, _ in sorted_lines[:10]]

        return {
            "status": "success",
            "diversity_analysis": diversity_scores,
            "top_diverse_lines": top_lines,
            "summary": f"Analyzed {len(diversity_scores)} lines for genetic diversity. Top diverse lines: {', '.join(top_lines[:5])}",
            "interpretation": self._generate_diversity_interpretation(diversity_scores)
        }

    def _analyze_kinship(self, genotype_data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Analyze kinship relationships between lines"""
        logger.info("Analyzing kinship relationships...")

        if len(genotype_data) < 2:
            return {
                "status": "insufficient_data",
                "message": "Need at least 2 lines to calculate kinship",
                "kinship_matrix": None
            }

        # Calculate simple kinship matrix based on marker similarity
        # In a real scenario, this would use proper kinship algorithms
        numeric_data = genotype_data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            return {
                "status": "no_numeric_markers",
                "message": "No numeric genetic markers available for kinship analysis",
                "kinship_matrix": None
            }

        # Calculate similarity matrix
        # For demonstration, we'll use cosine similarity
        similarity_matrix = cosine_similarity(numeric_data.fillna(0))

        # Convert to kinship-like matrix (higher values = more related)
        kinship_matrix = similarity_matrix

        self.kinship_matrix = pd.DataFrame(
            kinship_matrix,
            index=genotype_data['entry'].values,
            columns=genotype_data['entry'].values
        )

        # Find most related pairs
        related_pairs = []
        for i in range(len(kinship_matrix)):
            for j in range(i+1, len(kinship_matrix)):
                if kinship_matrix[i, j] > 0.8:  # High kinship threshold
                    related_pairs.append({
                        'line1': genotype_data.iloc[i]['entry'],
                        'line2': genotype_data.iloc[j]['entry'],
                        'kinship_score': kinship_matrix[i, j]
                    })

        related_pairs.sort(key=lambda x: x['kinship_score'], reverse=True)

        return {
            "status": "success",
            "kinship_matrix": self.kinship_matrix.to_dict(),
            "related_pairs": related_pairs[:10],  # Top 10 most related pairs
            "summary": f"Calculated kinship for {len(genotype_data)} lines. Found {len(related_pairs)} highly related pairs (>0.8 similarity).",
            "interpretation": self._generate_kinship_interpretation(related_pairs)
        }

    def _analyze_genetic_selection(self, genotype_data: pd.DataFrame, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze lines for genetic advancement selection"""
        logger.info("Analyzing genetic selection criteria...")

        target_traits = context.get('target_traits', config.TARGET_TRAITS)

        selection_scores = {}

        for idx, row in genotype_data.iterrows():
            line_id = row.get('entry', f'line_{idx}')

            # Calculate selection score based on genetic markers
            # This would normally be based on marker-trait associations
            numeric_markers = row.select_dtypes(include=[np.number]).dropna()

            if len(numeric_markers) > 0:
                # Simple selection score - favor diversity and marker presence
                diversity_component = numeric_markers.std() / (numeric_markers.mean() + 1e-8)
                completeness_component = len(numeric_markers) / len(genotype_data.columns)

                selection_score = (diversity_component * 0.6) + (completeness_component * 0.4)
            else:
                selection_score = 0.0

            selection_scores[line_id] = {
                'selection_score': selection_score,
                'diversity_component': diversity_component if 'diversity_component' in locals() else 0,
                'completeness_component': completeness_component if 'completeness_component' in locals() else 0,
                'recommendation': self._get_selection_recommendation(selection_score)
            }

        # Sort by selection score
        sorted_lines = sorted(selection_scores.items(), key=lambda x: x[1]['selection_score'], reverse=True)
        top_lines = [line_id for line_id, _ in sorted_lines[:int(len(sorted_lines) * 0.1)]]  # Top 10%

        return {
            "status": "success",
            "selection_scores": selection_scores,
            "recommended_lines": top_lines,
            "summary": f"Evaluated {len(selection_scores)} lines for genetic selection. Recommended {len(top_lines)} lines for advancement.",
            "interpretation": self._generate_selection_interpretation(selection_scores, target_traits)
        }

    def _general_genotype_analysis(self, genotype_data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """General genotype analysis"""
        logger.info("Performing general genotype analysis...")

        summary = self.format_data_summary(genotype_data)

        return {
            "status": "success",
            "data_summary": summary,
            "total_lines": len(genotype_data),
            "marker_types": genotype_data.dtypes.value_counts().to_dict(),
            "interpretation": f"General analysis of {len(genotype_data)} genetic lines completed. {summary}"
        }

    def _interpret_diversity_score(self, score: float) -> str:
        """Interpret genetic diversity score"""
        if score > 0.5:
            return "High genetic diversity - good candidate for maintaining population diversity"
        elif score > 0.2:
            return "Moderate genetic diversity - balanced contribution to breeding program"
        else:
            return "Low genetic diversity - may be too similar to existing germplasm"

    def _generate_diversity_interpretation(self, diversity_scores: Dict) -> str:
        """Generate interpretation of diversity analysis"""
        if not diversity_scores:
            return "No diversity data to interpret."

        scores = [info['diversity_score'] for info in diversity_scores.values()]
        mean_score = np.mean(scores)
        max_score = np.max(scores)

        return f"Diversity analysis shows average genetic diversity score of {mean_score:.3f} across all lines. " \
               f"The highest diversity score is {max_score:.3f}. " \
               f"{'High diversity detected - good for breeding program.' if mean_score > 0.3 else 'Consider increasing genetic diversity in the breeding population.'}"

    def _generate_kinship_interpretation(self, related_pairs: List[Dict]) -> str:
        """Generate interpretation of kinship analysis"""
        if not related_pairs:
            return "No highly related pairs found in the current population."

        return f"Found {len(related_pairs)} pairs of highly related lines (similarity > 0.8). " \
               f"The most related pair has a kinship score of {related_pairs[0]['kinship_score']:.3f}. " \
               f"This indicates {'good genetic relationships for breeding' if len(related_pairs) < 10 else 'potential redundancy in the breeding population'}."

    def _get_selection_recommendation(self, score: float) -> str:
        """Get selection recommendation based on score"""
        if score > 0.7:
            return "Strongly recommend for advancement"
        elif score > 0.5:
            return "Recommend for advancement"
        elif score > 0.3:
            return "Consider for advancement"
        else:
            return "Do not recommend for advancement"

    def _generate_selection_interpretation(self, selection_scores: Dict, target_traits: List[str]) -> str:
        """Generate interpretation of selection analysis"""
        if not selection_scores:
            return "No selection data to interpret."

        scores = [info['selection_score'] for info in selection_scores.values()]
        mean_score = np.mean(scores)

        return f"Selection analysis shows average score of {mean_score:.3f}. " \
               f"Lines with scores above 0.5 show particular promise for genetic advancement. " \
               f"Consider focusing on {'diversity and completeness' if mean_score > 0.4 else 'increasing marker coverage'} in future evaluations."
