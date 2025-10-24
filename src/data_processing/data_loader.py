import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from src.config.settings import config
from src.utils.logger import logger

class DataLoader:
    """Load and preprocess breeding data from Excel file"""

    def __init__(self, file_path: str = None):
        self.file_path = file_path or config.DATA_FILE_PATH
        self.raw_data = None
        self.processed_data = {}

    def load_data(self) -> pd.DataFrame:
        """Load raw data from Excel file"""
        try:
            logger.info(f"Loading data from {self.file_path}")
            self.raw_data = pd.read_excel(self.file_path)
            logger.info(f"Loaded {len(self.raw_data)} rows and {len(self.raw_data.columns)} columns")
            return self.raw_data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def preprocess_data(self) -> Dict[str, pd.DataFrame]:
        """Preprocess and organize data into different categories"""
        if self.raw_data is None:
            self.load_data()

        logger.info("Preprocessing data...")

        # Identify data types based on column names and content
        genotype_cols, phenotype_cols, environment_cols = self._categorize_columns()

        # Extract and clean different data types
        # Check available identifier columns
        id_cols = ['entry', 'plot', 'name']
        available_ids = [col for col in id_cols if col in self.raw_data.columns]

        self.processed_data['genotype'] = self._process_genotype_data(genotype_cols, available_ids)
        self.processed_data['phenotype'] = self._process_phenotype_data(phenotype_cols, available_ids)
        self.processed_data['environment'] = self._process_environment_data(environment_cols, available_ids)

        logger.info("Data preprocessing completed")
        return self.processed_data

    def _categorize_columns(self) -> Tuple[List[str], List[str], List[str]]:
        """Categorize columns into genotype, phenotype, and environment data"""
        columns = self.raw_data.columns.tolist()

        genotype_cols = []
        phenotype_cols = []
        environment_cols = []

        # Common patterns in column names
        genotype_patterns = ['snp', 'marker', 'allele', 'genotype', 'haplotype', 'kinship']
        phenotype_patterns = ['yield', 'height', 'density', 'maturity', 'oil', 'protein', 'lodging', 'score']
        environment_patterns = ['location', 'loc', 'year', 'soil', 'climate', 'weather', 'ndvi', 'rep']

        for col in columns:
            col_lower = col.lower()
            # Skip identifier columns
            if col_lower in ['entry', 'plot', 'name', 'notes']:
                continue
            if any(pattern in col_lower for pattern in genotype_patterns):
                genotype_cols.append(col)
            elif any(pattern in col_lower for pattern in phenotype_patterns):
                phenotype_cols.append(col)
            elif any(pattern in col_lower for pattern in environment_patterns):
                environment_cols.append(col)

        logger.info(f"Categorized columns - Genotype: {len(genotype_cols)}, Phenotype: {len(phenotype_cols)}, Environment: {len(environment_cols)}")
        return genotype_cols, phenotype_cols, environment_cols

    def _process_genotype_data(self, genotype_cols: List[str], available_ids: List[str]) -> pd.DataFrame:
        """Process genotype-related data"""
        if not genotype_cols:
            logger.warning("No genotype columns found")
            return pd.DataFrame()

        # Extract genotype data
        genotype_data = self.raw_data[available_ids + genotype_cols].copy()

        # Clean data - handle missing values, normalize formats
        genotype_data = genotype_data.dropna(subset=genotype_cols, how='all')

        # Convert categorical genotype data to numerical where appropriate
        for col in genotype_cols:
            if genotype_data[col].dtype == 'object':
                # Check if it's binary or categorical
                unique_vals = genotype_data[col].dropna().unique()
                if len(unique_vals) <= 10:  # Likely categorical
                    genotype_data[col] = pd.Categorical(genotype_data[col]).codes

        logger.info(f"Processed genotype data: {len(genotype_data)} lines, {len(genotype_cols)} markers")
        return genotype_data

    def _process_phenotype_data(self, phenotype_cols: List[str], available_ids: List[str]) -> pd.DataFrame:
        """Process phenotype-related data"""
        if not phenotype_cols:
            logger.warning("No phenotype columns found")
            return pd.DataFrame()

        # Extract phenotype data
        phenotype_data = self.raw_data[available_ids + phenotype_cols].copy()

        # Clean data - handle missing values, outliers
        phenotype_data = phenotype_data.dropna(subset=phenotype_cols, how='all')

        # Normalize numerical data
        for col in phenotype_cols:
            if pd.api.types.is_numeric_dtype(phenotype_data[col]):
                # Handle outliers using IQR method
                Q1 = phenotype_data[col].quantile(0.25)
                Q3 = phenotype_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Cap outliers
                phenotype_data[col] = np.clip(phenotype_data[col], lower_bound, upper_bound)

                # Standardize if needed
                if phenotype_data[col].std() > 0:
                    phenotype_data[col] = (phenotype_data[col] - phenotype_data[col].mean()) / phenotype_data[col].std()

        logger.info(f"Processed phenotype data: {len(phenotype_data)} lines, {len(phenotype_cols)} traits")
        return phenotype_data

    def _process_environment_data(self, environment_cols: List[str], available_ids: List[str]) -> pd.DataFrame:
        """Process environment-related data"""
        if not environment_cols:
            logger.warning("No environment columns found")
            return pd.DataFrame()

        # Extract environment data
        environment_data = self.raw_data[available_ids + environment_cols].copy()

        # Clean data
        environment_data = environment_data.dropna(subset=environment_cols, how='all')

        logger.info(f"Processed environment data: {len(environment_data)} records, {len(environment_cols)} environmental factors")
        return environment_data

    def get_summary_statistics(self) -> Dict[str, Dict]:
        """Get summary statistics for all data types"""
        if not self.processed_data:
            self.preprocess_data()

        stats = {}

        for data_type, df in self.processed_data.items():
            if df.empty:
                stats[data_type] = {"error": f"No {data_type} data available"}
                continue

            stats[data_type] = {
                "num_lines": len(df),
                "num_features": len(df.columns) - 2,  # Exclude line_id and entry
                "missing_data_percentage": df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100,
                "data_types": df.dtypes.value_counts().to_dict()
            }

        return stats

    def save_processed_data(self, output_dir: str = None):
        """Save processed data to files"""
        if not self.processed_data:
            self.preprocess_data()

        output_dir = output_dir or config.PROCESSED_DATA_DIR
        os.makedirs(output_dir, exist_ok=True)

        for data_type, df in self.processed_data.items():
            if not df.empty:
                file_path = os.path.join(output_dir, f"{data_type}_data.csv")
                df.to_csv(file_path, index=False)
                logger.info(f"Saved {data_type} data to {file_path}")

    def get_data_for_analysis(self) -> Dict[str, pd.DataFrame]:
        """Get processed data ready for agent analysis"""
        if not self.processed_data:
            self.preprocess_data()

        return self.processed_data
