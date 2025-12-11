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
        if not self.file_path:
            logger.warning("No data file path provided. Skipping load.")
            return None

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
        logger.info("Preprocessing data...")

        # 1. Load Genotype Data (Prefer specific file)
        if os.path.exists(config.GENOTYPE_FILE_PATH):
            logger.info(f"Loading genotype data from {config.GENOTYPE_FILE_PATH}")
            try:
                genotype_df = pd.read_csv(config.GENOTYPE_FILE_PATH, nrows=100)
                
                # Transpose if it's in marker-major format (rows=markers, cols=lines)
                # Check if columns look like plant IDs (e.g., C002, C003) and rows are markers
                if 'rs#' in genotype_df.columns or 'alleles' in genotype_df.columns:
                    # Set marker ID as index
                    if 'rs#' in genotype_df.columns:
                        genotype_df = genotype_df.set_index('rs#')
                    
                    # Drop metadata columns to keep only plant columns
                    metadata_cols = ['alleles', 'chrom', 'pos', 'strand', 'assembly#', 'center', 'protLSID', 'assayLSID', 'panelLSID', 'QCcode']
                    cols_to_drop = [c for c in metadata_cols if c in genotype_df.columns]
                    genotype_df = genotype_df.drop(columns=cols_to_drop)
                    
                    # Transpose: Now Rows=Lines, Cols=Markers
                    genotype_df = genotype_df.transpose()
                    
                    # Convert alleles to numeric if needed (simple encoding for now)
                    # For Agent analysis, keeping as strings might be okay if Agent handles it, 
                    # but generally numeric is better. Let's do a simple factorize for now if object type.
                    for col in genotype_df.columns:
                         if genotype_df[col].dtype == 'object':
                             genotype_df[col] = pd.Categorical(genotype_df[col]).codes

                self.processed_data['genotype'] = genotype_df
                logger.info(f"Processed specific genotype data: {len(genotype_df)} lines, {len(genotype_df.columns)} markers")
            except Exception as e:
                logger.error(f"Failed to load specific genotype file: {e}")
                self.processed_data['genotype'] = pd.DataFrame()
        else:
            # Fallback to splitting raw_data
            if self.raw_data is None and self.file_path: self.load_data()
            if self.raw_data is not None:
                genotype_cols, _, _ = self._categorize_columns()
                available_ids = self._get_available_ids()
                self.processed_data['genotype'] = self._process_genotype_data(genotype_cols, available_ids)
            else:
                self.processed_data['genotype'] = pd.DataFrame()

        # 2. Load Phenotype Data (Prefer specific file)
        if os.path.exists(config.PHENOTYPE_FILE_PATH):
            logger.info(f"Loading phenotype data from {config.PHENOTYPE_FILE_PATH}")
            try:
                phenotype_df = pd.read_excel(config.PHENOTYPE_FILE_PATH)
                # Assume standard format: Rows=Lines, Cols=Traits
                # If there are ID columns, we might want to set them as index or keep them
                
                # Basic cleaning
                phenotype_df = self._clean_phenotype_data(phenotype_df)
                
                self.processed_data['phenotype'] = phenotype_df
                logger.info(f"Processed specific phenotype data: {len(phenotype_df)} lines, {len(phenotype_df.columns)} traits")
            except Exception as e:
                logger.error(f"Failed to load specific phenotype file: {e}")
                self.processed_data['phenotype'] = pd.DataFrame()
        else:
             # Fallback
            if self.raw_data is None and self.file_path: self.load_data()
            if self.raw_data is not None:
                _, phenotype_cols, _ = self._categorize_columns()
                available_ids = self._get_available_ids()
                self.processed_data['phenotype'] = self._process_phenotype_data(phenotype_cols, available_ids)
            else:
                self.processed_data['phenotype'] = pd.DataFrame()

        # 3. Environment Data (From main file for now, unless phenotype has it)
        # We'll use the original logic for environment for now as no specific file was provided
        if self.raw_data is None and self.file_path: self.load_data()
        
        if self.raw_data is not None:
            _, _, environment_cols = self._categorize_columns()
            available_ids = self._get_available_ids()
            self.processed_data['environment'] = self._process_environment_data(environment_cols, available_ids)
        else:
            # Try to get environment data from phenotype file if available
            if os.path.exists(config.PHENOTYPE_FILE_PATH):
                try:
                   phenotype_df = pd.read_excel(config.PHENOTYPE_FILE_PATH)
                   environment_patterns = ['location', 'loc', 'year', 'soil', 'climate', 'weather', 'ndvi', 'rep']
                   environment_cols = []
                   for col in phenotype_df.columns:
                       col_lower = col.lower()
                       if any(pattern in col_lower for pattern in environment_patterns):
                           environment_cols.append(col)
                   
                   if environment_cols:
                       # Also include ID columns
                       id_cols = ['entry', 'plot', 'name', 'Entry', 'Plot', 'Name']
                       available_ids = [c for c in id_cols if c in phenotype_df.columns]
                       self.processed_data['environment'] = phenotype_df[available_ids + environment_cols].copy()
                       logger.info(f"Extracted environment data from phenotype file: {len(self.processed_data['environment'])} records")
                   else:
                       self.processed_data['environment'] = pd.DataFrame()
                except Exception as e:
                   logger.error(f"Failed to extract environment data from phenotype file: {e}")
                   self.processed_data['environment'] = pd.DataFrame()
            else:
                self.processed_data['environment'] = pd.DataFrame()

        logger.info("Data preprocessing completed")
        return self.processed_data

    def _get_available_ids(self) -> List[str]:
        """Get available identifier columns in raw_data"""
        if self.raw_data is None: return []
        id_cols = ['entry', 'plot', 'name', 'Entry', 'Plot', 'Name']
        return [col for col in id_cols if col in self.raw_data.columns]

    def _clean_phenotype_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean phenotype dataframe"""
        # Drop empty rows/cols
        df = df.dropna(how='all')
        
        # Normalize numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            # Simple outlier clipping
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower, upper)
            
        return df


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
