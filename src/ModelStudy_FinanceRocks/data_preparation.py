#!/usr/bin/python3

"""
Data Preparation Module

This module handles all data preprocessing and transformation steps for the
revenue modeling pipeline.

Author: Sharat Sharma
Date: Jan-2026
"""

import os
import logging
import pandas as pd
from typing import Tuple
from .ModelStudy_FR import setup_temp_plot_directory, PreProcessor  # type: ignore

pd.set_option('display.max_columns', None)


class DataPreparationPipeline:
    """
    Handles data preprocessing and transformation for revenue modeling.
    
    This class orchestrates the complete data preparation pipeline including:
    - Loading and treating predictor columns
    - One-hot encoding of nominal variables
    - Ordinal encoding with predefined mappings
    - Column name cleaning and standardization
    - Log transformations of predictors and outcome variables
    
    Attributes:
        file_path (str): Path to the data directory
        file_name (str): Name of the data file
        plot_dir (str): Directory for saving plots
        logger (logging.Logger): Logger instance for tracking operations
        preprocessor (PreProcessor): Instance of PreProcessor for data operations
        model_predictors_df (pd.DataFrame): DataFrame with treated predictors
        log_transformed_predictors_df (pd.DataFrame): DataFrame with log-transformed predictors
        log_transformed_predictors_and_outcome_df (pd.DataFrame): Final dataset with predictors and outcome
    """
    
    def __init__(
        self, 
        file_path: str = None, 
        file_name: str = "customer_data.parquet",
        plot_dir: str = None,
        logger: logging.Logger = None
    ):
        """
        Initialize the data preparation pipeline.
        
        Parameters
        ----------
        file_path : str, optional
            Path to data directory. If None, uses setup_temp_plot_directory()
        file_name : str, default="customer_data.parquet"
            Name of the data file to load
        plot_dir : str, optional
            Directory for plots. If None, uses setup_temp_plot_directory()
        logger : logging.Logger, optional
            Logger instance. If None, creates a new logger
        """
        # Setup paths
        if file_path is None or plot_dir is None:
            default_file_path, default_plot_dir = setup_temp_plot_directory()
            self.file_path = file_path or default_file_path
            self.plot_dir = plot_dir or default_plot_dir
        else:
            self.file_path = file_path
            self.plot_dir = plot_dir
            
        self.file_name = file_name
        
        # Setup logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        
        # Initialize PreProcessor
        self.preprocessor = PreProcessor(
            self.file_path, self.file_name, self.plot_dir
        )
        
        # Initialize data containers
        self.model_predictors_df = None
        self.log_transformed_predictors_df = None
        self.log_transformed_predictors_and_outcome_df = None
    
    def load_and_treat_data(self) -> pd.DataFrame:
        """
        Load and apply initial treatment to predictor columns.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with treated predictor columns
        """
        self.logger.info("Loading and treating predictor columns...")
        self.model_predictors_df = self.preprocessor.treat_predictor_columns()
        self.logger.info("Initial dataset loaded and predictor columns treated.")
        return self.model_predictors_df
    
    def apply_one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply one-hot encoding to nominal predictor variables.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with nominal variables
            
        Returns
        -------
        pd.DataFrame
            DataFrame with one-hot encoded nominal variables
        """
        self.logger.info("Applying one-hot encoding to nominal predictors...")
        onehot_encoded_df = self.preprocessor.one_hot_encode_nominal_predictors(df)
        self.logger.info("One-hot encoding completed. New columns:")
        self.logger.info(onehot_encoded_df.columns.tolist())
        return onehot_encoded_df
    
    def apply_ordinal_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply ordinal encoding using predefined mappings.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with ordinal variables
            
        Returns
        -------
        pd.DataFrame
            DataFrame with ordinally encoded variables
        """
        self.logger.info("Applying ordinal encoding using predefined mappings...")
        label_encoded_df = self.preprocessor.map_ordinal_predictors(df)
        self.logger.info("Ordinal encoding completed. New columns:")
        self.logger.info(label_encoded_df.columns.tolist())
        self.logger.info("\n\n")
        
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            self.logger.info(label_encoded_df.dtypes)
        
        return label_encoded_df
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize column names.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with columns to clean
            
        Returns
        -------
        pd.DataFrame
            DataFrame with cleaned column names
        """
        self.logger.info("Cleaning and renaming predictor column names...")
        
        # Clean column names
        cleaned_cols = self.preprocessor.treat_predictor_column_names(
            df.columns.tolist()
        )
        df.columns = cleaned_cols
        
        # Apply tol_2 remapping
        df.rename(
            columns=self.preprocessor.tol_2_remaps(),
            inplace=True
        )
        
        self.logger.info("Predictor column names cleaned and renamed.")
        self.logger.info(df.columns.tolist())
        
        return df
    
    def apply_log_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log transformations to selected predictor variables.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with variables to transform
            
        Returns
        -------
        pd.DataFrame
            DataFrame with log-transformed predictors
        """
        self.logger.info("\n\n")
        self.logger.info("Applying log transformation to variables...\n")
        
        log_transformed_df = self.preprocessor.log_transform_df(df)
        self.log_transformed_predictors_df = log_transformed_df
        
        return log_transformed_df
    
    def transform_outcome_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log transformation to the outcome variable.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with outcome variable
            
        Returns
        -------
        pd.DataFrame
            DataFrame with log-transformed outcome variable
        """
        self.logger.info("Applying log transformation to outcome variable...")
        
        log_transformed_outcome_df = self.preprocessor.log_transform_outcome(df)
        self.log_transformed_predictors_and_outcome_df = log_transformed_outcome_df
        
        self.logger.info("\n\n")
        
        return log_transformed_outcome_df
    
    def prepare_modeling_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the complete data preparation pipeline.
        
        This method runs all preprocessing steps in sequence:
        1. Load and treat data
        2. One-hot encoding
        3. Ordinal encoding
        4. Column name cleaning
        5. Log transformations
        6. Outcome variable transformation
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            - log_transformed_predictors_df: Predictors with log transformations
            - log_transformed_predictors_and_outcome_df: Full dataset with outcome
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING DATA PREPARATION PIPELINE")
        self.logger.info("=" * 80)
        
        # Step 1: Load and treat data
        df = self.load_and_treat_data()
        
        # Step 2: One-hot encoding
        df = self.apply_one_hot_encoding(df)
        
        # Step 3: Ordinal encoding
        df = self.apply_ordinal_encoding(df)
        
        # Step 4: Clean column names
        df = self.clean_column_names(df)
        
        # Step 5: Log transformations
        df = self.apply_log_transformations(df)
        
        # Step 6: Transform outcome variable
        df_with_outcome = self.transform_outcome_variable(df)
        
        self.logger.info("=" * 80)
        self.logger.info("DATA PREPARATION PIPELINE COMPLETED")
        self.logger.info("=" * 80)
        
        return self.log_transformed_predictors_df, self.log_transformed_predictors_and_outcome_df
    
    def get_preprocessor(self) -> PreProcessor:
        """
        Get the underlying PreProcessor instance.
        
        Returns
        -------
        PreProcessor
            The PreProcessor instance used for data operations
        """
        return self.preprocessor
