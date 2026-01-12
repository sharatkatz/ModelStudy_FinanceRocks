#!/usr/bin/python3

"""
Model Training Module

This module handles OLS model training, feature selection, and model fitting.

Author: Sharat Sharma
Date: Jan-2026
"""

import logging
import pandas as pd
import numpy as np
import statsmodels.api as sm  # type: ignore
from typing import List, Tuple
from data_preparation import DataPreparationPipeline


class OLSModelTrainer:
    """
    Handles OLS model training, feature selection, and iterative refinement.
    
    This class manages the complete model training pipeline including:
    - Feature selection based on correlation and significance
    - Iterative model refinement
    - Discount column calculation
    - Model fitting with statsmodels OLS
    
    Attributes:
        data_prep (DataPreparationPipeline): Data preparation pipeline instance
        model_type (str): Type of model ("Additive" or "Multiplicative")
        endog (str): Name of endogenous (dependent) variable
        logger (logging.Logger): Logger instance
        preprocessor: PreProcessor instance from data_prep
        additive_type (bool): Flag for additive model
        multiplicative_type (bool): Flag for multiplicative model
        endog_glm (str): Logged endogenous variable name
        exog_glm (List[str]): List of exogenous variable names
        add_constant (bool): Whether to add constant term
        addon_usage_columns (List[str]): Add-on usage column names
        usage_metric_columns (List[str]): Usage metric column names
        customer_profile_columns (List[str]): Customer profile column names
        discount_column_name (str): Name of discount column
        discounts_offered (pd.DataFrame): Calculated discounts
        final_ols_results: Fitted OLS model results
        mod_dsn (pd.DataFrame): Model design matrix
        log_transformed_predictors_df (pd.DataFrame): Log-transformed predictors
        log_transformed_predictors_and_outcome_df (pd.DataFrame): Full dataset with outcome
    """
    
    def __init__(
        self,
        data_prep: DataPreparationPipeline,
        model_type: str = "Additive",
        endog: str = "total_revenue",
        logger: logging.Logger = None
    ):
        """
        Initialize the OLS model trainer.
        
        Parameters
        ----------
        data_prep : DataPreparationPipeline
            Data preparation pipeline instance
        model_type : str, default="Additive"
            Type of model ("Additive" or "Multiplicative")
        endog : str, default="total_revenue"
            Name of endogenous (dependent) variable
        logger : logging.Logger, optional
            Logger instance
        """
        self.data_prep = data_prep
        self.model_type = model_type
        self.endog = endog
        
        # Setup logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        
        # Get preprocessor and data from data_prep
        self.preprocessor = data_prep.get_preprocessor()
        self.log_transformed_predictors_df = data_prep.log_transformed_predictors_df
        self.log_transformed_predictors_and_outcome_df = data_prep.log_transformed_predictors_and_outcome_df
        
        # Initialize model flags
        self.additive_type = False
        self.multiplicative_type = False
        self.add_constant = False
        
        # Get column groups from preprocessor
        self.addon_usage_columns = self.preprocessor.get_addon_usage_columns()
        self.usage_metric_columns = self.preprocessor.get_usage_metrics_columns()
        self.customer_profile_columns = self.preprocessor.get_customer_profile_columns()
        
        # Initialize model attributes
        self.endog_glm = endog
        self.exog_glm = []
        self.discount_column_name = 'discounts_offered'
        self.discounts_offered = None
        self.final_ols_results = None
        self.mod_dsn = None
    
    def calculate_and_exclude_highly_correlated_vars(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float = 0.9
    ) -> List[str]:
        """
        Identify variables to exclude based on high correlation.
        
        Parameters
        ----------
        corr_matrix : pd.DataFrame
            Correlation matrix of variables
        threshold : float, default=0.9
            Correlation threshold for exclusion
            
        Returns
        -------
        List[str]
            Variables to exclude due to high correlation
        """
        to_exclude = set()
        cols = corr_matrix.columns.tolist()
        
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr_value = float(corr_matrix.iloc[i, j])
                if abs(corr_value) > threshold:
                    var1 = cols[i]
                    var2 = cols[j]
                    # Exclude var2 arbitrarily; could implement more sophisticated logic
                    to_exclude.add(var2)
                    self.logger.info(
                        f"Excluding '{var2}' due to high correlation ({corr_value:.2f}) with '{var1}'"
                    )
        
        return list(to_exclude)
    
    def exclude_insignificant_vars(
        self,
        model_results,
        alpha: float = 0.05
    ) -> List[str]:
        """
        Identify variables to exclude based on statistical insignificance.
        
        Parameters
        ----------
        model_results : statsmodels RegressionResultsWrapper
            Fitted model results
        alpha : float, default=0.05
            Significance level
            
        Returns
        -------
        List[str]
            Variables to exclude due to insignificance
        """
        to_exclude = []
        pvalues = model_results.pvalues
        
        for var, pval in pvalues.items():
            if pval > alpha:
                to_exclude.append(var)
                self.logger.info(
                    f"Excluding '{var}' due to insignificance (p-value: {pval:.4f})"
                )
        
        return to_exclude
    
    def build_feature_set(self) -> List[str]:
        """
        Build the initial feature set for modeling.
        
        Returns
        -------
        List[str]
            List of feature names to include in model
        """
        predictor_cols_superset = self.log_transformed_predictors_df.columns.tolist()
        
        # Build exclusion list
        exclude_cols = []
        
        # Exclude revenue after discounts columns
        for colname in self.preprocessor.get_revenue_after_discounts_columns():
            exclude_cols.append(f"{colname}_logged")
        
        # Exclude high-collinearity proxies
        exclude_cols += [
            'total_records_mean_logged',
            'total_SI_PI_vouchers_mean_logged',
        ]
        
        # Select relevant predictor columns based on naming patterns
        package_cols = [
            col for col in predictor_cols_superset if col.lower().startswith("package")
        ]
        company_type_cols = [
            col for col in predictor_cols_superset if col.lower().startswith("company_type_")
        ]
        tol_1_eng_cols = [
            col for col in predictor_cols_superset if col.startswith("tol_1_eng_")
        ]
        _class_cols = [
            col for col in predictor_cols_superset if col.lower().endswith("_class")
        ]
        logged_cols = [
            f"{col}_logged" for col in self.preprocessor.to_log_transform_columns()
            if col != self.endog_glm
        ]
        add_usage_cols = [
            col for col in predictor_cols_superset 
            if col.lower() in self.addon_usage_columns
        ]
        
        # Combine all feature groups
        include_cols = (package_cols + company_type_cols + tol_1_eng_cols + 
                       _class_cols + logged_cols + add_usage_cols)
        
        self.logger.info(f"Total number of exogenous variables considered: {len(include_cols)}")
        
        # Define exogenous variables by excluding specified columns
        exog_features = [col for col in include_cols if col not in exclude_cols]
        
        return exog_features
    
    def sum_discount_columns(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate total discounts from discount columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing discount columns
            
        Returns
        -------
        pd.Series
            Total discounts per row
        """
        return self.preprocessor.sum_discount_columns(df)
    
    def setup_OLS(self):
        """
        Configure and run the OLS modeling pipeline.
        
        This method:
        1. Sets up model type flags
        2. Builds feature set
        3. Excludes highly correlated variables
        4. Prepares design matrix
        5. Fits baseline OLS model
        6. Refines model by excluding insignificant variables
        7. Fits final OLS model
        
        Sets attributes:
        - self.final_ols_results: Final fitted model
        - self.mod_dsn: Model design matrix
        - self.exog_glm: Final feature list
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STARTING OLS MODEL TRAINING")
        self.logger.info("=" * 80)
        
        # Set model type flags
        self.additive_type = True
        
        if self.model_type is None:
            self.model_type = "Additive"
        
        if self.model_type.lower()[:3] == "add":
            self.additive_type = True
        
        if self.model_type.lower()[:3] == "mul":
            self.multiplicative_type = True
        
        # Update endogenous variable name to logged version
        self.endog_glm = f"{self.endog}_logged"
        
        # Build initial feature set
        self.exog_glm = self.build_feature_set()
        
        # Exclude highly correlated variables
        exclude_correlated_cols = self.calculate_and_exclude_highly_correlated_vars(
            self.log_transformed_predictors_and_outcome_df[self.exog_glm].corr(), 
            threshold=0.7
        )
        
        # Re-construct exog_glm after excluding highly correlated vars
        self.exog_glm = [col for col in self.exog_glm if col not in exclude_correlated_cols]
        
        self.logger.info(f"Final number of exogenous variables selected: {len(self.exog_glm)}")
        
        # Handle additive model
        if self.additive_type:
            self.logger.info("Additive model selected")
            self.add_constant = False
            self.logger.info("Not adding constant term to the model.")
            
            self.logger.info("Setting up and fitting OLS model...")
            self.logger.info(f"Endogenous variable (outcome): {self.endog_glm}")
            self.logger.info(f"Exogenous variables (predictors): {len(self.exog_glm)} features")
            
            # Prepare exogenous and endogenous DataFrames, handling infinite values
            self.logger.info("Preparing exogenous and endogenous DataFrames...")
            exog_glm_df = self.log_transformed_predictors_and_outcome_df[
                self.exog_glm
            ].replace([float('inf'), -float('inf')], float('nan'))
            
            # Calculate discounts for each customer
            self.discounts_offered = pd.DataFrame(
                self.sum_discount_columns(self.preprocessor.customer_data),
                columns=[self.discount_column_name]
            )
            exog_glm_df = pd.concat([exog_glm_df, self.discounts_offered], axis=1)
            
            endog_glm_df = self.log_transformed_predictors_and_outcome_df[
                self.endog_glm
            ].replace([float('inf'), -float('inf')], float('nan'))
            
            self.logger.info("Handling missing values by dropping rows with NaNs...")
            mod_dsn = pd.concat([exog_glm_df, endog_glm_df], axis=1).dropna()
            
            # Re-set discounts_offered after dropping NaNs
            self.discounts_offered = mod_dsn[[self.discount_column_name]]
            
            # Fit baseline OLS model
            self.logger.info("\n--- BASELINE OLS MODEL ---")
            self.logger.info("Fitting baseline OLS model...")
            glm_ols_baseline = sm.OLS(
                endog=mod_dsn.loc[:, self.endog_glm],
                exog=mod_dsn.loc[:, self.exog_glm],
            )
            
            ols_results = glm_ols_baseline.fit()
            self.logger.info("Baseline OLS model fitting completed.")
            self.logger.info(ols_results.summary())
            
            # Identify insignificant coefficients to exclude
            insignificant_coefs = self.exclude_insignificant_vars(ols_results, alpha=0.05)
            self.logger.info(f"Insignificant coefficients to exclude: {insignificant_coefs}")
            
            # Fit refined OLS model (excluding insignificant variables)
            self.logger.info("\n--- REFINED OLS MODEL ---")
            self.logger.info("Fitting refined OLS model excluding insignificant coefficients...")
            
            refined_features = [col for col in self.exog_glm if col not in insignificant_coefs]
            
            glm_ols_refined = sm.OLS(
                endog=mod_dsn.loc[:, self.endog_glm],
                exog=mod_dsn.loc[:, refined_features],
            )
            refined_ols_results = glm_ols_refined.fit()
            self.logger.info("Refined OLS model fitted.")
            self.logger.info(refined_ols_results.summary())
            
            # Store final results
            self.final_ols_results = refined_ols_results
            self.mod_dsn = mod_dsn
            
            # Update exog_glm to reflect final features used
            self.exog_glm = refined_features
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("OLS MODEL TRAINING COMPLETED")
            self.logger.info(f"Final model uses {len(self.exog_glm)} features")
            self.logger.info(f"R-squared: {self.final_ols_results.rsquared:.4f}")
            self.logger.info(f"Adjusted R-squared: {self.final_ols_results.rsquared_adj:.4f}")
            self.logger.info("=" * 80)
    
    def get_model_results(self):
        """Get the final OLS model results."""
        return self.final_ols_results
    
    def get_design_matrix(self) -> pd.DataFrame:
        """Get the model design matrix."""
        return self.mod_dsn
    
    def get_exog_features(self) -> List[str]:
        """Get the list of exogenous features used in final model."""
        return self.exog_glm
    
    def get_endog_feature(self) -> str:
        """Get the endogenous feature name."""
        return self.endog_glm
