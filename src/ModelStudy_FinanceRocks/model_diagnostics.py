#!/usr/bin/python3

"""
Model Diagnostics Module

This module handles comprehensive model diagnostics, plots, and statistical tests.

Author: Sharat Sharma
Date: Jan-2026
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from typing import Dict, Optional, Union, List
from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore
import statsmodels.api as sm  # type: ignore


class ModelDiagnostics:
    """
    Handles comprehensive model diagnostics and visualization.
    
    This class provides:
    - VIF (Variance Inflation Factor) calculation
    - Residual analysis
    - Heteroscedasticity tests
    - Normality tests
    - Diagnostic plots
    - Encoding reversal utilities
    
    Attributes:
        logger (logging.Logger): Logger instance
        plot_dir (str): Directory for saving plots
        preprocessor: PreProcessor instance for encoding operations
        reverse_mappings (Dict): Reverse ordinal encoding mappings
        vif_scores (pd.DataFrame): VIF scores for features
    """
    
    def __init__(
        self,
        preprocessor,
        plot_dir: str,
        logger: logging.Logger = None
    ):
        """
        Initialize the model diagnostics handler.
        
        Parameters
        ----------
        preprocessor : PreProcessor
            PreProcessor instance for encoding operations
        plot_dir : str
            Directory for saving diagnostic plots
        logger : logging.Logger, optional
            Logger instance
        """
        self.preprocessor = preprocessor
        self.plot_dir = plot_dir
        
        # Setup logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        
        # Initialize reverse mappings
        self.reverse_mappings = self.reverse_ordinal_variables_and_category()
        self.vif_scores = None
    
    def reverse_ordinal_variables_and_category(self) -> Dict[str, Dict[int, str]]:
        """
        Reverse ordinal encoding mappings.
        
        Returns
        -------
        Dict[str, Dict[int, str]]
            Reverse mappings from encoded integers to original labels
        """
        reverse_mappings = {}
        original_mappings = self.preprocessor.map_ordinal_variables_and_category_order()
        
        for var_name, mapping_dict in original_mappings.items():
            for var_level, code in mapping_dict.items():
                if not isinstance(code, int):
                    raise ValueError(
                        f"Expected integer codes in mapping for variable '{var_name}', but got {type(code)}"
                    )
                reverse_ordinal_mapping = {var_name: {int(v): k for k, v in mapping_dict.items()}}
                reverse_mappings.update(reverse_ordinal_mapping)
        
        return reverse_mappings
    
    def reverse_ordinal_predictors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse ordinal encoding back to original categorical values.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with ordinally encoded columns
            
        Returns
        -------
        pd.DataFrame
            DataFrame with reversed encoding
        """
        outdf = df.copy()
        
        for map_col in self.reverse_mappings.keys():
            if map_col not in outdf.columns:
                self.logger.debug(f"Column '{map_col}' not found in data. Skipping.")
                continue
            
            self.logger.debug(f"Reversing mapping of column: {map_col}")
            mapping = self.reverse_mappings[map_col]
            outdf[map_col] = outdf[map_col].map(mapping).astype('category')
        
        return outdf
    
    def undummify_company_type_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert one-hot encoded company_type columns back to single column.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with one-hot encoded company_type columns
            
        Returns
        -------
        pd.DataFrame
            DataFrame with single company_type column
        """
        outdf = df.copy()
        company_type_cols = [col for col in outdf.columns if col.startswith('company_type_')]
        
        if company_type_cols:
            outdf['company_type'] = outdf[company_type_cols].idxmax(axis='columns').str.replace('company_type_', '')
            outdf['company_type'] = outdf['company_type'].astype('category')
        
        return outdf
    
    def undummify_tol_1_eng_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert one-hot encoded tol_1_eng columns back to single column."""
        outdf = df.copy()
        tol_1_eng_cols = [col for col in outdf.columns if col.startswith('tol_1_eng_')]
        
        if tol_1_eng_cols:
            outdf['tol_1_eng'] = outdf[tol_1_eng_cols].idxmax(axis='columns').str.replace('tol_1_eng_', '')
            outdf['tol_1_eng'] = outdf['tol_1_eng'].astype('category')
        
        return outdf
    
    def undummify_package_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert one-hot encoded package columns back to single column."""
        outdf = df.copy()
        package_cols = [col for col in outdf.columns if col.startswith('package_')]
        
        if package_cols:
            outdf['package'] = outdf[package_cols].idxmax(axis='columns').str.replace('package_', '')
            outdf['package'] = outdf['package'].astype('category')
        
        return outdf
    
    def calculate_vif(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Variance Inflation Factor for features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
            
        Returns
        -------
        pd.DataFrame
            VIF scores for each feature
        """
        self.logger.info("Calculating VIF scores...")
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        
        self.vif_scores = vif_data.sort_values('VIF', ascending=False)
        
        self.logger.info("\nVARIANCE INFLATION FACTORS (VIF)")
        high_vif = self.vif_scores[self.vif_scores['VIF'] > 10]
        if not high_vif.empty:
            self.logger.warning("⚠️ High VIF detected (potential multicollinearity):")
            for _, row in high_vif.iterrows():
                self.logger.warning(f"  {row['Feature']}: VIF = {row['VIF']:.2f}")
        else:
            self.logger.info("✓ No high VIF values detected")
        
        return self.vif_scores
    
    def model_diagnostics(
        self,
        mod_dsn: pd.DataFrame,
        model_results,
        exog_features: List[str],
        endog_feature: str,
        discount_column: pd.DataFrame,
        discount_column_name: str,
        model_type: str = "OLS"
    ):
        """
        Run comprehensive model diagnostics.
        
        Parameters
        ----------
        mod_dsn : pd.DataFrame
            Model design matrix
        model_results : statsmodels results
            Fitted model results
        exog_features : List[str]
            List of exogenous feature names
        endog_feature : str
            Endogenous feature name
        discount_column : pd.DataFrame
            Discount column data
        discount_column_name : str
            Name of discount column
        model_type : str, default="OLS"
            Type of model
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("RUNNING MODEL DIAGNOSTICS")
        self.logger.info("=" * 80)
        
        if model_results is None:
            self.logger.error("Model results object is None. Cannot perform diagnostics.")
            return
        
        # Reverse ordinal encoding
        outdf = self.reverse_ordinal_predictors(mod_dsn)
        performance_cols = list(self.preprocessor.map_ordinal_variables_and_category_order().keys())
        
        # Add discount column
        if outdf.shape[0] != discount_column.shape[0]:
            self.logger.error("Mismatch in number of rows between model design DataFrame and discounts offered data.")
            return
        
        outdf[discount_column_name] = discount_column
        
        # Calculate residuals
        residuals = None
        if hasattr(model_results, 'resid'):
            residuals = model_results.resid
        else:
            # For LASSO/RIDGE, calculate residuals manually
            self.logger.info("Calculating residuals manually for LASSO/RIDGE model...")
            endog_array = mod_dsn.loc[:, endog_feature].values
            exog_array = mod_dsn.loc[:, exog_features].values
            predictions = model_results.predict(exog_array)
            residuals = endog_array - predictions
        
        # Generate performance plots for ordinal variables
        for p_col in performance_cols:
            if p_col not in outdf.columns:
                self.logger.warning(f"Performance column '{p_col}' not found. Skipping.")
                continue
            
            self.performance_plots(
                outdf, model_results, p_col, resid=residuals, model_type=model_type
            )
        
        # Generate scatter plots for all predictors
        for predictor_col in exog_features:
            if predictor_col not in outdf.columns:
                self.logger.warning(f"Predictor column '{predictor_col}' not found. Skipping.")
                continue
            
            self.scatter_resid_with_predictors(
                outdf, predictor_col, model_results, resid=residuals, model_type=model_type
            )
            
            # Joint plots
            self.jointplot_predictors_vs_discounts_offered(
                outdf, predictor_col, discount_column_name
            )
            
            self.jointplot_predictors_vs_total_revenue(
                outdf, predictor_col, endog_feature
            )
        
        self.logger.info("=" * 80)
        self.logger.info("MODEL DIAGNOSTICS COMPLETED")
        self.logger.info("=" * 80)
    
    def performance_plots(
        self,
        df: pd.DataFrame,
        model_results,
        performance_var: str,
        resid: Union[np.ndarray, pd.Series],
        model_type: str
    ):
        """
        Generate performance diagnostic plots.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with data
        model_results : statsmodels results
            Fitted model results
        performance_var : str
            Performance variable to group by
        resid : pd.Series, optional
            Residuals (calculated if not provided)
        model_type : str, default="OLS"
            Type of model
        """
        df = df.copy()
        
        if resid is not None:
            df['Residuals'] = resid
        else:
            df['Residuals'] = model_results.resid
        
        # Group by performance variable
        package_performance = df.groupby([performance_var], observed=True)['Residuals'].agg(
            ['mean', 'count']
        ).sort_values(by='mean', ascending=False)
        
        self.logger.info(f"Mean residuals by {performance_var}:\n{package_performance}")
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        package_performance['mean'].plot(kind='bar', color='blue')
        plt.title(f'Mean Residuals by {performance_var}')
        plt.xlabel(performance_var)
        plt.ylabel('Mean Residuals')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{model_type}_mean_residuals_by_{performance_var}.png"
        plot_filepath = os.path.join(self.plot_dir, plot_filename)
        plt.savefig(plot_filepath, dpi=300)
        self.logger.info(f"Saved plot to {plot_filepath}")
        plt.close()
    
    def scatter_resid_with_predictors(
        self,
        df: pd.DataFrame,
        predictor: str,
        model_results,
        resid: Optional[pd.Series] = None,
        model_type: str = "OLS"
    ):
        """
        Create scatter plot of residuals vs predictor.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing predictor
        predictor : str
            Name of predictor variable
        model_results : statsmodels results
            Fitted model results
        resid : pd.Series, optional
            Residuals
        model_type : str, default="OLS"
            Type of model
        """
        df = df.copy()
        
        if resid is not None:
            df['Residuals'] = resid
        else:
            df['Residuals'] = model_results.resid
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(df[predictor], df['Residuals'], alpha=0.5)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Residuals vs {predictor}')
        plt.xlabel(predictor)
        plt.ylabel('Residuals')
        plt.axhline(0, color='red', linestyle='--')
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{model_type}_residuals_vs_{predictor}.png"
        plot_filepath = os.path.join(self.plot_dir, plot_filename)
        plt.savefig(plot_filepath, dpi=300)
        self.logger.debug(f"Saved residuals plot for {predictor}")
        plt.close()
    
    def jointplot_predictors_vs_discounts_offered(
        self, df: pd.DataFrame, predictor_col: str, discount_col: str
    ):
        """Joint plot of predictor variable vs. discounts offered."""
        plt.figure(figsize=(10, 6))
        sns.jointplot(
            data=df,
            x=predictor_col,
            y=discount_col,
            kind='scatter',
            height=8,
            marginal_kws=dict(bins=25, fill=True)
        )
        plt.xticks(rotation=45, ha='right')
        plt.suptitle(f'Joint Plot of {predictor_col} vs {discount_col}', y=1)
        plot_filename = f"jointplot_{predictor_col}_vs_{discount_col}.png"
        plot_filepath = os.path.join(self.plot_dir, plot_filename)
        plt.savefig(plot_filepath)
        self.logger.debug(f"Saved joint plot: {predictor_col} vs {discount_col}")
        plt.close()
    
    def jointplot_predictors_vs_total_revenue(
        self, df: pd.DataFrame, predictor_col: str, total_revenue_col: str
    ):
        """Joint plot of predictor variable vs. total revenue."""
        plt.figure(figsize=(10, 6))
        sns.jointplot(
            data=df,
            x=predictor_col,
            y=total_revenue_col,
            kind='scatter',
            height=8,
            marginal_kws=dict(bins=25, fill=True)
        )
        plt.xticks(rotation=45, ha='right')
        plt.suptitle(f'Joint Plot of {predictor_col} vs {total_revenue_col}', y=1)
        plot_filename = f"jointplot_{predictor_col}_vs_{total_revenue_col}.png"
        plot_filepath = os.path.join(self.plot_dir, plot_filename)
        plt.savefig(plot_filepath)
        self.logger.debug(f"Saved joint plot: {predictor_col} vs {total_revenue_col}")
        plt.close()
    
    def ols_comprehensive_diagnostics(
        self,
        model_trainer,
        model_validator
    ):
        """
        Run comprehensive OLS-specific diagnostics.
        
        Parameters
        ----------
        model_trainer : OLSModelTrainer
            Trained model instance
        model_validator : ModelValidator
            Validator instance with CV results
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("COMPREHENSIVE OLS DIAGNOSTICS")
        self.logger.info("=" * 80)
        
        # Get model components
        mod_dsn = model_trainer.get_design_matrix()
        exog_features = model_trainer.get_exog_features()
        
        # Calculate VIF
        X = mod_dsn.loc[:, exog_features]
        self.calculate_vif(X)
        
        # Check multicollinearity
        self._check_multicollinearity()
        
        # Check homoscedasticity
        self._check_homoscedasticity(model_trainer, model_validator)
        
        # Check normality
        self._check_normality(model_trainer)
        
        # Generate performance report
        self._generate_ols_performance_report(model_trainer, model_validator)
        
        self.logger.info("=" * 80)
        self.logger.info("COMPREHENSIVE DIAGNOSTICS COMPLETED")
        self.logger.info("=" * 80)
    
    def _check_multicollinearity(self):
        """Check for multicollinearity using VIF."""
        self.logger.info("\n" + "-" * 80)
        self.logger.info("MULTICOLLINEARITY CHECK")
        self.logger.info("-" * 80)
        
        if self.vif_scores is not None:
            high_vif = self.vif_scores[self.vif_scores['VIF'] > 10]
            if not high_vif.empty:
                self.logger.warning(f"Found {len(high_vif)} features with high VIF")
            else:
                self.logger.info("✓ No multicollinearity issues detected")
    
    def _check_homoscedasticity(self, model_trainer, model_validator):
        """Check for homoscedasticity using residual plots."""
        self.logger.info("\n" + "-" * 80)
        self.logger.info("HOMOSCEDASTICITY CHECK")
        self.logger.info("-" * 80)
        
        if hasattr(model_validator, 'final_ols_cv_predictions'):
            mod_dsn = model_trainer.get_design_matrix()
            mod_dsn_copy = mod_dsn.copy()
            mod_dsn_copy['Predicted'] = model_validator.final_ols_cv_predictions
            
            model_results = model_trainer.get_model_results()
            self.scatter_resid_with_predictors(
                mod_dsn_copy,
                'Predicted',
                model_results,
                model_results.resid,
                model_type="OLS"
            )
            self.logger.info("✓ Residual vs Predicted plot generated")
    
    def _check_normality(self, model_trainer):
        """Check for normality of residuals."""
        self.logger.info("\n" + "-" * 80)
        self.logger.info("NORMALITY CHECK")
        self.logger.info("-" * 80)
        
        model_results = model_trainer.get_model_results()
        if hasattr(model_results, 'resid'):
            residuals = model_results.resid
            self.logger.info(f"Residuals mean: {residuals.mean():.6f}")
            self.logger.info(f"Residuals std: {residuals.std():.4f}")
            self.logger.info("✓ Residual statistics calculated")
    
    def _generate_ols_performance_report(self, model_trainer, model_validator):
        """Generate comprehensive performance report."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("COMPREHENSIVE OLS PERFORMANCE REPORT")
        self.logger.info("=" * 80)
        
        # CV metrics
        if hasattr(model_validator, 'ols_cv_mean'):
            self.logger.info(f"Cross-Validation R²: {model_validator.ols_cv_mean:.4f} (±{model_validator.ols_cv_std * 2:.4f})")
        
        # Training metrics
        model_results = model_trainer.get_model_results()
        if model_results is not None:
            self.logger.info(f"Training R²: {model_results.rsquared:.4f}")
            self.logger.info(f"Adjusted R²: {model_results.rsquared_adj:.4f}")
            self.logger.info(f"F-statistic: {model_results.fvalue:.2f}")
            self.logger.info(f"F-statistic p-value: {model_results.f_pvalue:.4f}")
        
        # Overfitting check
        if hasattr(model_validator, 'ols_train_test_gap'):
            gap = model_validator.ols_train_test_gap
            status = "⚠️ POTENTIAL OVERFITTING" if gap > 0.1 else "✓ GOOD GENERALIZATION"
            self.logger.info(f"Train-CV Gap: {gap:.4f} {status}")
