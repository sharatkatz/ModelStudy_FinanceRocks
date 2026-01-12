#!/usr/bin/python3

"""
Model Validation Module

This module handles cross-validation, feature stability analysis, and overfitting assessment.

Author: Sharat Sharma
Date: Jan-2026
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
import warnings

from model_training import OLSModelTrainer


class ModelValidator:
    """
    Handles model validation through cross-validation and stability analysis.
    
    This class provides:
    - K-fold cross-validation
    - Feature stability analysis across folds
    - Overfitting detection
    - Model comparison between CV and original OLS
    
    Attributes:
        model_trainer (OLSModelTrainer): Model trainer instance
        logger (logging.Logger): Logger instance
        ols_cv_scores (np.ndarray): Cross-validation scores
        ols_cv_mean (float): Mean CV score
        ols_cv_std (float): Standard deviation of CV scores
        ols_train_score (float): Training score
        ols_train_test_gap (float): Gap between train and CV performance
        feature_stability (pd.DataFrame): Feature stability metrics
        fold_r2_scores (List[float]): R² scores per fold
        final_ols_cv_model: Final CV model fitted on all data
        final_ols_cv_predictions: Predictions from CV model
    """
    
    def __init__(
        self,
        model_trainer: OLSModelTrainer,
        logger: logging.Logger = None
    ):
        """
        Initialize the model validator.
        
        Parameters
        ----------
        model_trainer : OLSModelTrainer
            Trained model instance
        logger : logging.Logger, optional
            Logger instance
        """
        self.model_trainer = model_trainer
        
        # Setup logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        
        # Initialize validation attributes
        self.ols_cv_scores = None
        self.ols_cv_mean = None
        self.ols_cv_std = None
        self.ols_train_score = None
        self.ols_train_test_gap = None
        self.feature_stability = None
        self.fold_r2_scores = None
        self.final_ols_cv_model = None
        self.final_ols_cv_predictions = None
        self.X_ols = None
        self.y_ols = None
        self.cv_folds = None
        self.scoring_metric = None
        self.ols_comparison = None
    
    def setup_OLS_with_CV(
        self,
        cv_folds: int = 5,
        scoring: str = 'r2',
        random_state: int = 42
    ) -> float:
        """
        Configure and run OLS regression with k-fold cross-validation.
        
        Parameters
        ----------
        cv_folds : int, default=5
            Number of cross-validation folds
        scoring : str, default='r2'
            Scoring metric ('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error')
        random_state : int, default=42
            Random state for reproducible CV splits
            
        Returns
        -------
        float
            Mean cross-validation score
        """
        try:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("STARTING CROSS-VALIDATION")
            self.logger.info("=" * 80)
            
            # Get data from model trainer
            mod_dsn = self.model_trainer.get_design_matrix()
            exog_features = self.model_trainer.get_exog_features()
            endog_feature = self.model_trainer.get_endog_feature()
            
            if mod_dsn is None or exog_features is None:
                raise ValueError("Model must be trained before running cross-validation")
            
            self.logger.info(f"Setting up OLS with {cv_folds}-fold cross-validation...")
            
            # Prepare data
            X = mod_dsn.loc[:, exog_features]
            y = mod_dsn.loc[:, endog_feature]
            
            # Store for reference
            self.X_ols = X
            self.y_ols = y
            
            # Perform cross-validation
            self.perform_cross_validation(X, y, cv_folds, scoring, random_state)
            
            # Analyze feature stability across folds
            self.analyze_feature_stability(X, y, cv_folds, random_state)
            
            # Compare with original OLS results
            self.compare_cv_with_original_ols()
            
            # Check for overfitting
            self.assess_overfitting()
            
            self.logger.info("=" * 80)
            self.logger.info("CROSS-VALIDATION COMPLETED")
            self.logger.info("=" * 80)
            
            return self.ols_cv_mean
            
        except Exception as e:
            self.logger.error(f"Error in OLS CV setup: {str(e)}")
            raise
    
    def perform_cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int,
        scoring: str,
        random_state: int
    ) -> np.ndarray:
        """
        Perform k-fold cross-validation for OLS.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        cv_folds : int
            Number of folds
        scoring : str
            Scoring metric
        random_state : int
            Random state
            
        Returns
        -------
        np.ndarray
            Cross-validation scores for each fold
        """
        # Create scorer based on input
        scorers = {
            'r2': make_scorer(r2_score),
            'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
            'neg_mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False)
        }
        
        scorer = scorers.get(scoring, make_scorer(r2_score))
        
        # Create OLS model (without intercept)
        ols_model = LinearRegression(fit_intercept=False)
        
        # Perform cross-validation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(ols_model, X, y, cv=kf, scoring=scorer)
        
        # Store results
        self.ols_cv_scores = cv_scores
        self.ols_cv_mean = cv_scores.mean()
        self.ols_cv_std = cv_scores.std()
        self.cv_folds = cv_folds
        self.scoring_metric = scoring
        
        # Fit final model on all data for comparison
        self.final_ols_cv_model = ols_model.fit(X, y)
        self.final_ols_cv_predictions = self.final_ols_cv_model.predict(X)
        
        # Calculate training score
        if scoring == 'r2':
            train_score = r2_score(y, self.final_ols_cv_predictions)
        elif scoring == 'neg_mean_squared_error':
            train_score = -mean_squared_error(y, self.final_ols_cv_predictions)
        else:
            train_score = r2_score(y, self.final_ols_cv_predictions)
        
        self.ols_train_score = train_score
        
        self.logger.info("\n" + "-" * 80)
        self.logger.info("CROSS-VALIDATION RESULTS")
        self.logger.info("-" * 80)
        self.logger.info(f"Scoring metric: {scoring}")
        self.logger.info(f"CV folds: {cv_folds}")
        self.logger.info(f"Cross-validation scores: {cv_scores}")
        self.logger.info(f"Mean CV score: {self.ols_cv_mean:.4f} (+/- {self.ols_cv_std * 2:.4f})")
        self.logger.info(f"Training score: {train_score:.4f}")
        
        return cv_scores
    
    def analyze_feature_stability(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int,
        random_state: int
    ) -> pd.DataFrame:
        """
        Analyze how consistently features are selected across CV folds.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        cv_folds : int
            Number of folds
        random_state : int
            Random state
            
        Returns
        -------
        pd.DataFrame
            Feature stability metrics including mean coefficient, std, CV
        """
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        feature_importance = pd.DataFrame(index=X.columns)
        fold_r2_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Fit OLS on training fold
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ols_fold = LinearRegression(fit_intercept=False)
                ols_fold.fit(X_train, y_train)
            
            # Store coefficients
            feature_importance[f'fold_{fold+1}'] = ols_fold.coef_
            
            # Calculate fold performance
            y_pred = ols_fold.predict(X_test)
            fold_r2 = r2_score(y_test, y_pred)
            fold_r2_scores.append(fold_r2)
        
        # Calculate feature stability metrics
        feature_importance['mean_coef'] = feature_importance.mean(axis=1)
        feature_importance['std_coef'] = feature_importance.std(axis=1)
        feature_importance['abs_mean_coef'] = feature_importance.iloc[:, :cv_folds].abs().mean(axis=1)
        feature_importance['cv'] = (
            feature_importance['std_coef'] / feature_importance['mean_coef'].abs()
        ).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Frequency of non-zero coefficients
        feature_importance['non_zero_freq'] = (feature_importance.iloc[:, :cv_folds] != 0).mean(axis=1)
        
        self.feature_stability = feature_importance
        self.fold_r2_scores = fold_r2_scores
        
        self.logger.info("\n" + "-" * 80)
        self.logger.info("FEATURE STABILITY ANALYSIS")
        self.logger.info("-" * 80)
        self.logger.info("Most stable features (lowest coefficient of variation):")
        stable_features = feature_importance.nsmallest(10, 'cv')['cv']
        for feature, cv in stable_features.items():
            self.logger.info(f"  {feature}: CV={cv:.4f}")
        
        return feature_importance
    
    def compare_cv_with_original_ols(self) -> pd.DataFrame:
        """
        Compare CV OLS results with original OLS results.
        
        Returns
        -------
        pd.DataFrame
            Comparison of coefficients between original and CV models
        """
        original_results = self.model_trainer.get_model_results()
        exog_features = self.model_trainer.get_exog_features()
        
        if original_results is None:
            self.logger.warning("Original OLS results not available for comparison")
            return None
        
        # Compare coefficients
        original_coefs = pd.Series(original_results.params, name='original_ols')
        cv_coefs = pd.Series(self.final_ols_cv_model.coef_, index=exog_features, name='cv_ols')
        
        comparison = pd.DataFrame({
            'Original_OLS': original_coefs,
            'CV_OLS': cv_coefs,
            'Difference': original_coefs - cv_coefs,
            'Abs_Difference': (original_coefs - cv_coefs).abs()
        })
        
        self.ols_comparison = comparison
        
        self.logger.info("\n" + "-" * 80)
        self.logger.info("OLS MODEL COMPARISON")
        self.logger.info("-" * 80)
        self.logger.info(f"Original OLS R²: {original_results.rsquared:.4f}")
        self.logger.info(f"CV OLS R²: {r2_score(self.y_ols, self.final_ols_cv_predictions):.4f}")
        self.logger.info("Largest coefficient differences:")
        
        largest_diffs = comparison.nlargest(5, 'Abs_Difference')
        for feature, row in largest_diffs.iterrows():
            self.logger.info(f"  {feature}: {row['Original_OLS']:.6f} vs {row['CV_OLS']:.6f}")
        
        return comparison
    
    def assess_overfitting(self) -> Dict[str, float]:
        """
        Assess potential overfitting by comparing train vs validation performance.
        
        Returns
        -------
        Dict[str, float]
            Dictionary with overfitting metrics
        """
        if not hasattr(self, 'ols_train_score') or not hasattr(self, 'ols_cv_mean'):
            return None
        
        train_cv_gap = self.ols_train_score - self.ols_cv_mean
        self.ols_train_test_gap = train_cv_gap
        
        self.logger.info("\n" + "-" * 80)
        self.logger.info("OVERFITTING ASSESSMENT")
        self.logger.info("-" * 80)
        self.logger.info(f"Train-CV performance gap: {train_cv_gap:.4f}")
        
        if abs(train_cv_gap) > 0.1:
            self.logger.warning("⚠️ Large train-CV gap detected - potential overfitting!")
        elif abs(train_cv_gap) > 0.05:
            self.logger.info("Moderate train-CV gap - model appears reasonably generalizable")
        else:
            self.logger.info("✓ Small train-CV gap - model generalizes well")
        
        return {
            'train_score': self.ols_train_score,
            'cv_mean': self.ols_cv_mean,
            'train_cv_gap': train_cv_gap
        }
