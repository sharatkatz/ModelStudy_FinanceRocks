#!/usr/bin/python3

"""
Pipeline Orchestrator Module

This module coordinates the entire modeling pipeline and provides a simple interface.

Author: Sharat Sharma
Date: Jan-2026
"""

import os
import logging
from typing import Dict, Any, Optional

from data_preparation import DataPreparationPipeline
from model_training import OLSModelTrainer
from model_validation import ModelValidator
from model_diagnostics import ModelDiagnostics
from revenue_analysis import RevenueAnalyzer
from ModelStudy_FR import setup_temp_plot_directory  # type: ignore


class ModelingPipeline:
    """
    Orchestrates the complete modeling pipeline.
    
    This class provides a high-level interface to:
    - Run data preparation
    - Train models
    - Validate models
    - Run diagnostics
    - Perform revenue analysis
    
    Attributes:
        file_path (str): Path to data directory
        file_name (str): Name of data file
        log_file (str): Path to log file
        logger (logging.Logger): Logger instance
        data_prep (DataPreparationPipeline): Data preparation instance
        model_trainer (OLSModelTrainer): Model trainer instance
        model_validator (ModelValidator): Model validator instance
        model_diagnostics (ModelDiagnostics): Diagnostics instance
        revenue_analyzer (RevenueAnalyzer): Revenue analyzer instance
    """
    
    def __init__(
        self,
        file_path: str = None,
        file_name: str = "customer_data.parquet",
        log_file: str = "modeling_pipeline.log"
    ):
        """
        Initialize the modeling pipeline.
        
        Parameters
        ----------
        file_path : str, optional
            Path to data directory
        file_name : str, default="customer_data.parquet"
            Name of data file
        log_file : str, default="modeling_pipeline.log"
            Path to log file
        """
        # Setup paths
        if file_path is None:
            self.file_path, self.plot_dir = setup_temp_plot_directory()
        else:
            self.file_path = file_path
            _, self.plot_dir = setup_temp_plot_directory()
        
        self.file_name = file_name
        self.log_file = log_file
        
        # Setup logging
        self.logger = self.setup_logging()
        
        # Initialize module instances (will be populated as pipeline runs)
        self.data_prep = None
        self.model_trainer = None
        self.model_validator = None
        self.model_diagnostics = None
        self.revenue_analyzer = None
    
    def setup_logging(self) -> logging.Logger:
        """
        Set up logging configuration.
        
        Returns
        -------
        logging.Logger
            Configured logger instance
        """
        # Remove existing log file if it exists
        if os.path.exists(self.log_file):
            try:
                os.remove(self.log_file)
            except OSError:
                pass
        
        # Create logger
        logger = logging.getLogger(f"{__name__}_{id(self)}")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter and add it to the handler
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(file_handler)
        
        logger.info("=" * 80)
        logger.info("MODELING PIPELINE INITIALIZED")
        logger.info("=" * 80)
        
        return logger
    
    def run_data_preparation(self) -> DataPreparationPipeline:
        """
        Run the data preparation pipeline.
        
        Returns
        -------
        DataPreparationPipeline
            Configured data preparation instance
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PHASE 1: DATA PREPARATION")
        self.logger.info("=" * 80)
        
        self.data_prep = DataPreparationPipeline(
            file_path=self.file_path,
            file_name=self.file_name,
            plot_dir=self.plot_dir,
            logger=self.logger
        )
        
        # Run full data preparation pipeline
        self.data_prep.prepare_modeling_dataset()
        
        self.logger.info("✓ Data preparation completed successfully")
        return self.data_prep
    
    def run_model_training(
        self,
        data_prep: Optional[DataPreparationPipeline] = None,
        model_type: str = "Additive",
        endog: str = "total_revenue"
    ) -> OLSModelTrainer:
        """
        Run model training.
        
        Parameters
        ----------
        data_prep : DataPreparationPipeline, optional
            Data preparation instance (uses self.data_prep if None)
        model_type : str, default="Additive"
            Type of model
        endog : str, default="total_revenue"
            Endogenous variable name
            
        Returns
        -------
        OLSModelTrainer
            Trained model instance
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PHASE 2: MODEL TRAINING")
        self.logger.info("=" * 80)
        
        if data_prep is None:
            if self.data_prep is None:
                raise ValueError("Data preparation must be run first")
            data_prep = self.data_prep
        
        self.model_trainer = OLSModelTrainer(
            data_prep=data_prep,
            model_type=model_type,
            endog=endog,
            logger=self.logger
        )
        
        # Train the model
        self.model_trainer.setup_OLS()
        
        self.logger.info("✓ Model training completed successfully")
        return self.model_trainer
    
    def run_model_validation(
        self,
        trainer: Optional[OLSModelTrainer] = None,
        cv_folds: int = 5,
        scoring: str = 'r2',
        random_state: int = 42
    ) -> ModelValidator:
        """
        Run model validation with cross-validation.
        
        Parameters
        ----------
        trainer : OLSModelTrainer, optional
            Model trainer instance (uses self.model_trainer if None)
        cv_folds : int, default=5
            Number of CV folds
        scoring : str, default='r2'
            Scoring metric
        random_state : int, default=42
            Random state for reproducibility
            
        Returns
        -------
        ModelValidator
            Model validator instance with results
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PHASE 3: MODEL VALIDATION")
        self.logger.info("=" * 80)
        
        if trainer is None:
            if self.model_trainer is None:
                raise ValueError("Model training must be run first")
            trainer = self.model_trainer
        
        self.model_validator = ModelValidator(
            model_trainer=trainer,
            logger=self.logger
        )
        
        # Run cross-validation
        self.model_validator.setup_OLS_with_CV(
            cv_folds=cv_folds,
            scoring=scoring,
            random_state=random_state
        )
        
        self.logger.info("✓ Model validation completed successfully")
        return self.model_validator
    
    def run_model_diagnostics(
        self,
        trainer: Optional[OLSModelTrainer] = None,
        validator: Optional[ModelValidator] = None
    ) -> ModelDiagnostics:
        """
        Run comprehensive model diagnostics.
        
        Parameters
        ----------
        trainer : OLSModelTrainer, optional
            Model trainer instance (uses self.model_trainer if None)
        validator : ModelValidator, optional
            Model validator instance (uses self.model_validator if None)
            
        Returns
        -------
        ModelDiagnostics
            Diagnostics instance with results
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PHASE 4: MODEL DIAGNOSTICS")
        self.logger.info("=" * 80)
        
        if trainer is None:
            if self.model_trainer is None:
                raise ValueError("Model training must be run first")
            trainer = self.model_trainer
        
        if validator is None:
            if self.model_validator is None:
                self.logger.warning("Model validator not available, some diagnostics may be limited")
        
        self.model_diagnostics = ModelDiagnostics(
            preprocessor=trainer.preprocessor,
            plot_dir=self.plot_dir,
            logger=self.logger
        )
        
        # Run comprehensive diagnostics
        self.model_diagnostics.ols_comprehensive_diagnostics(
            model_trainer=trainer,
            model_validator=validator if validator else self.model_validator
        )
        
        # Run full model diagnostics
        self.model_diagnostics.model_diagnostics(
            mod_dsn=trainer.get_design_matrix(),
            model_results=trainer.get_model_results(),
            exog_features=trainer.get_exog_features(),
            endog_feature=trainer.get_endog_feature(),
            discount_column=trainer.discounts_offered,
            discount_column_name=trainer.discount_column_name,
            model_type="OLS"
        )
        
        self.logger.info("✓ Model diagnostics completed successfully")
        return self.model_diagnostics
    
    def run_revenue_analysis(
        self,
        trainer: Optional[OLSModelTrainer] = None,
        top_n: int = 20
    ) -> RevenueAnalyzer:
        """
        Run revenue analysis.
        
        Parameters
        ----------
        trainer : OLSModelTrainer, optional
            Model trainer instance (uses self.model_trainer if None)
        top_n : int, default=20
            Number of top companies to analyze
            
        Returns
        -------
        RevenueAnalyzer
            Revenue analyzer instance with results
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PHASE 5: REVENUE ANALYSIS")
        self.logger.info("=" * 80)
        
        if trainer is None:
            if self.model_trainer is None:
                raise ValueError("Model training must be run first")
            trainer = self.model_trainer
        
        self.revenue_analyzer = RevenueAnalyzer(
            customer_data=trainer.preprocessor.customer_data,
            model_results=trainer.get_model_results(),
            mod_dsn=trainer.get_design_matrix(),
            exog_features=trainer.get_exog_features(),
            endog_feature=trainer.get_endog_feature(),
            logger=self.logger
        )
        
        # Run all revenue analysis strategies
        self.revenue_analyzer.find_highest_revenue_generating_companies(top_n=top_n)
        
        self.logger.info("✓ Revenue analysis completed successfully")
        return self.revenue_analyzer
    
    def run_full_pipeline(
        self,
        exclude_high_vif: bool = True,
        cv_folds: int = 5,
        revenue_top_n: int = 20,
        run_diagnostics: bool = True,
        run_revenue_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete modeling pipeline.
        
        Parameters
        ----------
        exclude_high_vif : bool, default=True
            Whether to exclude high VIF features and retrain
        cv_folds : int, default=5
            Number of CV folds
        revenue_top_n : int, default=20
            Number of top companies in revenue analysis
        run_diagnostics : bool, default=True
            Whether to run model diagnostics
        run_revenue_analysis : bool, default=True
            Whether to run revenue analysis
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all pipeline results
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STARTING FULL MODELING PIPELINE")
        self.logger.info("=" * 80)
        
        try:
            # Phase 1: Data Preparation
            self.run_data_preparation()
            
            # Phase 2: Model Training
            self.run_model_training()
            
            # Phase 3: Model Validation
            self.run_model_validation(cv_folds=cv_folds)
            
            # Optional: Exclude high VIF features and retrain
            if exclude_high_vif and self.model_diagnostics is None:
                # Need to calculate VIF first
                temp_diagnostics = ModelDiagnostics(
                    preprocessor=self.model_trainer.preprocessor,
                    plot_dir=self.plot_dir,
                    logger=self.logger
                )
                X = self.model_trainer.get_design_matrix().loc[:, self.model_trainer.get_exog_features()]
                temp_diagnostics.calculate_vif(X)
                
                # Get top 2 high VIF features
                if temp_diagnostics.vif_scores is not None:
                    top_two_vif = temp_diagnostics.vif_scores.head(2)
                    self.logger.info(f"\nTop 2 High VIF Features:\n{top_two_vif}")
                    
                    # Remove from feature list and retrain
                    self.model_trainer.exog_glm = [
                        col for col in self.model_trainer.exog_glm 
                        if col not in top_two_vif['Feature'].values
                    ]
                    
                    self.logger.info("\nRetraining model after VIF-based feature exclusion...")
                    self.model_trainer.setup_OLS()
                    self.run_model_validation(cv_folds=cv_folds)
            
            # Phase 4: Model Diagnostics (optional)
            if run_diagnostics:
                self.run_model_diagnostics()
            
            # Phase 5: Revenue Analysis (optional)
            if run_revenue_analysis:
                self.run_revenue_analysis(top_n=revenue_top_n)
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("✓ FULL PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            
            return self.get_results_summary()
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get summary of all pipeline results.
        
        Returns
        -------
        Dict[str, Any]
            Summary of results from all pipeline stages
        """
        summary = {
            'data_preparation': {
                'completed': self.data_prep is not None,
                'predictors_shape': self.data_prep.log_transformed_predictors_df.shape if self.data_prep else None,
                'full_data_shape': self.data_prep.log_transformed_predictors_and_outcome_df.shape if self.data_prep else None
            },
            'model_training': {
                'completed': self.model_trainer is not None,
                'n_features': len(self.model_trainer.get_exog_features()) if self.model_trainer else None,
                'r_squared': self.model_trainer.get_model_results().rsquared if self.model_trainer else None,
                'adj_r_squared': self.model_trainer.get_model_results().rsquared_adj if self.model_trainer else None
            },
            'model_validation': {
                'completed': self.model_validator is not None,
                'cv_mean': self.model_validator.ols_cv_mean if self.model_validator else None,
                'cv_std': self.model_validator.ols_cv_std if self.model_validator else None,
                'train_cv_gap': self.model_validator.ols_train_test_gap if self.model_validator else None
            },
            'model_diagnostics': {
                'completed': self.model_diagnostics is not None,
                'high_vif_count': len(self.model_diagnostics.vif_scores[self.model_diagnostics.vif_scores['VIF'] > 10]) if self.model_diagnostics and self.model_diagnostics.vif_scores is not None else None
            },
            'revenue_analysis': {
                'completed': self.revenue_analyzer is not None,
                'strategies_run': len(self.revenue_analyzer.revenue_analysis_results) if self.revenue_analyzer else 0
            }
        }
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PIPELINE RESULTS SUMMARY")
        self.logger.info("=" * 80)
        for phase, results in summary.items():
            self.logger.info(f"\n{phase.upper().replace('_', ' ')}:")
            for key, value in results.items():
                self.logger.info(f"  {key}: {value}")
        
        return summary
