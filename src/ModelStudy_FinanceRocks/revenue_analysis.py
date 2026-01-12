#!/usr/bin/python3

"""
Revenue Analysis Module

This module handles revenue analysis strategies and high-value customer identification.

Author: Sharat Sharma
Date: Jan-2026
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List


class RevenueAnalyzer:
    """
    Handles revenue analysis and high-value customer identification.
    
    This class implements multiple strategies:
    1. Direct revenue ranking
    2. Revenue by customer segment
    3. High-value feature adopters
    4. High-activity users
    5. Model-predicted high-value companies
    
    Attributes:
        customer_data (pd.DataFrame): Raw customer data
        model_results: Trained model results
        mod_dsn (pd.DataFrame): Model design matrix
        exog_features (List[str]): Exogenous feature names
        endog_feature (str): Endogenous feature name
        logger (logging.Logger): Logger instance
        revenue_analysis_results (Dict): Stored analysis results
    """
    
    def __init__(
        self,
        customer_data: pd.DataFrame,
        model_results=None,
        mod_dsn: pd.DataFrame = None,
        exog_features: List[str] = None,
        endog_feature: str = None,
        logger: logging.Logger = None
    ):
        """
        Initialize the revenue analyzer.
        
        Parameters
        ----------
        customer_data : pd.DataFrame
            Raw customer data with revenue columns
        model_results : optional
            Trained model results for predictions
        mod_dsn : pd.DataFrame, optional
            Model design matrix
        exog_features : List[str], optional
            Exogenous feature names
        endog_feature : str, optional
            Endogenous feature name
        logger : logging.Logger, optional
            Logger instance
        """
        self.customer_data = customer_data.copy()
        self.model_results = model_results
        self.mod_dsn = mod_dsn
        self.exog_features = exog_features
        self.endog_feature = endog_feature
        
        # Setup logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        
        # Initialize results storage
        self.revenue_analysis_results = {}
    
    def find_highest_revenue_generating_companies(
        self,
        top_n: int = 20
    ) -> Dict[str, pd.DataFrame]:
        """
        Find highest revenue generating companies using multiple strategies.
        
        Parameters
        ----------
        top_n : int, default=20
            Number of top companies to display in each analysis
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing results from each analysis strategy
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("HIGHEST REVENUE GENERATING COMPANIES - COMPREHENSIVE ANALYSIS")
        self.logger.info("=" * 80)
        
        # Run all strategies
        self.analyze_direct_revenue_ranking(top_n)
        self.analyze_revenue_by_segment()
        self.analyze_feature_adopters(top_n)
        self.analyze_high_activity_users(top_n)
        self.analyze_model_predicted_companies(top_n)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("REVENUE ANALYSIS COMPLETED")
        self.logger.info("=" * 80)
        
        return self.revenue_analysis_results
    
    def analyze_direct_revenue_ranking(self, top_n: int) -> pd.DataFrame:
        """
        Strategy 1: Rank companies by total discounted revenue.
        
        Parameters
        ----------
        top_n : int
            Number of top companies to return
            
        Returns
        -------
        pd.DataFrame
            Top N companies by revenue with breakdown
        """
        self.logger.info("\n" + "-" * 80)
        self.logger.info("STRATEGY 1: DIRECT REVENUE RANKING")
        self.logger.info("-" * 80)
        
        # Get all discounted revenue columns
        revenue_cols = [col for col in self.customer_data.columns 
                       if col.startswith('line_total_discounted_vat_0_rev_')]
        
        # Calculate total revenue per company
        self.customer_data['total_revenue'] = self.customer_data[revenue_cols].sum(axis=1)
        
        # Calculate revenue breakdown
        self.customer_data['revenue_package'] = self.customer_data.get('line_total_discounted_vat_0_rev_package', 0)
        self.customer_data['revenue_vouchers'] = self.customer_data.get('line_total_discounted_vat_0_rev_ex_vouchers', 0)
        self.customer_data['revenue_employees'] = self.customer_data.get('line_total_discounted_vat_0_rev_ex_employees', 0)
        self.customer_data['revenue_integrations'] = self.customer_data.get('line_total_discounted_vat_0_rev_integrations', 0)
        self.customer_data['revenue_mobile'] = self.customer_data.get('line_total_discounted_vat_0_rev_mobile', 0)
        self.customer_data['revenue_addon'] = self.customer_data.get('line_total_discounted_vat_0_rev_addon', 0)
        
        # Get top N companies by revenue
        top_revenue_companies = self.customer_data.nlargest(top_n, 'total_revenue')[[
            'id', 'package', 'company_type_label', 'tol_1_eng', 
            'headcount_class', 'revenue_class',
            'total_revenue', 'revenue_package', 'revenue_vouchers', 
            'revenue_employees', 'revenue_integrations', 'revenue_mobile', 'revenue_addon'
        ]]
        
        self.logger.info(f"\nTop {top_n} Companies by Total Revenue:")
        self.logger.info(f"\n{top_revenue_companies.to_string()}")
        
        # Summary statistics
        self.logger.info(f"\nRevenue Summary Statistics:")
        self.logger.info(f"Total Revenue (All Companies): €{self.customer_data['total_revenue'].sum():,.2f}")
        self.logger.info(f"Mean Revenue: €{self.customer_data['total_revenue'].mean():,.2f}")
        self.logger.info(f"Median Revenue: €{self.customer_data['total_revenue'].median():,.2f}")
        self.logger.info(f"Top {top_n} Companies Revenue: €{top_revenue_companies['total_revenue'].sum():,.2f}")
        self.logger.info(f"Top {top_n} Revenue Contribution: {(top_revenue_companies['total_revenue'].sum() / self.customer_data['total_revenue'].sum() * 100):.2f}%")
        
        self.revenue_analysis_results['top_revenue_companies'] = top_revenue_companies
        return top_revenue_companies
    
    def analyze_revenue_by_segment(self) -> Dict[str, pd.DataFrame]:
        """
        Strategy 2: Analyze revenue by customer segments.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            Revenue analysis by different segments
        """
        self.logger.info("\n" + "-" * 80)
        self.logger.info("STRATEGY 2: REVENUE BY CUSTOMER SEGMENT")
        self.logger.info("-" * 80)
        
        segment_results = {}
        
        # Analyze by package
        package_analysis = self.customer_data.groupby('package').agg({
            'total_revenue': ['sum', 'mean', 'median', 'count']
        }).round(2)
        package_analysis.columns = ['Total_Revenue', 'Avg_Revenue', 'Median_Revenue', 'Company_Count']
        package_analysis = package_analysis.sort_values('Total_Revenue', ascending=False)
        
        self.logger.info("\nRevenue by Package:")
        self.logger.info(f"\n{package_analysis.to_string()}")
        segment_results['package'] = package_analysis
        
        # Analyze by company type
        if 'company_type_label' in self.customer_data.columns:
            company_type_analysis = self.customer_data.groupby('company_type_label').agg({
                'total_revenue': ['sum', 'mean', 'median', 'count']
            }).round(2)
            company_type_analysis.columns = ['Total_Revenue', 'Avg_Revenue', 'Median_Revenue', 'Company_Count']
            company_type_analysis = company_type_analysis.sort_values('Total_Revenue', ascending=False)
            
            self.logger.info("\nRevenue by Company Type:")
            self.logger.info(f"\n{company_type_analysis.to_string()}")
            segment_results['company_type'] = company_type_analysis
        
        # Analyze by industry (tol_1_eng)
        if 'tol_1_eng' in self.customer_data.columns:
            industry_analysis = self.customer_data.groupby('tol_1_eng').agg({
                'total_revenue': ['sum', 'mean', 'median', 'count']
            }).round(2)
            industry_analysis.columns = ['Total_Revenue', 'Avg_Revenue', 'Median_Revenue', 'Company_Count']
            industry_analysis = industry_analysis.sort_values('Total_Revenue', ascending=False).head(10)
            
            self.logger.info("\nTop 10 Industries by Revenue:")
            self.logger.info(f"\n{industry_analysis.to_string()}")
            segment_results['industry'] = industry_analysis
        
        # Analyze by size class
        if 'headcount_class' in self.customer_data.columns:
            size_analysis = self.customer_data.groupby('headcount_class').agg({
                'total_revenue': ['sum', 'mean', 'median', 'count']
            }).round(2)
            size_analysis.columns = ['Total_Revenue', 'Avg_Revenue', 'Median_Revenue', 'Company_Count']
            size_analysis = size_analysis.sort_values('Total_Revenue', ascending=False)
            
            self.logger.info("\nRevenue by Company Size (Headcount):") 
            self.logger.info(f"\n{size_analysis.to_string()}")
            segment_results['size'] = size_analysis
        
        self.revenue_analysis_results['segment_analysis'] = segment_results
        return segment_results
    
    def analyze_feature_adopters(self, top_n: int) -> pd.DataFrame:
        """
        Strategy 3: Identify high-value feature adopters.
        
        Parameters
        ----------
        top_n : int
            Number of top companies to return
            
        Returns
        -------
        pd.DataFrame
            Companies with high feature adoption and revenue
        """
        self.logger.info("\n" + "-" * 80)
        self.logger.info("STRATEGY 3: HIGH-VALUE FEATURE ADOPTERS")
        self.logger.info("-" * 80)
        
        # Identify add-on columns
        addon_cols = [col for col in self.customer_data.columns if col.startswith('add_')]
        
        if addon_cols:
            # Calculate number of add-ons per company
            self.customer_data['addon_count'] = self.customer_data[addon_cols].sum(axis=1)
            
            # Analyze revenue by add-on adoption
            addon_analysis = self.customer_data.groupby('addon_count').agg({
                'total_revenue': ['sum', 'mean', 'median', 'count']
            }).round(2)
            addon_analysis.columns = ['Total_Revenue', 'Avg_Revenue', 'Median_Revenue', 'Company_Count']
            
            self.logger.info("\nRevenue by Number of Add-ons Adopted:")
            self.logger.info(f"\n{addon_analysis.to_string()}")
            
            # Find companies with high add-on adoption and high revenue
            high_adopters = self.customer_data[self.customer_data['addon_count'] >= 3].nlargest(top_n, 'total_revenue')[[
                'id', 'package', 'addon_count', 'total_revenue'
            ] + addon_cols]
            
            self.logger.info(f"\nTop {top_n} High-Adoption, High-Revenue Companies:")
            self.logger.info(f"\n{high_adopters.to_string()}")
            
            self.revenue_analysis_results['feature_adopters'] = high_adopters
            return high_adopters
        else:
            self.logger.warning("No add-on columns found in data")
            return pd.DataFrame()
    
    def analyze_high_activity_users(self, top_n: int) -> Dict[str, pd.DataFrame]:
        """
        Strategy 4: Find companies with high usage metrics.
        
        Parameters
        ----------
        top_n : int
            Number of top companies to return
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Top companies by different usage metrics
        """
        self.logger.info("\n" + "-" * 80)
        self.logger.info("STRATEGY 4: HIGH-ACTIVITY USERS")
        self.logger.info("-" * 80)
        
        activity_results = {}
        
        # Define usage metrics
        usage_metrics = {
            'total_records_sum': 'Total Vouchers',
            'total_SI_PI_vouchers_sum': 'Total Invoices',
            'record_count_salary_mean': 'Average Employees'
        }
        
        # Calculate efficiency metrics
        if 'total_records_sum' in self.customer_data.columns:
            self.customer_data['revenue_per_voucher'] = (
                self.customer_data['total_revenue'] / 
                self.customer_data['total_records_sum'].replace(0, np.nan)
            )
        
        if 'total_SI_PI_vouchers_sum' in self.customer_data.columns:
            self.customer_data['revenue_per_invoice'] = (
                self.customer_data['total_revenue'] / 
                self.customer_data['total_SI_PI_vouchers_sum'].replace(0, np.nan)
            )
        
        if 'record_count_salary_mean' in self.customer_data.columns:
            self.customer_data['revenue_per_employee'] = (
                self.customer_data['total_revenue'] / 
                self.customer_data['record_count_salary_mean'].replace(0, np.nan)
            )
        
        # Find high-activity companies
        for metric_col, metric_name in usage_metrics.items():
            if metric_col in self.customer_data.columns:
                top_activity = self.customer_data.nlargest(top_n, metric_col)[[
                    'id', 'package', metric_col, 'total_revenue'
                ]]
                
                self.logger.info(f"\nTop {top_n} Companies by {metric_name}:")
                self.logger.info(f"\n{top_activity.to_string()}")
                activity_results[metric_col] = top_activity
        
        self.revenue_analysis_results['high_activity'] = activity_results
        return activity_results
    
    def analyze_model_predicted_companies(self, top_n: int) -> Dict[str, pd.DataFrame]:
        """
        Strategy 5: Use trained model to identify high-value companies.
        
        Parameters
        ----------
        top_n : int
            Number of top companies to return
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Companies with highest predicted revenue, over/under-performers
        """
        self.logger.info("\n" + "-" * 80)
        self.logger.info("STRATEGY 5: MODEL-PREDICTED HIGH-VALUE COMPANIES")
        self.logger.info("-" * 80)
        
        if not hasattr(self, 'model_results') or self.model_results is None:
            self.logger.warning("⚠️ Skipping Strategy 5: Model results not available")
            return {}
        
        if self.mod_dsn is None or self.exog_features is None:
            self.logger.warning("⚠️ Skipping Strategy 5: Model design matrix not available")
            return {}
        
        # DIAGNOSTIC: Log shapes and features
        self.logger.info("\n=== PREDICTION DIAGNOSTICS ===")
        self.logger.info(f"Model was trained with {len(self.model_results.params)} parameters")
        self.logger.info(f"Model parameter names: {list(self.model_results.params.index)}")
        self.logger.info(f"exog_features has {len(self.exog_features)} features")
        self.logger.info(f"mod_dsn has {len(self.mod_dsn.columns)} columns")
        
        # Get actual features from trained model
        model_features = list(self.model_results.params.index)
        
        # Use intersection of model features and available columns
        prediction_features = [col for col in model_features if col in self.mod_dsn.columns]
        
        self.logger.info(f"\nUsing {len(prediction_features)} features for prediction")
        
        try:
            # Extract data with only features the model was trained on
            X_pred = self.mod_dsn[prediction_features]
            
            # Make predictions
            predictions = self.model_results.predict(X_pred)
            
            self.logger.info("✓ Predictions successful!")
            
            # Create analysis DataFrame
            prediction_analysis = pd.DataFrame({
                'predicted_log_revenue': predictions,
                'actual_log_revenue': self.mod_dsn[self.endog_feature],
                'residual': self.mod_dsn[self.endog_feature] - predictions
            })
            
            # Convert back from log scale
            prediction_analysis['predicted_revenue'] = np.exp(prediction_analysis['predicted_log_revenue'])
            prediction_analysis['actual_revenue'] = np.exp(prediction_analysis['actual_log_revenue'])
            
            # Top predicted revenue companies
            top_predicted = prediction_analysis.nlargest(top_n, 'predicted_revenue')
            self.logger.info(f"\nTop {top_n} Companies by Predicted Revenue:")
            self.logger.info(f"\n{top_predicted[['predicted_revenue', 'actual_revenue', 'residual']].to_string()}")
            
            # Over-performers (positive residuals)
            over_performers = prediction_analysis.nlargest(top_n, 'residual')
            self.logger.info(f"\nTop {top_n} Over-Performing Companies (Actual > Predicted):")
            self.logger.info(f"\n{over_performers[['predicted_revenue', 'actual_revenue', 'residual']].to_string()}")
            
            # Under-performers (negative residuals) - potential churn risk
            under_performers = prediction_analysis.nsmallest(top_n, 'residual')
            self.logger.info(f"\nTop {top_n} Under-Performing Companies (Actual < Predicted) - CHURN RISK:")
            self.logger.info(f"\n{under_performers[['predicted_revenue', 'actual_revenue', 'residual']].to_string()}")
            
            prediction_results = {
                'top_predicted': top_predicted,
                'over_performers': over_performers,
                'under_performers': under_performers
            }
            
            self.revenue_analysis_results['model_predictions'] = prediction_results
            return prediction_results
            
        except ValueError as e:
            self.logger.error(f"ValueError during prediction: {e}")
            self.logger.warning("⚠️ Skipping Strategy 5 due to prediction error")
            return {}
