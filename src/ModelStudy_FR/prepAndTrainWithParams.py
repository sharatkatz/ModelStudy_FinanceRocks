#!/usr/bin/python3

"""
Flow:
 1. this the main module for te ModelStudy_FR package.
 2. This module serves as a pipeline for cayying out preprocecssing steps and modeling tasks.
 3. It integrates various components from the pakage to facilitate a steamlined workflow.



 Author: Sharat Sharma
 Date: Nov-25
"""

from optparse import Option
from typing import Dict, List
from asyncio.log import logger
import os
import sys
import logging
import numpy as np  # type: ignore
import json

import statsmodels.api as sm  # type: ignore
from statsmodels.sandbox.regression.gmm import IV2SLS  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from pprint import pprint
import seaborn as sns  # type: ignore

import pandas as pd  # type: ignore

from ModelStudy_FR import setup_temp_plot_directory, PreProcessor  # type: ignore


pd.set_option('display.max_columns', None)


class PrepareInDatForModeling:

    fileName, (filePath, plot_dir) = "customer_data.parquet", setup_temp_plot_directory()
    logile = "prepAndTrainWithParams.log"

    if os.path.exists(logile):
        os.remove(logile)

    logging.basicConfig(
        filename=logile,
        filemode='w',
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO)

    logger = logging.getLogger(__name__)

    def __init__(self):
        """
        Initialize the model training pipeline with data preprocessing steps.
        This constructor performs the following operations in sequence:
        1. Initializes a PreProcessor instance to handle data treatment
        2. Loads and treats predictor columns from the input dataset
        3. Applies one-hot encoding to nominal predictor variables
        4. Applies ordinal encoding to ordinal predictors using predefined mappings
        5. Cleans and renames predictor column names for consistency
        6. Applies log transformation to selected predictor variables
        7. Applies log transformation to the outcome variable
        Attributes:
            preprocessor (PreProcessor): Instance of PreProcessor for data treatment operations
            model_predictors_df (pd.DataFrame): DataFrame containing treated predictor columns
            log_transformed_predictors_df (pd.DataFrame): DataFrame with log-transformed predictors
            log_transformed_predictors_and_outcome_df (pd.DataFrame): Final DataFrame with both
                log-transformed predictors and outcome variable
        Side Effects:
            - Logs information about each preprocessing step
            - Modifies column names in-place during the cleaning process
            - Creates transformed versions of the input data
        Notes:
            This initialization assumes that filePath, fileName, plot_dir, and logger
            attributes are defined in the parent class or earlier in the class definition.
        """

        # Initialize PreProcessor to handle data treatment
        self.preprocessor = PreProcessor(
            self.filePath, self.fileName, self.plot_dir)
        self.model_predictors_df = self.preprocessor.treat_predictor_columns()
        self.logger.info(
            "Initial dataset loaded and predictor columns treated.")
        # self.logger.info(self.model_predictors_df.head())

        # One-hot encode nominal predictors
        onehot_encoded_df = self.preprocessor.one_hot_encode_nominal_predictors(
            self.model_predictors_df)
        self.logger.info(
            "One-hot encoding completed. Here are the new columns:")
        self.logger.info(onehot_encoded_df.columns.tolist())
        # self.logger.info(onehot_encoded_df.head())

        # Ordinal encode ordinal predictors using predefined mappings
        label_encoded_df = self.preprocessor.map_ordinal_predictors(
            onehot_encoded_df
        )
        self.logger.info(
            "Ordinal encoding using mappings is completed. Here are the new columns:")
        self.logger.info(label_encoded_df.columns.tolist())
        self.logger.info("\n\n")
        # self.logger.info(label_encoded_df.head())
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            self.logger.info(label_encoded_df.dtypes)

        # Clean and rename predictor column names
        cleaned_cols = self.preprocessor.treat_predictor_column_names(
            label_encoded_df.columns.tolist()
        )
        label_encoded_df.columns = cleaned_cols
        label_encoded_df.rename(
            columns=self.preprocessor.tol_2_remaps(),
            inplace=True)
        self.logger.info("Predictor column names cleaned and renamed.")
        self.logger.info(label_encoded_df.columns.tolist())

        # Apply log transformation to selected predictor variables
        self.logger.info("\n\n")
        self.logger.info("Applying log transformation to variables...\n")
        log_transformed_predictors_df = self.preprocessor.log_transform_df(
            label_encoded_df
        )
        self.log_transformed_predictors_df = log_transformed_predictors_df
        # self.logger.info(
        #     "Log transformation completed. Here is the transformed DataFrame preview:")
        # self.logger.info(log_transformed_predictors_df.head())

        # Log-transform the outcome variable
        self.log_transformed_predictors_and_outcome_df = self.preprocessor.log_transform_outcome(
            log_transformed_predictors_df)

        self.logger.info("\n\n")


class ModelTrain(PrepareInDatForModeling, PreProcessor):

    def __init__(
            self,
            model_type: str = "Additive",
            endog: str = "total_revenue"):
        """
        Initialize the model preparation and training class.

        Args:
            model_type (str, optional): Type of model to use. Must be either "Additive" or "Multiplicative".
                Defaults to "Additive".
            endog (str, optional): Name of the endogenous (dependent) variable for GLM modeling.
                Defaults to "total_revenue".

        Attributes:
            model_type (str): The type of model being used.
            additive_type (bool): Flag indicating if the model is additive type. Initialized to False.
            multiplicative_type (bool): Flag indicating if the model is multiplicative type. Initialized to False.
            endog_glm (str): The endogenous variable name for GLM.
            exog_glm (list): List of exogenous (independent) variables for GLM. Initialized as empty list.
            add_constant (bool): Flag indicating whether to add a constant term to the model. Initialized to False.
            addon_usage_columns (list): List of add-on usage column names retrieved from preprocessor.
            usage_metric_columns (list): List of usage metric column names retrieved from preprocessor.
            customer_profile_columns (list): List of customer profile column names retrieved from preprocessor.
        """
        super().__init__()
        self.model_type: str = model_type
        self.additive_type: bool = False
        self.multiplicative_type: bool = False
        self.endog_glm: str = endog
        self.add_constant: bool = False
        self.addon_usage_columns: list = self.preprocessor.get_addon_usage_columns()
        self.usage_metric_columns: list = self.preprocessor.get_usage_metrics_columns()
        self.customer_profile_columns: list = self.preprocessor.get_customer_profile_columns()
        self.reverse_mappings = self.reverse_ordinal_variables_and_category()
        self.discount_column_name = 'discounts_offered'

    def calculate_and_exclude_highly_correlated_vars_from_predictors(
            self,
            corr_matrix: pd.DataFrame,
            threshold: float = 0.9) -> List[str]:
        """Identify and return a list of variable names to exclude based on high correlation.
        This method analyzes the provided correlation matrix to identify pairs of variables
        that exhibit a correlation coefficient above the specified threshold. For each pair,
        one variable is selected for exclusion to mitigate multicollinearity in modeling.
        Args:
            corr_matrix (pd.DataFrame): A DataFrame representing the correlation matrix of variables.
            threshold (float, optional): The correlation coefficient threshold above which variables
                are considered highly correlated. Defaults to 0.9.
        Returns:
            List[str]: A list of variable names identified for exclusion due to high correlation.
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
                        f"Excluding '{var2}' due to high correlation ({corr_value:.2f}) with '{var1}'")
        return list(to_exclude)

    def exclude_insignificant_vars_from_model(
            self,
            model_results: sm.regression.linear_model.RegressionResultsWrapper,
            alpha: float = 0.05) -> List[str]:
        """Identify and return a list of variable names to exclude based on statistical insignificance.
        This method examines the p-values of coefficients in the provided model results
        to identify variables that are statistically insignificant at the specified alpha level.
        Args:
            model_results (sm.regression.linear_model.RegressionResultsWrapper): Fitted model results
                from which to extract p-values.
            alpha (float, optional): Significance level for determining insignificance. Defaults to 0.05.
        Returns:
            List[str]: A list of variable names identified for exclusion due to insignificance.
        """
        to_exclude = []
        pvalues = model_results.pvalues
        for var, pval in pvalues.items():
            if pval > alpha:
                to_exclude.append(var)
                self.logger.info(
                    f"Excluding '{var}' due to insignificance (p-value: {pval:.4f})")
        return to_exclude

    def setup_OLS(self):
        """Configure and run the OLS modeling pipeline for a logged outcome variable.
        This method sets up and fits a series of Ordinary Least Squares (OLS) regression models
        on log-transformed data, progressively refining the specification by removing counter-intuitive
        and statistically insignificant coefficients.
        Workflow
        --------
        1. Normalizes model_type to "Additive" if None and sets corresponding flags.
        2. Converts the endogenous variable name to its logged counterpart (appends "_logged").
        3. Builds an exogenous feature set from log-transformed predictors by including:
            - package_*, company_type_*, tol_1_eng_*, *_class columns
            - Selected *_logged columns from to_log_transform_columns()
            - Add-on usage columns from addon_usage_columns
        4. Excludes:
            - Revenue-after-discounts *_logged columns
            - High-collinearity proxies (total_records_mean_logged, total_SI_PI_vouchers_mean_logged)
        5. Replaces +/-inf with NaN and drops rows with missing data.
        6. Fits three sequential OLS models:
            - Baseline: Full logged specification
            - Revision 1: Excludes counter-intuitive coefficients (negative add-on effects)
            - Revision 2 (Final): Excludes both counter-intuitive and insignificant coefficients (α=0.05)
        ----------
             Results are stored in self.final_ols_results
        Attributes Modified
        -------------------
        model_type : str
             Set to "Additive" if None.
        additive_type : bool
             Set to True for additive modeling.
        multiplicative_type : bool
             Set to True if model_type starts with "mul".
        endog_glm : str
             Updated to "<original_endog>_logged".
        exog_glm : list of str
             Selected exogenous variable names after inclusion/exclusion rules.
        add_constant : bool
             Set to False (no intercept added).
        final_ols_results : statsmodels.regression.linear_model.RegressionResultsWrapper
             Fitted results from the final refined OLS model.
        ------------
        The instance must have the following attributes:
        - logger : logging.Logger
        - model_type : str or None
        - endog_glm : str (base name before suffixing)
        - addon_usage_columns : iterable of str
        - log_transformed_predictors_df : pandas.DataFrame
        - log_transformed_predictors_and_outcome_df : pandas.DataFrame
        And methods:
        - get_revenue_after_discounts_columns() -> iterable of str
        - to_log_transform_columns() -> iterable of str
        ------------
        - Logs detailed INFO/WARNING messages about variable selection, data filtering,
          model summaries, and coefficient diagnostics
        - Drops rows with NaNs (created by replacing infinities), potentially reducing sample size
        - Writes three OLS model summaries to the logger
        -----
        - No constant term is added (self.add_constant = False by default)
        ------
        AttributeError
             If required attributes or methods are missing.
        KeyError
             If expected columns are not present in the DataFrames.
        ValueError
             If the design matrix is empty, singular, or otherwise invalid for OLS.
        statsmodels.tools.sm_exceptions.MissingDataError
             If data contains unhandled inf or NaN values.
        Examples
        --------
        >>> analyzer = YourClass()
        >>> analyzer.setup_OLS()
        >>> print(analyzer.final_ols_results.summary())

        docs: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html
        """
        self.additive_type = True

        if self.model_type is None:
            self.model_type = "Additive"

        if self.model_type.lower()[:3] == "add":
            self.additive_type = True

        if self.model_type.lower()[:3] == "mul":
            self.multiplicative_type = True

        # set up params to be used for modeling using method described here:
        # https://www.statsmodels.org/stable/glm.html
        self.endog_glm = f"{self.endog_glm}_logged"
        predictor_cols_superset: List[str] = self.log_transformed_predictors_df.columns.tolist()

        # Fit the GLM model
        # Using Gamma family with log link as per README recommendations
        # https://www.statsmodels.org/stable/glm.html
        # Also refer to README.md in the package for detailed reasoning
        # capture statsmodels.tools.sm_exceptions.MissingDataError: exog contains inf or nans
        # pick up exog variables from
        # log_transformed_predictors_and_outcome_df?

        exclude_cols = []
        for colname in self.get_revenue_after_discounts_columns():
            exclude_cols.append(f"{colname}_logged")

        exclude_cols += [
            # 'record_count_salary_mean_logged',
            'total_records_mean_logged',
            'total_SI_PI_vouchers_mean_logged',
            # 'headcount_class'
        ]

        # Select relevant predictor columns based on naming patterns
        package_cols = [
            col for col in predictor_cols_superset if col.lower().startswith("package")]
        company_type_cols = [
            col for col in predictor_cols_superset if col.lower().startswith("company_type_")]
        tol_1_eng_cols = [
            col for col in predictor_cols_superset if col.startswith("tol_1_eng_")]
        tol_2_eng_cols = [
            col for col in predictor_cols_superset if col.lower().startswith("tol_2_eng_")]
        _class_cols = [
            col for col in predictor_cols_superset if col.lower().endswith("_class")]
        logged_cols = [f"{col}_logged" for col in self.to_log_transform_columns(
        ) if col != self.endog_glm]
        add_usage_cols = [
            col for col in predictor_cols_superset if col.lower() in self.addon_usage_columns]

        # Select columns to include in the model
        include_cols = package_cols + company_type_cols + tol_1_eng_cols + \
            _class_cols + logged_cols + add_usage_cols
        self.logger.info(
            f"Total number of exogenous variables considered for GLM: {
                len(include_cols)}")
        # Define exogenous variables for GLM by excluding specified columns
        self.exog_glm = [
            col for col in include_cols if col not in exclude_cols
        ]
        exclude_correlated_cols = self.calculate_and_exclude_highly_correlated_vars_from_predictors(
            self.log_transformed_predictors_and_outcome_df[self.exog_glm].corr(), threshold=0.7
        )
        exclude_cols += exclude_correlated_cols
        self.logger.info(
            f"Excluding the following columns from exogenous variables: {exclude_cols}")

        # re-construct exog_glm after excluding highly correlated vars
        self.exog_glm = [
            col for col in self.exog_glm if col not in exclude_cols
        ]
        self.logger.info(
            f"Final number of exogenous variables selected for GLM: {len(self.exog_glm)}")

        # Add constant term if additive model
        if self.additive_type:
            self.logger.info("Additive model selected")
            self.add_constant = False
            self.logger.info("Not adding constant term to the model.")
            if self.add_constant:
                self.exog_glm = sm.add_constant(
                    self.exog_glm, has_constant='add')

            self.logger.info(
                "Setting up and fitting GLM with Gamma family and log link function...")
            self.logger.info(
                f"Endogenous variable (outcome): {
                    self.endog_glm}")
            self.logger.info(
                f"Exogenous variables (predictors): {
                    self.exog_glm}")

            # Prepare exogenous and endogenous DataFrames, handling infinite
            # values
            self.logger.info(
                "Preparing exogenous and endogenous DataFrames for GLM...")
            exog_glm_df = self.log_transformed_predictors_and_outcome_df[
                self.exog_glm].replace(
                    [float('inf'), -float('inf')], float('nan')
            )

            # Calculate discounts run for each customer
            self.discounts_offered = pd.DataFrame(
                self.sum_discount_columns(self.preprocessor.customer_data), columns=[self.discount_column_name]
            )
            exog_glm_df =pd.concat(
                [exog_glm_df, self.discounts_offered], axis=1
            )

            endog_glm_df = self.log_transformed_predictors_and_outcome_df[
                self.endog_glm].replace(
                    [float('inf'), -float('inf')], float('nan')
            )
            self.logger.info(
                "Handling missing values by dropping rows with NaNs...")
            mod_dsn = pd.concat(
                [exog_glm_df, endog_glm_df], axis=1
            ).dropna()

            # re-set discounts_offered after dropping NaNs
            self.discounts_offered = mod_dsn[[self.discount_column_name]]

            # Fit OLS model as a starting point
            self.logger.info("Fitting OLS model as a starting point...")
            glm_ols_doublelogged_model = sm.OLS(
                endog=mod_dsn.loc[:, self.endog_glm],
                exog=mod_dsn.loc[:, self.exog_glm],
            )

            ols_results = glm_ols_doublelogged_model.fit()
            self.logger.info("OLS model fitting completed.")
            self.logger.info(ols_results.summary())


            # Identify counter-intuitive coefficients to exclude
            counter_intuitive_coefs = self.exclude_insignificant_vars_from_model(
                ols_results, alpha=0.05
            )
            self.logger.info(f"Counter-intuitive coefficients to exclude: {counter_intuitive_coefs}")

            # exclude counter-intuitive coefficients and re-fit OLS model
            # and re-run model
            self.logger.info(
                "Fitting OLS model excluding counter-intuitive coefficients...")
            glm_ols_doublelogged_model_rev1 = sm.OLS(
                endog=mod_dsn.loc[:, self.endog_glm],
                exog=mod_dsn.loc[:, [_ for _ in self.exog_glm if _ not in counter_intuitive_coefs]],
            )
            refined_ols_results = glm_ols_doublelogged_model_rev1.fit()
            self.logger.info("Refined OLS model fitted.")
            self.logger.info(refined_ols_results.summary())

            insignificant_coefs = []
            self.logger.info(
                "Instantiating final OLS model excluding insignificant coefficients based on alpha=0.05..."
            )
            glm_ols_doublelogged_model_rev2 = sm.OLS(
                endog=mod_dsn.loc[:, self.endog_glm],
                exog=mod_dsn.loc[:, [_ for _ in self.exog_glm if _ not in counter_intuitive_coefs + insignificant_coefs]],
            )
            final_ols_results = glm_ols_doublelogged_model_rev2.fit()
            self.logger.info("Final OLS model fitted.")
            self.logger.info(final_ols_results.summary())
            self.final_ols_results = final_ols_results
            self.mod_dsn = mod_dsn

    def reverse_ordinal_variables_and_category(self) -> Dict[str, Dict[int, str]]:
        """
        Reverses the ordinal encoding mappings for ordinal predictor variables.
        This method retrieves the original ordinal mappings from the PreProcessor
        class and inverts them to create a reverse mapping from encoded integers
        back to their original categorical labels.

        Returns:
            Dict[str, Dict[int, str]]: A dictionary where each key is an ordinal
            predictor variable name and the value is another dictionary mapping
            encoded integers back to their original categorical labels.
        """
        reverse_mappings: Dict[str, Dict[int, str]] = {}
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

    def reverse_ordinal_predictors(self, indf: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse the ordinal encoding of predictor columns back to their original categorical values.

        This method takes a DataFrame with ordinally encoded columns and converts them back
        to their original categorical representations using the stored reverse mappings.

        Parameters
        ----------
        indf : pd.DataFrame
            Input DataFrame containing ordinally encoded predictor columns.

        Returns
        -------
        pd.DataFrame
            A copy of the input DataFrame with specified columns converted back to their
            original categorical values. Columns not found in the DataFrame are skipped
            with a warning message.

        Notes
        -----
        - The method creates a copy of the input DataFrame to avoid modifying the original.
        - For each column in `self.reverse_mappings`, the method attempts to apply the
          reverse mapping.
        - If a column from the mappings is not present in the DataFrame, it is skipped
          and a message is printed.
        - Mapped columns are converted to categorical dtype.
        - Values that don't have a mapping are kept as their original values.

        Examples
        --------
        >>> processor = DataProcessor()
        >>> # Assuming reverse_mappings contains {'size': {0: 'small', 1: 'medium', 2: 'large'}}
        >>> encoded_df = pd.DataFrame({'size': [0, 1, 2, 1]})
        >>> decoded_df = processor.reverse_ordinal_predictors(encoded_df)
        >>> print(decoded_df['size'].tolist())
        ['small', 'medium', 'large', 'medium']
        """
        outdf = indf.copy()
        print("\n\n")
        # dict.keys() is iterable; you can also iterate the dict directly
        for map_col in self.reverse_mappings.keys():
            if map_col not in outdf.columns:
                print(f"Column '{map_col}' not found in data. Skipping.")
                continue
            print(f"Doing the reverse mapping of col -- {map_col}")
            mapping = self.reverse_mappings[map_col]
            # Use Series.map to convert categories -> codes. Keep originals if
            # unmapped.
            outdf[map_col] = outdf[map_col].map(mapping).astype('category')
        return outdf

    def undummify_company_type_labels(
            self,
            indf: pd.DataFrame) -> pd.DataFrame:
        """
        Converts one-hot encoded 'company_type_' columns back to a single categorical column.

        This method identifies all columns in the input DataFrame that start with
        'company_type_', which are assumed to be one-hot encoded representations of
        company types. It then reconstructs a single 'company_type' column by determining
        the original category for each row based on the one-hot encoding.

        Parameters
        ----------
        indf : pd.DataFrame
            Input DataFrame containing one-hot encoded 'company_type_' columns.

        Returns
        -------
        pd.DataFrame
            A copy of the input DataFrame with a new 'company_type' column added,
            representing the original categorical values. The one-hot encoded columns
            are retained in the output DataFrame.
        """
        outdf = indf.copy()
        company_type_cols = [col for col in outdf.columns if col.startswith('company_type_')]
        # Create 'company_type' column by finding the max one-hot encoded column
        outdf['company_type'] = outdf[company_type_cols].idxmax(axis='columns').str.replace('company_type_', '')
        outdf['company_type'] = outdf['company_type'].astype('category')
        # Generate scatter plot of residuals vs. company_type
        # self.logger.info("Generating scatter plot of residuals vs. company_type")
        self.scatter_resid_with_predictors(outdf, 'company_type', self.final_ols_results)
        return outdf

    def undummify_tol_1_eng_labels(self, indf: pd.DataFrame) -> pd.DataFrame:
        """
        Converts one-hot encoded 'tol_1_eng_' columns back to a single categorical column.

        This method identifies all columns in the input DataFrame that start with
        'tol_1_eng_', which are assumed to be one-hot encoded representations of
        TOL 1 English levels. It then reconstructs a single 'tol_1_eng' column by determining
        the original category for each row based on the one-hot encoding.

        Parameters
        ----------
        indf : pd.DataFrame
            Input DataFrame containing one-hot encoded 'tol_1_eng_' columns.

        Returns
        -------
        pd.DataFrame
            A copy of the input DataFrame with a new 'tol_1_eng' column added,
            representing the original categorical values. The one-hot encoded columns
            are retained in the output DataFrame.
        """
        outdf = indf.copy()
        tol_1_eng_cols = [col for col in outdf.columns if col.startswith('tol_1_eng_')]
        # Create 'tol_1_eng' column by finding the max one-hot encoded column
        outdf['tol_1_eng'] = outdf[tol_1_eng_cols].idxmax(axis='columns').str.replace('tol_1_eng_', '')
        outdf['tol_1_eng'] = outdf['tol_1_eng'].astype('category')
        # Generate scatter plot of residuals vs. tol_1_eng
        # self.logger.info("Generating scatter plot of residuals vs. tol_1_eng")
        self.scatter_resid_with_predictors(outdf, 'tol_1_eng', self.final_ols_results)
        return outdf

    def model_diagnostics(
            self,
            mod_dsn: pd.DataFrame,
            model_results: sm.OLS) -> None:
        """
        Generates diagnostic plots and summary statistics for model residuals by package.
        This method takes the model design DataFrame and the fitted OLS model results,
        computes residuals, and evaluates the mean residual per package. It logs the
        process and saves a bar plot visualizing the mean residuals for each package
        to the specified plot directory.
        Args:
            mod_dsn (pd.DataFrame): The model design DataFrame all the qualified predictors and outcome.
            model_results (sm.OLS): The fitted OLS model results object from statsmodels.
        Returns:
            None
        """

        if model_results is None:
            self.logger.error("Model results object is None. Cannot perform diagnostics.")
            return

        outdf: pd.DataFrame = self.reverse_ordinal_predictors(mod_dsn)
        performance_cols = list(self.map_ordinal_variables_and_category_order().keys())
        # make sure the dimensions match
        if outdf.shape[0] != self.discounts_offered.shape[0]:
            self.logger.error(
                "Mismatch in number of rows between model design DataFrame and discounts offered data."
            )
            return
        outdf[self.discount_column_name] = self.discounts_offered

        self.logger.info(f"Here is the list of performance cols: {performance_cols}")
        # Generate performance plots for each performance/ordinal variable
        for p_col in performance_cols:
            if p_col not in outdf.columns:
                self.logger.warning(
                    f"Performance column '{p_col}' not found in DataFrame. Skipping model diagnostics.")
                continue
            self.performance_plots(
                outdf,
                model_results,
                p_col
            )

        for predictor_col in self.exog_glm:
            if predictor_col not in outdf.columns:
                self.logger.warning(
                    f"Predictor column '{predictor_col}' not found in DataFrame. Skipping model diagnostics.")
                continue
            self.scatter_resid_with_predictors(
                outdf,
                predictor_col,
                model_results
            )

            # Generate joint plots of predictors vs. discounts offered
            self.jointplot_predictors_vs_discounts_offered(
                outdf,
                predictor_col,
                self.discount_column_name
            )

            self.jointplot_preditors_vs_total_revenue(
                outdf,
                predictor_col,
                self.endog_glm
            )

        return

    def jointplot_predictors_vs_discounts_offered(self, indf: pd.DataFrame, predictor_col: str, discount_col: str) -> None:
        """Joint plot of predictor variable vs. discounts offered."""
        plt.figure(figsize=(10, 6))
        sns.jointplot(
            data=indf,
            x=predictor_col,
            y=discount_col,
            kind='scatter',
            height=8,
            marginal_kws=dict(bins=25, fill=True)
        )
        plt.xticks(rotation=45, ha='right') # Rotate labels by 45 degrees, align right
        plt.suptitle(f'Joint Plot of {predictor_col} vs {discount_col}', y=1)
        plot_filename = f"jointplot_{predictor_col}_vs_{discount_col}.png"
        plot_filepath = os.path.join(ModelTrain.plot_dir, plot_filename)
        plt.savefig(plot_filepath)
        self.logger.info(f"Saved joint plot of {predictor_col} vs {discount_col} to {plot_filepath}")
        plt.cla() # clear the current axes
        plt.clf() # clear the current figure
        plt.close() # close the figure window
        return

    def jointplot_preditors_vs_total_revenue(self, indf: pd.DataFrame, predictor_col: str, total_revenue_col: str) -> None:
        """Joint plot of predictor variable vs. total revenue."""
        plt.figure(figsize=(10, 6))
        sns.jointplot(
            data=indf,
            x=predictor_col,
            y=total_revenue_col,
            kind='scatter',
            height=8,
            marginal_kws=dict(bins=25, fill=True)
        )
        plt.xticks(rotation=45, ha='right') # Rotate labels by 45 degrees, align right
        plt.suptitle(f'Joint Plot of {predictor_col} vs {total_revenue_col}', y=1)
        plot_filename = f"jointplot_{predictor_col}_vs_{total_revenue_col}.png"
        plot_filepath = os.path.join(ModelTrain.plot_dir, plot_filename)
        plt.savefig(plot_filepath)
        self.logger.info(f"Saved joint plot of {predictor_col} vs {total_revenue_col} to {plot_filepath}")
        plt.cla() # clear the current axes
        plt.clf() # clear the current figure
        plt.close() # close the figure window
        return

    def performance_plots(self, indf: pd.DataFrame, model_results: sm.OLS, performance_var: str) -> None:
        """
        Generates and saves a bar plot of mean residuals by a specified performance variable.

        This function computes the residuals from the provided model results and
        groups them by the specified performance variable. It then creates a bar plot
        showing the mean residuals for each category of the performance variable.

        Parameters
        ----------
        indf : pd.DataFrame
            Input DataFrame containing the data used for modeling.
        model_results : sm.OLS
            Fitted OLS model results object from statsmodels.
        performance_var : str
            The name of the column in `indf` to group by for performance evaluation.

        Returns
        -------
        None
            The function saves the plot to a file and does not return any value.
        """
        # Calculate residuals
        indf = indf.copy()
        indf['Residuals'] = model_results.resid

        # Group by performance variable and calculate mean residuals
        package_performance: pd.DataFrame = indf.groupby([performance_var], observed=True)['Residuals'].agg(
            ['mean', 'count']).sort_values(by='mean', ascending=False)

        self.logger.info(f"Mean residuals by {performance_var}:\n{package_performance}")
        # Create bar plot
        plt.figure(figsize=(10, 6))
        package_performance['mean'].plot(kind='bar', color='blue')
        plt.title(f'Mean Residuals by {performance_var}')
        plt.xlabel(performance_var)
        plt.ylabel('Mean Residuals')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot to file
        plot_filename = f"mean_residuals_by_{performance_var}.png"
        plot_filepath = os.path.join(ModelTrain.plot_dir, plot_filename)
        plt.savefig(plot_filepath)
        self.logger.info(f"Saved mean residuals plot for {performance_var} to {plot_filepath}")
        plt.cla() # clear the current axes
        plt.clf() # clear the current figure
        plt.close() # close the figure window

    def scatter_resid_with_predictors(self, indf: pd.DataFrame, predictor_col: str, model_results: sm.OLS) -> None:
        """
        Generates and saves a scatter plot of residuals against a specified predictor variable.

        This function computes the residuals from the provided model results and
        creates a scatter plot showing the relationship between the residuals and
        the specified predictor variable.

        Parameters
        ----------
        indf : pd.DataFrame
            Input DataFrame containing the data used for modeling.
        predictor_col : str
            The name of the predictor column in `indf` to plot against residuals.
        model_results : sm.OLS
            Fitted OLS model results object from statsmodels
        Returns
        -------
        None
            The function saves the plot to a file and does not return any value.
        """
        # Calculate residuals
        indf = indf.copy()
        indf['Residuals'] = model_results.resid

        # Create scatter plot
        # self.logger.info(f"Generating scatter plot of residuals vs. {predictor_col}")
        plt.figure(figsize=(10, 6))
        plt.scatter(indf[predictor_col], indf['Residuals'], alpha=0.5)
        plt.xticks(rotation=45, ha='right') # Rotate labels by 45 degrees, align right
        plt.title(f'Residuals vs {predictor_col}')
        plt.xlabel(predictor_col)
        plt.ylabel('Residuals')
        plt.axhline(0, color='red', linestyle='--')
        plt.tight_layout()

        # Save plot to file
        plot_filename = f"residuals_vs_{predictor_col}.png"
        plot_filepath = os.path.join(ModelTrain.plot_dir, plot_filename)
        plt.savefig(plot_filepath)
        self.logger.info(f"Saved residuals scatter plot for {predictor_col} to {plot_filepath}")
        plt.cla() # clear the current axes
        plt.clf() # clear the current figure
        plt.close() # close the figure window

if __name__ == "__main__":
    import traceback
    try:
        _ = PrepareInDatForModeling()
        model_trainer = ModelTrain()
        model_trainer.setup_OLS()
        model_trainer.model_diagnostics(
            model_trainer.mod_dsn,
            model_trainer.final_ols_results
        )

        company_type_labels = model_trainer.undummify_company_type_labels(
            model_trainer.mod_dsn
        )
        logger.info("Undummification of company_type labels completed.")

        tol_1_eng = model_trainer.undummify_tol_1_eng_labels(
            model_trainer.mod_dsn
        )
        logger.info("Undummification of tol_1_eng labels completed.")

    except Exception as e:
        traceback.print_exc()
        sys.exit(1)



