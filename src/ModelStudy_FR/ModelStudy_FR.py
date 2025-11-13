#!/usr/bin/python3

"""Main module."""
"""File: ModelStudy_FR.py
Summary:
 This is the prep module for the ModelStudy_FR package.
 This module serves as a pipeline for carrying out preprocessing steps and modeling tasks.
 It integrates various components from the package to facilitate a streamlined workflow.

Flow:
1. Import necessary modules and classes.
2. Define global variables for file paths.
3. Set up a temporary directory for storing plots.
4. The module is designed to be imported and used by other scripts or run as a standalone
    script.
5. When run as a standalone script, it initializes the plot directory.
6. Input files are expected to be located in a specific directory structure.
7. We model Total Revenue (Sum of all revenue components): $Y = \text{Total Discounted Revenue} = \\sum (\text{All } line\\_total\\_discounted\\_vat\\_0\\_rev\\_\text{...})$$
as the outcome variable.
8. The package variable is "package".
9. Predictor variables include various revenue components and customer attributes.
10. All the meta information is classified under four types of variables:
    - Customer/Company Profile:
        ["id",
        "package",
        "accounting_office_id",
        "company_type_label",
        "tol_1_eng",
        "tol_2_eng",
        "headcount_class",
        "revenue_class",
        "ao_revenue_class",
        "ao_headcount_class"]

    - Usage/Activity Metrics (over 12 months):
        ["total_records_months_used",
         "total_records_mean",
         "total_records_sum",
         "total_SI_PI_vouchers_months_used",
         "total_SI_PI_vouchers_mean",
         "total_SI_PI_vouchers_sum",
         "record_count_salary_months_used",
         "record_count_salary_mean"]

    - Add-on/Feature Usage (Binary/Mobile):
        ["add_api",
         "add_bank_account",
         "add_contract_invoicing",
         "add_cust_invoice",
         "add_ext_dimensions",
         "add_inventory",
         "add_junior",
         "add_mobile",
         "add_sftp",
         "mobile_user_count"]

    - Revenue Metrics (Before and After Discounts):
        ["line_total_vat_0_rev_package",
         "line_total_vat_0_rev_ex_vouchers",
         "line_total_vat_0_rev_ex_employees",
         "line_total_vat_0_rev_integrations",
         "line_total_vat_0_rev_mobile",
         "line_total_vat_0_rev_addon",
         "line_total_vat_0_rev_trx",
         "line_total_discounted_vat_0_rev_package",
         "line_total_discounted_vat_0_rev_ex_vouchers",
         "line_total_discounted_vat_0_rev_ex_employees",
         "line_total_discounted_vat_0_rev_integrations",
         "line_total_discounted_vat_0_rev_mobile",
         "line_total_discounted_vat_0_rev_addon",
         "line_total_discounted_vat_0_rev_trx"]

 11. All the columns and their data descriptors is provided as:
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| COLUMNS                                      | DESCRIPTION                                                                                                                    |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| id                                           | unique end-customer id                                                                                                         |
| product_package                              | product package                                                                                                                |
| accounting_office_id                         | the id of the customer’s accounting office                                                                                     |
| company_type_label                           | the legal type of the customer’s business (e.g. limited liability (LTD), sole proprietor (TMI), etc.)                          |
| tol_1_eng                                    | industry category of the customer’s business (level 1)                                                                         |
| tol_2_eng                                    | industry category of the customer’s business (level 2)                                                                         |
| headcount_class                              | the revenue and customer company/install falls into based on the number of employees it has                                    |
| revenue_class                                | the revenue bracket the customer company falls into based on its annual revenue                                                |
| record_count_accounting_office               | the number of clients of the customer’s accounting office                                                                      |
| total_records_months_used                    | number of months the customer has used a voucher during the 12-month period                                                    |
| total_records_mean                           | average number of vouchers used per month in 12 months                                                                         |
| total_records_sum                            | total number of vouchers used in 12 months                                                                                     |
| total_SI_PI_vouchers_months_used             | number of months the customer generated sales and purchase invoices during the 12-month period                                 |
| total_SI_PI_vouchers_mean                    | average number of sales and purchase invoices used per month in 12 months                                                      |
| total_SI_PI_vouchers_sum                     | total number of sales and purchase invoices used in 12 months                                                                  |
| record_count_salary_months_used              | number of months the customer has used payroll functionality during the past 12 months                                         |
| record_count_salary_mean                     | average number of employees on payroll during past 12 months                                                                   |
| add_api                                      | 1. Indicates if the customer is using one of the add-on features such as api, credit search, bank account, consolidation, etc. |
|                                              | 2. "add_api" and "add_sftp" denote integration usage                                       |                                   |
|                                              | 3. "add_mobile" denotes the usage of the mobile app                                                                            |
| add_stfp                                     |                                                                                                                                |
| add_bank_account                             |                                                                                                                                |
| add_contract_invoicing                       |                                                                                                                                |
| add_cust_invoice                             |                                                                                                                                |
| add_ext_dimensions                           |                                                                                                                                |
| add_inventory                                |                                                                                                                                |
| add_junior                                   |                                                                                                                                |
| add_mobile                                   |                                                                                                                                |
| mobile_user_count                            | number of users using the mobile app if the customer has enabled the mobile app                                                |
| line_total_vat_0_rev_package                 | revenue from package fees (before discounts)                                                                                   |
| line_total_vat_0_rev_vouchers                | revenue from exceeded voucher fees (before discounts)                                                                          |
| line_total_vat_0_rev_employees               | revenue from exceeded payroll fees (before discounts)                                                                          |
| line_total_vat_0_rev_integrations            | revenue from integrations (before discounts)                                                                                   |
| line_total_vat_0_rev_mobile                  | revenue from mobile (before discounts)                                                                                         |
| line_total_vat_0_rev_addon                   | revenue from other add-ons (before discounts)                                                                                  |
| line_total_discounted_vat_0_rev_package      | revenue from package fees (after discounts)                                                                                    |
| line_total_discounted_vat_0_rev_vouchers     | revenue from exceeded voucher fees (after discounts)                                                                           |
| line_total_discounted_vat_0_rev_employees    | revenue from exceeded payroll fees (after discounts)                                                                           |
| line_total_discounted_vat_0_rev_integrations | revenue from integrations (after discounts)                                                                                    |
| line_total_discounted_vat_0_rev_mobile       | revenue from mobile (after discounts)                                                                                          |
| line_total_discounted_vat_0_rev_addon        | revenue from other add-ons (after discounts)                                                                                   |
| ---------------------------------------------| ------------------------------------------------------------------------------------------------------------------------------ |

 Author: Sharat Sharma
 Date: Nov-25
"""

from typing import List, Literal, Dict, cast
from collections.abc import Callable
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime
import pandas as pd # type: ignore
from matplotlib import pyplot as plt # type: ignore
from pprint import pprint
import seaborn as sns # type: ignore
import numpy as np # type: ignore
from typing import Optional
from pandas import DataFrame, Series # type: ignore


def setup_temp_plot_directory():
    """
    Sets up a temporary plot directory for storing generated plots.

    Returns:
        tuple: (filePath, plot_dir)
    """
    filePath = "../../../input_files"
    plot_dir = os.path.join(filePath, "ModelStudy_FR", "temp_plots")

    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
        print(f"Folder '{plot_dir}' deleted.")

    try:
        print(f"Creating new folder: {plot_dir}")
        os.makedirs(plot_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating folder '{plot_dir}': {e}")
    else:
        print(f"Folder '{plot_dir}' created successfully.")
    finally:
        return filePath, plot_dir


class PreProcessor():
    """_Summary_

    Args:
        ExploratoryDataAnalysis (module): _inherit from EDA class

    This class handles preprocessing tasks such as removing special characters
    from strings and normalizing column values for different data types.
    """

    def __init__(self, filePath: str, fileName: str, plot_dir: str):
        """_summary_

        Args:
            filePath (str): _path to the data file
            fileName (str): _name of the data file
            plot_dir (str): _directory to save plots
        """
        # Verify file existence
        input_file_full_path = os.path.join(filePath, fileName)
        if Path(input_file_full_path).is_file() is False:
            raise FileNotFoundError(
                f"File '{input_file_full_path}' not found.")

        # Load customer data
        self.customer_data = pd.read_parquet(input_file_full_path)
        self.package_var = "package"
        self.unique_packages = self.customer_data[self.package_var].unique(
        ) if self.package_var in self.customer_data.columns else [None]

        # preparing outcome variable: total_revenue
        revenue_columns: list[str] = self.get_revenue_before_discounts_columns(
        )
        # type: ignore
        revenue_data: DataFrame = cast(pd.DataFrame, self.customer_data.loc[:, revenue_columns])
        self.outcome_col = 'total_revenue'
        self.customer_data[self.outcome_col] = revenue_data.sum(axis="columns")

        print(self.customer_data, "\n\n")

        print("Here is the info of the data:\n")
        print(self.customer_data.info(), "\n\n")

        self.plot_dir = plot_dir

        # Carrying out missing reports and visualizations
        self.missing_reports()
        self.jointplot_outcome_vs_predictors()
        self.jointplot_outcome_vs_predictors_by_package()

    def string_and_function(self, str: str, func: Callable) -> None:
        """_summary_

        Args:
            str (str): _description_
            func (Callable): _function to be called

        Returns:
            _type_: _None
        """
        print("\n\n")
        print("-" * 80)
        print(str, "\n")
        _ = func()
        return None

    def missing_reports(self) -> None:
        strings = [
            "Here is the missing values report:\n",  # 6
            "Here is the missing values report by package:\n",  # 7
            "Visualizing missing values:\n",  # 8
            "Visualizing missing values by package:\n",  # 9
        ]

        functions = [
            self.report_missings,  # 6
            self.report_missings_bypackage,  # 7
            self.visualize_missings,  # 8
            self.visualize_missings_bypackage,  # 9
        ]

        for string, func in zip(strings, functions):
            self.string_and_function(string, func)

        return None

    def report_missings(self) -> pd.Series:
        """Report missing values in the dataset."""
        missing_report = self.customer_data.isnull().sum()
        missing_report = missing_report[missing_report > 0]
        print("Missing Values Report:")
        pprint(missing_report.to_dict())
        return missing_report

    def report_missings_bypackage(self) -> None:
        """Report missing values in the dataset by package."""
        if self.package_var not in self.customer_data.columns:
            print(
                f"Package variable '{
                    self.package_var}' not found in data. Skipping missing report by package.")
            return None

        for one_package in self.unique_packages:
            data_subset = self.customer_data.loc[
                self.customer_data[self.package_var] == one_package]
            missing_report = data_subset.isnull().sum()
            missing_report = missing_report[missing_report > 0]
            print(f"Missing Values Report for Package: {one_package}")
            pprint(missing_report.to_dict())
            print("\n")

        return None

    def visualize_missings_bypackage(self) -> None:
        """Visualize missing values in the dataset by package.

        This method creates heatmap visualizations of missing values for each package in the dataset.
        The heatmaps are saved as PNG files in a subdirectory named 'missing_values_by_package'
        under the main plots directory.

        The method uses seaborn's heatmap to visualize where missing values (nulls) occur in the data,
        with a separate plot generated for each unique package value.

        Returns:
            None

        Raises:
            No explicit exceptions are raised, but will print a message and return None if
            the package variable is not found in the dataset.

        Side Effects:
            - Creates 'missing_values_by_package' directory if it doesn't exist
            - Saves heatmap plots as PNG files
            - Prints status messages about saved files
        """
        if self.package_var not in self.customer_data.columns:
            print(
                f"Package variable '{
                    self.package_var}' not found in data. Skipping missing visualization by package."
            )
            return None

        for one_package in self.unique_packages:
            data_subset = self.customer_data.loc[
                self.customer_data[self.package_var] == one_package]

            plt.figure(figsize=(12, 8))
            sns.heatmap(
                data_subset.isnull(),
                cbar=False,
                cmap='viridis')
            plt.title(f'Missing Values Heatmap for Package: {one_package}')
            plt.tight_layout()

            miss_plot_dir = os.path.join(
                self.plot_dir, "missing_values_by_package")
            os.makedirs(miss_plot_dir, exist_ok=True)
            save_path = os.path.join(
                miss_plot_dir, f"missing_values_heatmap_{one_package}.png")
            plt.savefig(save_path)
            plt.close()  # Close the figure to free memory

            print(
                f"Saved missing values heatmap for package '{one_package}': {save_path}")

        return None

    def visualize_missings(self) -> None:
        """Visualize missing values in the dataset."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            self.customer_data.isnull(),
            cbar=False,
            cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.tight_layout()

        miss_plot_dir = os.path.join(self.plot_dir, "missing_values")
        os.makedirs(miss_plot_dir, exist_ok=True)
        save_path = os.path.join(miss_plot_dir, "missing_values_heatmap.png")
        plt.savefig(save_path)
        plt.close()  # Close the figure to free memory

        print(f"Saved missing values heatmap: {save_path}\n\n")
        return None

    def jointplot_outcome_vs_predictors(self) -> None:
        """
        Create and save joint scatter plots comparing the outcome variable to each predictor.

        This method:
        - Retrieves predictor column names from self.get_all_predictor_variables().
        - For each predictor, checks that both the outcome column (self.outcome_col)
            and the predictor column are numeric in self.customer_data.
        - For numeric pairs, creates a seaborn jointplot (scatter) and saves it as a
            PNG file named "jointplot_{outcome}_vs_{predictor}.png" inside a
            "joint_plots" directory located under the path referred to by `plot_dir`.
        - Ensures the target directory exists (creates it if necessary).
        - Clears and closes matplotlib figures after saving to free memory.
        - Prints the saved file path for each generated plot.

        Notes and requirements:
        - Expects self.customer_data to be a pandas DataFrame containing columns
            referenced by self.outcome_col and each predictor name returned by
            get_all_predictor_variables().
        - The variable `plot_dir` must be defined in the surrounding scope (or be an
            accessible attribute) and should be a valid path string or os.PathLike object.
        - Only predictors with numeric dtype (and an outcome with numeric dtype) will
            produce a plot; non-numeric columns are skipped silently.
        - Side effects: creates directories on disk and writes image files; prints to
            standard output.

        Returns:
                None

        Raises:
                KeyError: If expected columns (outcome or predictor) are missing from
                        self.customer_data, a downstream KeyError may be raised when indexing.
                OSError: If directory creation or file saving fails due to filesystem errors.
        """
        """Create joint plots of outcome vs predictors."""
        predictor_cols = self.get_predictor_columns()
        joint_plot_dir = os.path.join(self.plot_dir, "joint_plots")
        os.makedirs(joint_plot_dir, exist_ok=True)

        for predictor_col in predictor_cols:
            if (
                pd.api.types.is_numeric_dtype(
                    self.customer_data[self.outcome_col]) and
                pd.api.types.is_numeric_dtype(
                    self.customer_data[predictor_col])
            ):
                plt.figure(figsize=(12, 10))
                sns.jointplot(
                    x=predictor_col,
                    y=self.outcome_col,
                    data=self.customer_data,
                    kind='scatter',
                    height=7)
                plt.suptitle(
                    f'Joint Plot of {self.outcome_col} vs {predictor_col}',
                    y=1.00)

                save_path = os.path.join(
                    joint_plot_dir,
                    f"jointplot_{self.outcome_col}_vs_{predictor_col}.png"
                )
                plt.savefig(save_path)  # Save the figure
                plt.cla()  # Clear the current axes
                plt.clf()  # Clear the current figure
                plt.close('all')  # Close the figure to free memory

                print(
                    f"Saved joint plot for predictor '{predictor_col}': {save_path}")

        return None

    def jointplot_outcome_vs_predictors_by_package(self) -> None:
        """
        Create and save joint plots of the outcome variable versus each predictor,
        separately for each package value present in the data.

        This method expects the instance (self) to provide the following attributes/methods:
        - customer_data: pandas.DataFrame containing the dataset.
        - package_var: str, name of the column that identifies package/group.
        - outcome_col: str, name of the outcome column to plot on the y-axis.
        - get_all_predictor_variables(): method that returns an iterable of predictor column names.
        - unique_packages: iterable of unique package values to iterate over.
        - plot_dir: base directory path where plot subfolders and files will be written.

        Behavior:
        - If package_var is not present in customer_data, prints a message and returns without doing work.
        - Ensures an output directory named "<plot_dir>/joint_plots_by_package" exists.
        - For each package value in unique_packages and for each predictor returned by
            get_all_predictor_variables():
            - Subsets customer_data to rows where package_var equals the current package.
            - If both the outcome and predictor columns are numeric, adjusts figure sizing.
            - Uses seaborn.jointplot(kind='scatter', height=7) to create the joint plot.
            - Sets a descriptive suptitle indicating the outcome, predictor, and package.
            - Saves the plot to a PNG file with the pattern:
                "jointplot_<outcome_col>_vs_<predictor_col>_package_<package_value>.png"
                inside the joint_plots_by_package directory.
            - Any exception raised while creating or saving an individual plot is caught;
                an error message is printed and processing continues with the next plot.
            - Matplotlib figures/axes are cleared and closed after each attempt to free memory.
            - Prints a confirmation message with the saved file path after each successful save.

        Returns:
        - None

        Notes:
        - The method is side-effecting: it writes files to disk and prints status/error messages.
        - Exceptions during individual plot creation are handled internally and do not stop
            the overall iteration across predictors and packages.
        """
        if self.package_var not in self.customer_data.columns:
            print(
                f"Package variable '{
                    self.package_var}' not found in data. Skipping jointplot by package.")
            return None

        predictor_cols = self.get_predictor_columns()

        # prepare output dir and path before entering try so save_path is
        # always defined
        joint_plot_dir = os.path.join(self.plot_dir, "joint_plots_by_package")
        os.makedirs(joint_plot_dir, exist_ok=True)

        for one_package in self.unique_packages:
            data_subset = self.customer_data.loc[
                self.customer_data[self.package_var] == one_package]
            for predictor_col in predictor_cols:
                if (
                    pd.api.types.is_numeric_dtype(
                        data_subset[self.outcome_col]) and
                    pd.api.types.is_numeric_dtype(
                        data_subset[predictor_col])
                ):
                    plt.figure(figsize=(12, 10))

                save_path = os.path.join(
                    joint_plot_dir, f"jointplot_{
                        self.outcome_col}_vs_{predictor_col}_package_{one_package}.png")

                try:
                    # seaborn.jointplot returns a JointGrid; capture it so we
                    # can set title & save reliably
                    g = sns.jointplot(
                        x=predictor_col,
                        y=self.outcome_col,
                        data=data_subset,
                        kind='scatter',
                        height=7
                    )
                    g.fig.suptitle(
                        f'Joint Plot of {
                            self.outcome_col} vs {predictor_col} for Package: {one_package}',
                        y=1.00)
                    # save using the JointGrid figure
                    g.fig.savefig(save_path)
                except Exception as e:
                    print(
                        f"Error creating joint plot for package '{one_package}' and predictor '{predictor_col}': {e}")
                    continue
                finally:
                    plt.cla()  # Clear the current axes
                    plt.clf()  # Clear the current figure
                    plt.close('all')  # Close the figure to free memory

                    print(
                        f"Saved joint plot for package '{one_package}': {save_path}")

        print("All jointplots by package completed.\n\n ")

        return None

    def remove_special_characters(self, input_string: str) -> str:
        """Remove special characters from the input string."""
        special_characters = ['/', '\\', ':', '*',
                              '?', '"', '<', '>', '|', '-', ",", ";"]
        for char in special_characters:
            input_string = input_string.replace(char, '_')
        return input_string

    def normalize_column_values_category(self, input_string: str) -> str:
        """Normalize column values for categorical data."""
        # Capture AttributeError: 'NoneType' object has no attribute 'strip'
        if input_string is None:
            return None
        normalized_string = input_string.strip().lower().replace(' ', '_')
        return normalized_string

    def normalize_column_values_numerical(
            self, input_string: str) -> Optional[float]:
        """Normalize column values for numerical data."""
        try:
            normalized_value = float(input_string)
            return normalized_value
        except ValueError:
            return None  # or handle the error as needed

    def normalize_column_values_datetime(
            self,
            input_string: str,
            date_format="%Y-%m-%d") -> Optional[datetime]:
        """Normalize column values for datetime data."""
        try:
            normalized_date = datetime.strptime(input_string, date_format)
            return normalized_date
        except ValueError:
            return None  # or handle the error as needed

    def get_revenue_before_discounts_columns(self) -> list[str]:
        """Get total revenue before discounts ."""
        return [
            "line_total_vat_0_rev_package",
            "line_total_vat_0_rev_ex_vouchers",
            "line_total_vat_0_rev_ex_employees",
            "line_total_vat_0_rev_integrations",
            "line_total_vat_0_rev_mobile",
            "line_total_vat_0_rev_addon",
            "line_total_vat_0_rev_trx"
        ]

    def get_revenue_after_discounts_columns(self) -> list[str]:
        """Get total revenue after discounts ."""
        return [
            "line_total_discounted_vat_0_rev_package",
            "line_total_discounted_vat_0_rev_ex_vouchers",
            "line_total_discounted_vat_0_rev_ex_employees",
            "line_total_discounted_vat_0_rev_integrations",
            "line_total_discounted_vat_0_rev_mobile",
            "line_total_discounted_vat_0_rev_addon",
            "line_total_discounted_vat_0_rev_trx"
        ]

    def discount_calculations_from_revenue_columns(self, indf: pd.DataFrame) -> pd.DataFrame:
        """Calculate discounts from revenue columns."""
        before_discount_cols = self.get_revenue_before_discounts_columns()
        after_discount_cols = self.get_revenue_after_discounts_columns()

        discount_data = pd.DataFrame()

        for before_col, after_col in zip(
                before_discount_cols, after_discount_cols):
            discount_col_name = before_col.replace(
                "line_total_vat_0_rev", "discount_amount")
            discount_data[discount_col_name] = (
                indf[before_col] -
                indf[after_col]
            )

        return discount_data

    def sum_discount_columns(self, indf: pd.DataFrame) -> pd.Series:
        """Sum discount columns to get total discount."""
        discount_data = self.discount_calculations_from_revenue_columns(indf)
        total_discount = discount_data.sum(axis="columns")
        return total_discount

    def get_addon_usage_columns(self) -> list[str]:
        """Get add-on usage columns ."""
        return [
            "add_api",
            "add_bank_account",
            "add_contract_invoicing",
            "add_cust_invoice",
            "add_ext_dimensions",
            "add_inventory",
            "add_junior",
            "add_mobile",
            "add_sftp",
            "mobile_user_count"
        ]

    def get_usage_metrics_columns(self) -> list[str]:
        """
        Get the list of column names for usage metrics.

        This method returns a predefined list of column names that represent various
        usage metrics related to financial records and vouchers. The metrics include
        monthly usage counts, means, and sums for total records and specific voucher types
        (SI/PI vouchers), as well as salary-related record counts.

        Returns
        -------
        list[str]
            A list of column names containing the following metrics:
            - total_records_months_used: Number of months with total records
            - total_records_mean: Average of total records
            - total_records_sum: Sum of total records
            - total_SI_PI_vouchers_months_used: Number of months with SI/PI vouchers
            - total_SI_PI_vouchers_mean: Average of SI/PI vouchers
            - total_SI_PI_vouchers_sum: Sum of SI/PI vouchers
            - record_count_salary_months_used: Number of months with salary records
            - record_count_salary_mean: Average of salary record counts
        """

        return [
            "total_records_months_used",
            "total_records_mean",
            "total_records_sum",
            "total_SI_PI_vouchers_months_used",
            "total_SI_PI_vouchers_mean",
            "total_SI_PI_vouchers_sum",
            "record_count_salary_months_used",
            "record_count_salary_mean"
        ]

    def get_customer_profile_columns(self) -> list[str]:
        """Get customer profile columns ."""
        customer_cols = [
            "id",
            "package",
            "accounting_office_id",
            "company_type_label",
            "tol_1_eng",
            "tol_2_eng",
            "headcount_class",
            "revenue_class",
            "ao_revenue_class",
            "ao_headcount_class"
        ]
        # return only those columns which are not ids.
        return [
            _ for _ in customer_cols if not (
                _.endswith("_id") or _ == "id")]

    def get_predictor_columns(self) -> list[str]:
        """
        Get all predictor variable columns used in the model.

        This method combines all feature columns from customer profiles, usage metrics,
        and addon usage into a single list that represents all predictor variables
        available for modeling.

        Returns:
            list[str]: A complete list of column names for all predictor variables,
                including customer profile features, usage metrics, and addon usage features.

        Example:
            >>> model = ModelStudy_FR()
            >>> predictors = model.get_predictor_columns()
            >>> len(predictors)
            45  # Total number of predictor variables
        """
        """Get all predictor variable columns ."""
        return (
            self.get_customer_profile_columns() +
            self.get_usage_metrics_columns() +
            self.get_addon_usage_columns()
        )

    def get_outcome_columns(self) -> List[str]:
        """Get outcome variable column ."""
        return self.get_revenue_before_discounts_columns()

    def log_transform_column(
            self,
            indf: pd.DataFrame,
            column_name: str) -> pd.Series:
        """Apply logarithmic transformation to values in a specified column.

        This function performs log(x+1) transformation on non-null, non-negative values
        in the specified column while preserving null values and negative numbers unchanged.

        Args:
            indf (pd.DataFrame): Input DataFrame containing the column to transform
            column_name (str): Name of the column to apply log transformation to

        Returns:
            pd.Series: Series containing the transformed values

        Examples:
            >>> df = pd.DataFrame({'A': [0, 1, 2, -1, np.nan]})
            >>> log_transform_column(df, 'A')
            0    0.000000
            1    0.693147
            2    1.098612
            3   -1.000000
            4         NaN
            Name: A, dtype: float64
        """
        """Apply log transformation to a specified column."""
        print(f"Applying log transformation to column: {column_name}")
        transformed_column = indf[column_name].apply(
            lambda x: np.log(x + 1) if pd.notnull(x) and x >= 0 else x,
        )
        return transformed_column

    def treat_predictor_columns(self) -> pd.DataFrame:
        """Normalize predictor columns in a copy of self.customer_data and return the transformed DataFrame.

        This method:
        - Creates a shallow copy of self.customer_data and applies column-wise normalization only to
            columns that appear in the list returned by self.get_all_predictor_variables().
        - Detects column types using pandas' type-checking utilities and dispatches normalization as follows:
                * String/object dtypes -> self.normalize_column_values_category
                * Numeric dtypes -> self.normalize_column_values_numerical
                * Datetime dtypes -> self.normalize_column_values_datetime
        - Leaves the original self.customer_data unmodified; the normalized DataFrame is returned.

        Behavior details:
        - If a predictor listed by get_all_predictor_variables() is missing from self.customer_data,
            the method logs a message (prints) and skips that predictor.
        - If a predictor column has a dtype that is not recognized by the three supported categories
            (string/object, numeric, datetime), the method logs a message (prints) and skips normalization
            for that column.
        - Any errors raised by the normalization helper methods (normalize_column_values_category,
            normalize_column_values_numerical, normalize_column_values_datetime) or by pandas operations
            will propagate to the caller.

        Parameters
        ----------
        self : object
                Instance expected to provide:
                - self.customer_data : pandas.DataFrame containing the raw data.
                - self.get_all_predictor_variables() -> Iterable[str] returning predictor column names.
                - Normalization callables: self.normalize_column_values_category, self.normalize_column_values_numerical,
                    self.normalize_column_values_datetime.

        Returns
        -------
        pandas.DataFrame
                A copy of self.customer_data with predictor columns normalized in place according to their
                detected data types. Non-predictor columns are left unchanged.

        Notes
        -----
        - Normalization helper methods should be implemented to handle missing values (NaN) and
            return values compatible with the original column dtype where appropriate.
        - The dtype checks rely on pandas.api.types; columns with ambiguous or mixed dtypes may require
            prior cleaning to ensure predictable behavior.
        """
        outdf = self.customer_data.copy()
        predictor_cols = self.get_predictor_columns()

        for col in predictor_cols:
            if col not in self.customer_data.columns:
                print(f"Column '{col}' not found in data. Skipping.")
                continue

            if col in ['headcount_class', 'revenue_class',
                       'ao_revenue_class', 'ao_headcount_class']:
                # Skip ordinal variables for now
                print(f"Skipping ordinal column: {col}")
                continue

            if pd.api.types.is_string_dtype(
                self.customer_data[col]
            ) or pd.api.types.is_object_dtype(
                self.customer_data[col]
            ):
                # Normalize categorical string columns
                print(f"Normalizing categorical column: {col}")
                outdf[col] = outdf[col].apply(
                    self.normalize_column_values_category)
            elif pd.api.types.is_numeric_dtype(outdf[col]):
                # Normalize numerical columns
                print(f"Normalizing numerical column: {col}")
                outdf[col] = outdf[col].apply(
                    self.normalize_column_values_numerical)
            elif pd.api.types.is_datetime64_any_dtype(outdf[col]):
                # Normalize datetime columns
                print(f"Normalizing datetime column: {col}")
                outdf[col] = outdf[col].apply(
                    self.normalize_column_values_datetime)
            else:
                print(
                    f"Column '{col}' has unsupported data type {
                        col.__class__}. Skipping normalization.")
        return outdf

    def keep_nominal_predictors_copy(self) -> List[str]:
        """A subset of all the predictor columns are categorical (not necessarily having string dtype)."""
        return [
            "company_type_label",
            "tol_1_eng",
            "tol_2_eng",
            "add_api",
            "add_bank_account",
            "add_contract_invoicing",
            "add_cust_invoice",
            "add_ext_dimensions",
            "add_inventory",
            "add_junior",
            "add_mobile",
            "add_sftp",
        ]

    def keep_nominal_predictors(self) -> List[str]:
        """A subset of all the predictor columns are categorical (not necessarily having string dtype)."""
        return [
            "company_type_label",
            "tol_1_eng",
            "tol_2_eng",
        ]

    def map_ordinal_variables_and_category_order(
            self) -> Dict[str, Dict[str, int]]:
        """Maps ordinal variables and their categories to numeric values.

        This function provides a mapping dictionary that assigns numeric values to different
        categorical variables used in the model. The mapping includes:
        - Package levels (package_1 through package_8)
        - Headcount classes (ranging from empty to +1000 employees)
        - Revenue classes (ranging from empty to +100M)
        - AO (Accountor Online) revenue classes
        - AO headcount classes

            Dict[str, Dict[str, int]]: A nested dictionary where:
                - Outer keys represent variable names ('package', 'headcount_class', etc.)
                - Inner dictionaries map category labels to numeric values
                - Higher numeric values generally represent larger/higher categories

        Example:
            {
                "package": {'package_1': 0, 'package_2': 1, ...},
                "headcount_class": {'': 0, 'Unknown': 1, ...},
                ...
        """
        return {  # Use curly braces and colons
            "package": {'package_1': 0,
                        'package_2': 1,
                        'package_3': 2,
                        'package_4': 3,
                        'package_5': 4,
                        'package_6': 5,
                        'package_7': 6,
                        'package_8': 7},
            "headcount_class": {
                '': 0,
                'Unknown': 1,
                '1': 2,
                '2 - 4': 3,
                '5 - 9': 4,
                '10 - 19': 5,
                '20 - 49': 6,
                '50 - 99': 7,
                '100 - 249': 8,
                '250 - 499': 9,
                '500 - 999': 10,
                '+1000': 11},
            "revenue_class": {
                '': 0,
                'Unknown': 1,
                '0 - 0.2M': 2,
                '0.2 - 0.4M': 3,
                '0.4 - 1M': 4,
                '1 - 2M': 5,
                '2 - 10M': 6,
                '10 - 20M': 7,
                '20 - 50M': 8,
                '50 - 100M': 9,
                '+100M': 10},
            "ao_revenue_class": {
                '': 0,
                'Unknown': 1,
                '0.4 - 1M': 2,
                '0.2 - 0.4M': 3,
                '0 - 0.2M': 4,
                '10 - 20M': 5,
                '1 - 2M': 6,
                '2 - 10M': 7,
                '50 - 100M': 8,
                '20 - 50M': 9, },
            "ao_headcount_class": {
                '': 0,
                '1': 1,
                '2 - 4': 2,
                '5 - 9': 3,
                '10 - 19': 4,
                '20 - 49': 5,
                '50 - 99': 6,
                '100 - 249': 7,
                '250 - 499': 8,
                '500 - 999': 9,
            }
        }

    def tol_2_remaps(self) -> Dict[str, str]:
        """Returns a dictionary mapping of industry classification codes to simplified names.

        This function provides a mapping between detailed industry classification names (prefixed with 'tol_2_eng_')
        and their simplified renamed versions (prefixed with 'tol_2_RENAMED_'). The mapping covers 75 different
        industry classifications including manufacturing, services, trade, and other economic activities.

            Dict[str, str]: A dictionary where:
                - keys: Original detailed industry names (e.g., 'tol_2_eng_air_transport')
                - values: Simplified renamed versions (e.g., 'tol_2_RENAMED_5')
        """
        return {
            "tol_2_eng_activities_auxiliary_to_financial_services_and_insurance_activities": "tol_2_RENAMED_1",
            "tol_2_eng_activities_of_head_offices__management_consultancy_activities": "tol_2_RENAMED_2",
            "tol_2_eng_activities_of_membership_organisations": "tol_2_RENAMED_3",
            "tol_2_eng_advertising_and_market_research": "tol_2_RENAMED_4",
            "tol_2_eng_air_transport": "tol_2_RENAMED_5",
            "tol_2_eng_architectural_and_engineering_activities__technical_testing_and_analysis": "tol_2_RENAMED_6",
            "tol_2_eng_civil_engineering": "tol_2_RENAMED_7",
            "tol_2_eng_computer_programming__consultancy_and_related_activities": "tol_2_RENAMED_8",
            "tol_2_eng_construction_of_buildings": "tol_2_RENAMED_9",
            "tol_2_eng_creative__arts_and_entertainment_activities": "tol_2_RENAMED_10",
            "tol_2_eng_crop_and_animal_production__hunting_and_related_service_activities": "tol_2_RENAMED_11",
            "tol_2_eng_education": "tol_2_RENAMED_12",
            "tol_2_eng_electricity__gas__steam_and_air_conditioning_supply": "tol_2_RENAMED_13",
            "tol_2_eng_employment_activities": "tol_2_RENAMED_14",
            "tol_2_eng_financial_service_activities__except_insurance_and_pension_funding": "tol_2_RENAMED_15",
            "tol_2_eng_fishing_and_aquaculture": "tol_2_RENAMED_16",
            "tol_2_eng_food_and_beverage_service_activities": "tol_2_RENAMED_17",
            "tol_2_eng_forestry_and_logging": "tol_2_RENAMED_18",
            "tol_2_eng_human_health_activities": "tol_2_RENAMED_19",
            "tol_2_eng_industry_unknown": "tol_2_RENAMED_20",
            "tol_2_eng_information_service_activities": "tol_2_RENAMED_21",
            "tol_2_eng_insurance__reinsurance_and_pension_funding__except_compulsory_social_security": "tol_2_RENAMED_22",
            "tol_2_eng_land_transport_and_transport_via_pipelines": "tol_2_RENAMED_23",
            "tol_2_eng_legal_and_accounting_activities": "tol_2_RENAMED_24",
            "tol_2_eng_libraries__archives__museums_and_other_cultural_activities": "tol_2_RENAMED_25",
            "tol_2_eng_manufacture_of_beverages": "tol_2_RENAMED_26",
            "tol_2_eng_manufacture_of_chemicals_and_chemical_products": "tol_2_RENAMED_27",
            "tol_2_eng_manufacture_of_computer__electronic_and_optical_products": "tol_2_RENAMED_28",
            "tol_2_eng_manufacture_of_electrical_equipment": "tol_2_RENAMED_29",
            "tol_2_eng_manufacture_of_fabricated_metal_products__except_machinery_and_equipment": "tol_2_RENAMED_30",
            "tol_2_eng_manufacture_of_food_products": "tol_2_RENAMED_31",
            "tol_2_eng_manufacture_of_furniture": "tol_2_RENAMED_32",
            "tol_2_eng_manufacture_of_leather_and_related_products": "tol_2_RENAMED_33",
            "tol_2_eng_manufacture_of_machinery_and_equipment_n.e.c.": "tol_2_RENAMED_34",
            "tol_2_eng_manufacture_of_motor_vehicles__trailers_and_semi_trailers": "tol_2_RENAMED_35",
            "tol_2_eng_manufacture_of_other_non_metallic_mineral_products": "tol_2_RENAMED_36",
            "tol_2_eng_manufacture_of_other_transport_equipment": "tol_2_RENAMED_37",
            "tol_2_eng_manufacture_of_paper_and_paper_products": "tol_2_RENAMED_38",
            "tol_2_eng_manufacture_of_rubber_and_plastic_products": "tol_2_RENAMED_39",
            "tol_2_eng_manufacture_of_textiles": "tol_2_RENAMED_40",
            "tol_2_eng_manufacture_of_wearing_apparel": "tol_2_RENAMED_41",
            "tol_2_eng_manufacture_of_wood_and_of_products_of_wood_and_cork__except_furniture__manufacture_of_articles_of_straw_and_plaiting_materials": "tol_2_RENAMED_42",
            "tol_2_eng_mining_support_service_activities": "tol_2_RENAMED_43",
            "tol_2_eng_motion_picture__video_and_television_programme_production__sound_recording_and_music_publishing_activities": "tol_2_RENAMED_44",
            "tol_2_eng_office_administrative__office_support_and_other_business_support_activities": "tol_2_RENAMED_45",
            "tol_2_eng_other_manufacturing": "tol_2_RENAMED_46",
            "tol_2_eng_other_mining_and_quarrying": "tol_2_RENAMED_47",
            "tol_2_eng_other_personal_service_activities": "tol_2_RENAMED_48",
            "tol_2_eng_other_professional__scientific_and_technical_activities": "tol_2_RENAMED_49",
            "tol_2_eng_postal_and_courier_activities": "tol_2_RENAMED_50",
            "tol_2_eng_printing_and_reproduction_of_recorded_media": "tol_2_RENAMED_51",
            "tol_2_eng_public_admin_social_insurance": "tol_2_RENAMED_52",
            "tol_2_eng_publishing_activities": "tol_2_RENAMED_53",
            "tol_2_eng_real_estate_activities": "tol_2_RENAMED_54",
            "tol_2_eng_rental_and_leasing_activities": "tol_2_RENAMED_55",
            "tol_2_eng_repair_and_installation_of_machinery_and_equipment": "tol_2_RENAMED_56",
            "tol_2_eng_repair_of_computers_and_personal_and_household_goods": "tol_2_RENAMED_57",
            "tol_2_eng_residential_care_activities": "tol_2_RENAMED_58",
            "tol_2_eng_retail_trade__except_of_motor_vehicles_and_motorcycles": "tol_2_RENAMED_59",
            "tol_2_eng_scientific_research_and_development": "tol_2_RENAMED_60",
            "tol_2_eng_security_and_investigation_activities": "tol_2_RENAMED_61",
            "tol_2_eng_services_to_buildings_and_landscape_activities": "tol_2_RENAMED_62",
            "tol_2_eng_sewerage": "tol_2_RENAMED_63",
            "tol_2_eng_social_work_activities_without_accommodation": "tol_2_RENAMED_64",
            "tol_2_eng_specialised_construction_activities": "tol_2_RENAMED_65",
            "tol_2_eng_sports_activities_and_amusement_and_recreation_activities": "tol_2_RENAMED_66",
            "tol_2_eng_telecommunications": "tol_2_RENAMED_67",
            "tol_2_eng_travel_agency__tour_operator_and_other_reservation_service_and_related_activities": "tol_2_RENAMED_68",
            "tol_2_eng_veterinary_activities": "tol_2_RENAMED_69",
            "tol_2_eng_warehousing_and_support_activities_for_transportation": "tol_2_RENAMED_70",
            "tol_2_eng_waste_collection__treatment_and_disposal_activities__materials_recovery": "tol_2_RENAMED_71",
            "tol_2_eng_water_collection__treatment_and_supply": "tol_2_RENAMED_72",
            "tol_2_eng_water_transport": "tol_2_RENAMED_73",
            "tol_2_eng_wholesale_and_retail_trade_and_repair_of_motor_vehicles_and_motorcycles": "tol_2_RENAMED_74",
            "tol_2_eng_wholesale_trade__except_of_motor_vehicles_and_motorcycles": "tol_2_RENAMED_75",
        }

    def one_hot_encode_nominal_predictors(
            self, indf: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical predictor columns.

        This method performs one-hot encoding on categorical (nominal) predictor columns
        in the input DataFrame. It uses pandas get_dummies() function with drop_first=True
        to avoid multicollinearity in the encoded features.

        Args:
            indf (pd.DataFrame): Input DataFrame containing categorical columns to encode

        Returns:
            pd.DataFrame: DataFrame with categorical columns one-hot encoded
                         Original columns are replaced with binary indicator columns

        Notes:
            - Requires keep_nominal_predictors() method to identify categorical columns
            - Uses drop_first=True to avoid the dummy variable trap
            - Encoded columns are cast to integer dtype
        """
        """One-hot encode categorical predictor columns."""
        nominal_cols = self.keep_nominal_predictors()
        print(f"\n\nOne-hot encoding nominal columns: {nominal_cols}")
        outdf = pd.get_dummies(
            indf,
            columns=nominal_cols,
            drop_first=True,
            dtype=int)
        return outdf

    def map_ordinal_predictors(self, indf: pd.DataFrame) -> pd.DataFrame:
        """Maps ordinal categorical variables to their corresponding numeric codes.

        This method applies ordinal encoding to specified categorical columns using predefined mappings.
        The mappings are retrieved from map_ordinal_variables_and_category_order() method.

        Args:
            indf (pd.DataFrame): Input DataFrame containing categorical columns to be encoded.

        Returns:
            pd.DataFrame: DataFrame with mapped ordinal columns. Original values are preserved
                         if no mapping exists.

        Notes:
            - Method makes a copy of input DataFrame to avoid modifying original data
            - Prints warning if specified column in mapping is not found in DataFrame
            - Uses pandas Series.map() for the encoding transformation
            - Skips columns that are in mapping but not in DataFrame
        """
        """map encodings on ordinal columns based on nested dictionary"""
        outdf = indf.copy()
        ordinal_mappings = self.map_ordinal_variables_and_category_order()
        print("\n\n")
        # dict.keys() is iterable; you can also iterate the dict directly
        for map_col in ordinal_mappings.keys():
            if map_col not in outdf.columns:
                print(f"Column '{map_col}' not found in data. Skipping.")
                continue
            print(f"Doing the ordinal mapping of col -- {map_col}")
            mapping = ordinal_mappings[map_col]
            # Use Series.map to convert categories -> codes. Keep originals if
            # unmapped.
            outdf[map_col] = outdf[map_col].map(mapping).astype('category')
        return outdf

    def treat_predictor_column_names(self, incols: List[str]) -> List[str]:
        """Treat predictor column names by removing special characters."""
        outcols = [
            self.remove_special_characters(col) for col in incols]
        return outcols

    def to_log_transform_columns(self) -> List[str]:
        """Get list of columns to log transform."""
        return [
            "total_records_months_used",
            "total_records_mean",
            "total_records_sum",
            "total_SI_PI_vouchers_months_used",
            "total_SI_PI_vouchers_mean",
            "total_SI_PI_vouchers_sum",
            "record_count_salary_months_used",
            "record_count_salary_mean",
            "mobile_user_count",
            # "total_revenue"
        ]

    def log_transform_df(self, indf: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation to numeric columns in a pandas DataFrame.

        This method performs log transformation on all numeric columns in the input DataFrame.
        Non-numeric columns are skipped with a warning message.

        Args:
            indf (pd.DataFrame): Input DataFrame containing predictor columns

        Returns:
            pd.DataFrame: DataFrame with log-transformed numeric columns. Non-numeric
                         columns remain unchanged.

        Note:
            - Creates a copy of input DataFrame to avoid modifying original data
            - Prints warning messages for missing columns or non-numeric columns
            - Uses log_transform_column() method for the actual transformation
        """
        # important note: log only usage metric columns and outcome column; not categorical predictors
        # this method natural-logs usage metric columns and revenue after discount columns only.
        # --------------------------------------------------------------------------------------------
        # WARNING: do not use revenue after discounts columns as predictors in model training.
        # --------------------------------------------------------------------------------------------

        outdf = indf.copy()
        outdf_cols = outdf.columns.tolist()
        to_log_predictors = self.to_log_transform_columns()

        print("\n\n")
        for col in outdf_cols:
            if col not in to_log_predictors:
                print(
                    f"Column '{col}' is not in the list of predictors to log transform. Skipping.")
                continue
            else:
                if pd.api.types.is_numeric_dtype(outdf[col]):
                    outdf[f"{col}_logged"] = self.log_transform_column(
                        outdf, col)
                else:
                    print(
                        f"Predictor column '{col}' is not numeric. Skipping log transformation.")

        return outdf

    def log_transform_outcome(self, indf: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation to the outcome variable.

        This method performs a logarithmic transformation on the outcome variable specified
        in self.outcome_col. The transformation is only applied if the column contains
        numeric data.

        Args:
            indf (pd.DataFrame): Input DataFrame containing the outcome variable to transform.

        Returns:
            pd.DataFrame: DataFrame with log-transformed outcome variable. If the outcome
                         variable is non-numeric, returns original DataFrame unchanged.

        Raises:
            ValueError: If the specified outcome column is not found in the DataFrame.

        Example:
            >>> model = Model(outcome_col='revenue')
            >>> df = pd.DataFrame({'revenue': [100, 200, 300]})
            >>> transformed_df = model.log_transform_outcome(df)

        """

        outdf = indf.copy()

        if self.outcome_col not in outdf.columns:
            raise ValueError(
                f"Outcome column '{self.outcome_col}' not found in data.")

        if pd.api.types.is_numeric_dtype(outdf[self.outcome_col]):
            print()
            print(
                f"Running module: log_transform_outcome on column: {
                    self.outcome_col}")
            outdf[f"{self.outcome_col}_logged"] = self.log_transform_column(
                outdf,
                self.outcome_col)
        else:
            print(
                f"Outcome column '{
                    self.outcome_col}' is not numeric. Skipping log transformation.")

        return outdf

    def get_regression_ready_cols(self) -> List[str]:
        """Get regression-ready predictor columns.

        This method returns a list of predictor column names that are ready for regression modeling.
        It combines the treated predictor column names and the logged versions of usage metric columns.
        Returns:
            List[str]: A list of regression-ready predictor column names.
        """
        treated_cols = self.treat_predictor_column_names(
            self.get_predictor_columns())
        logged_usage_metric_cols = [
            f"{col}_logged" for col in self.get_usage_metrics_columns()]
        regression_ready_cols = treated_cols + logged_usage_metric_cols
        return regression_ready_cols
