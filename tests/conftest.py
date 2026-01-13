import pytest
import pandas as pd
import numpy as np
import os
import shutil
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from ModelStudy_FinanceRocks.data_preparation import DataPreparationPipeline
from ModelStudy_FinanceRocks.ModelStudy_FR import PreProcessor

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    tmp_dir = tmp_path_factory.mktemp("data")
    return str(tmp_dir)

@pytest.fixture(scope="session")
def test_plot_dir(tmp_path_factory):
    """Create a temporary directory for test plots."""
    tmp_dir = tmp_path_factory.mktemp("plots")
    return str(tmp_dir)

@pytest.fixture(scope="session")
def synthetic_data():
    """Generate synthetic customer data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'id': range(n_samples),
        'package': np.random.choice(['package_1', 'package_2', 'package_3'], n_samples),
        'accounting_office_id': np.random.randint(1, 10, n_samples),
        'company_type_label': np.random.choice(['LTD', 'Sole Prop', 'Partnership'], n_samples),
        'tol_1_eng': np.random.choice(['Industry A', 'Industry B', 'Industry C'], n_samples),
        'tol_2_eng': np.random.choice(['Sub-Industry X', 'Sub-Industry Y'], n_samples),
        'headcount_class': np.random.choice(['1', '2 - 4', '5 - 9', '10 - 19'], n_samples),
        'revenue_class': np.random.choice(['0 - 0.2M', '0.2 - 0.4M', '0.4 - 1M'], n_samples),
        'ao_revenue_class': np.random.choice(['0 - 0.2M', '0.2 - 0.4M', '0.4 - 1M'], n_samples),
        'ao_headcount_class': np.random.choice(['1', '2 - 4', '5 - 9', '10 - 19'], n_samples),
        
        # Usage Metrics
        'total_records_months_used': np.random.randint(1, 13, n_samples),
        'total_records_mean': np.random.uniform(10, 100, n_samples),
        'total_records_sum': np.random.uniform(120, 1200, n_samples),
        'total_SI_PI_vouchers_months_used': np.random.randint(1, 13, n_samples),
        'total_SI_PI_vouchers_mean': np.random.uniform(5, 50, n_samples),
        'total_SI_PI_vouchers_sum': np.random.uniform(60, 600, n_samples),
        'record_count_salary_months_used': np.random.randint(0, 13, n_samples),
        'record_count_salary_mean': np.random.uniform(0, 20, n_samples),
        
        # Add-ons
        'add_api': np.random.randint(0, 2, n_samples),
        'add_bank_account': np.random.randint(0, 2, n_samples),
        'add_contract_invoicing': np.random.randint(0, 2, n_samples),
        'add_cust_invoice': np.random.randint(0, 2, n_samples),
        'add_ext_dimensions': np.random.randint(0, 2, n_samples),
        'add_inventory': np.random.randint(0, 2, n_samples),
        'add_junior': np.random.randint(0, 2, n_samples),
        'add_mobile': np.random.randint(0, 2, n_samples),
        'add_sftp': np.random.randint(0, 2, n_samples),
        'mobile_user_count': np.random.randint(0, 10, n_samples),
        
        # Revenue Before Discounts
        'line_total_vat_0_rev_package': np.random.uniform(100, 500, n_samples),
        'line_total_vat_0_rev_ex_vouchers': np.random.uniform(0, 200, n_samples),
        'line_total_vat_0_rev_ex_employees': np.random.uniform(0, 150, n_samples),
        'line_total_vat_0_rev_integrations': np.random.uniform(0, 100, n_samples),
        'line_total_vat_0_rev_mobile': np.random.uniform(0, 50, n_samples),
        'line_total_vat_0_rev_addon': np.random.uniform(0, 80, n_samples),
        'line_total_vat_0_rev_trx': np.random.uniform(0, 40, n_samples),
        
        # Revenue After Discounts
        'line_total_discounted_vat_0_rev_package': np.random.uniform(80, 450, n_samples),
        'line_total_discounted_vat_0_rev_ex_vouchers': np.random.uniform(0, 180, n_samples),
        'line_total_discounted_vat_0_rev_ex_employees': np.random.uniform(0, 130, n_samples),
        'line_total_discounted_vat_0_rev_integrations': np.random.uniform(0, 90, n_samples),
        'line_total_discounted_vat_0_rev_mobile': np.random.uniform(0, 45, n_samples),
        'line_total_discounted_vat_0_rev_addon': np.random.uniform(0, 75, n_samples),
        'line_total_discounted_vat_0_rev_trx': np.random.uniform(0, 35, n_samples),
    }
    
    df = pd.DataFrame(data)
    # Ensure some categorical columns have 'LTD' for specific tests if needed
    return df

@pytest.fixture(scope="session")
def parquet_file(synthetic_data, test_data_dir):
    """Save synthetic data to a parquet file."""
    file_path = os.path.join(test_data_dir, "test_customer_data.parquet")
    synthetic_data.to_parquet(file_path)
    return file_path

@pytest.fixture
def data_prep_pipeline(test_data_dir, test_plot_dir, parquet_file):
    """Fixture for DataPreparationPipeline. Depends on parquet_file to ensure it exists."""
    pipeline = DataPreparationPipeline(
        file_path=test_data_dir,
        file_name="test_customer_data.parquet",
        plot_dir=test_plot_dir
    )
    return pipeline

@pytest.fixture
def trained_model_trainer(data_prep_pipeline, parquet_file):
    """Fixture for OLSModelTrainer with an initial fit."""
    from ModelStudy_FinanceRocks.model_training import OLSModelTrainer
    data_prep_pipeline.prepare_modeling_dataset()
    trainer = OLSModelTrainer(data_prep=data_prep_pipeline)
    trainer.setup_OLS()
    return trainer

@pytest.fixture
def model_diagnostics(data_prep_pipeline, test_plot_dir):
    """Fixture for ModelDiagnostics."""
    from ModelStudy_FinanceRocks.model_diagnostics import ModelDiagnostics
    return ModelDiagnostics(
        preprocessor=data_prep_pipeline.preprocessor,
        plot_dir=test_plot_dir
    )
