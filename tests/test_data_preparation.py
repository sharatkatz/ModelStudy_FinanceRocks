import pytest
import pandas as pd
import numpy as np
import os
from ModelStudy_FinanceRocks.data_preparation import DataPreparationPipeline

def test_pipeline_initialization(data_prep_pipeline):
    """Test if the pipeline initializes correctly."""
    assert data_prep_pipeline.file_name == "test_customer_data.parquet"
    assert os.path.exists(data_prep_pipeline.file_path)

def test_prepare_modeling_dataset(data_prep_pipeline, parquet_file):
    """Test the full data preparation pipeline."""
    # Run the pipeline
    predictors_df, full_df = data_prep_pipeline.prepare_modeling_dataset()
    
    # Assertions
    assert isinstance(predictors_df, pd.DataFrame)
    assert isinstance(full_df, pd.DataFrame)
    assert not predictors_df.empty
    assert 'total_revenue' in full_df.columns
    
    # Check if log transformations were applied (columns suffixed with _logged)
    log_cols = [col for col in predictors_df.columns if col.endswith('_logged')]
    assert len(log_cols) > 0

def test_ordinal_encoding(data_prep_pipeline, synthetic_data):
    """Test ordinal encoding logic."""
    df = synthetic_data.copy()
    encoded_df = data_prep_pipeline.apply_ordinal_encoding(df)
    
    ordinal_cols = ['headcount_class', 'revenue_class', 'ao_revenue_class', 'ao_headcount_class']
    for col in ordinal_cols:
        # ModelStudy_FR maps these to codes then converts to 'category'
        assert encoded_df[col].dtype.name == 'category'

def test_one_hot_encoding(data_prep_pipeline, synthetic_data):
    """Test one-hot encoding logic."""
    df = synthetic_data.copy()
    # Need to have some specific columns for one-hot encoding
    encoded_df = data_prep_pipeline.apply_one_hot_encoding(df)
    
    # Check for some expected dummy columns
    dummy_cols = [col for col in encoded_df.columns if 'company_type_' in col or 'tol_1_eng_' in col]
    assert len(dummy_cols) > 0
