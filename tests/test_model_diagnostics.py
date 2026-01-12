import pytest
import os
import pandas as pd
from unittest.mock import patch
from ModelStudy_FinanceRocks.model_diagnostics import ModelDiagnostics


def test_diagnostics_initialization(data_prep_pipeline, test_plot_dir):
    """Test if diagnostics initializes correctly."""
    diagnostics = ModelDiagnostics(
        preprocessor=data_prep_pipeline.preprocessor,
        plot_dir=test_plot_dir
    )
    assert diagnostics.plot_dir == test_plot_dir
    assert diagnostics.reverse_mappings is not None

def test_calculate_vif(model_diagnostics, trained_model_trainer):
    """Test VIF calculation."""
    mod_dsn = trained_model_trainer.get_design_matrix()
    exog_features = trained_model_trainer.get_exog_features()
    X = mod_dsn[exog_features]
    
    vif_df = model_diagnostics.calculate_vif(X)
    assert isinstance(vif_df, pd.DataFrame)
    assert "Feature" in vif_df.columns
    assert "VIF" in vif_df.columns

def test_reverse_ordinal_encoding(model_diagnostics, synthetic_data):
    """Test reversing ordinal encoding."""
    # Create an encoded DF
    df = synthetic_data.copy()
    # Assume we know the mapping
    mapping = {0: 'Level 1', 1: 'Level 2'}
    model_diagnostics.reverse_mappings = {'test_col': mapping}
    
    encoded_df = pd.DataFrame({'test_col': [0, 1, 0, 1]})
    reversed_df = model_diagnostics.reverse_ordinal_predictors(encoded_df)
    
    assert reversed_df['test_col'].iloc[0] == 'Level 1'
    assert reversed_df['test_col'].iloc[1] == 'Level 2'

@patch('matplotlib.pyplot.savefig')
def test_plot_generation(mock_savefig, model_diagnostics, trained_model_trainer):
    """Test if plotting methods call savefig."""
    # We just want to check if the logic flows to the save step
    mod_dsn = trained_model_trainer.get_design_matrix()
    model_results = trained_model_trainer.get_model_results()
    
    # Add residuals for test
    df_with_resid = mod_dsn.copy()
    df_with_resid['Residuals'] = model_results.resid
    
    first_feature = trained_model_trainer.get_exog_features()[0]
    
    # Test one plotting method
    model_diagnostics.scatter_resid_with_predictors(
        df_with_resid.head(10),
        first_feature,
        model_results,
        model_type="Test"
    )
    
    assert mock_savefig.called
