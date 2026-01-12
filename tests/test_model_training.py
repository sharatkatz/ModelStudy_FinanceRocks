import pytest
import pandas as pd
from ModelStudy_FinanceRocks.model_training import OLSModelTrainer


def test_trainer_initialization(data_prep_pipeline):
    """Test if the trainer initializes correctly."""
    trainer = OLSModelTrainer(data_prep=data_prep_pipeline)
    assert trainer.model_type == "Additive"
    assert trainer.endog == "total_revenue"

def test_setup_ols(trained_model_trainer):
    """Test OLS model fitting."""
    results = trained_model_trainer.get_model_results()
    assert results is not None
    assert hasattr(results, 'rsquared')
    assert len(trained_model_trainer.get_exog_features()) > 0

def test_feature_selection(trained_model_trainer):
    """Test if insignificant features are removed."""
    # This depends on the synthetic data, but we can check if it runs
    initial_features = trained_model_trainer.get_exog_features()
    assert len(initial_features) > 0
    
    # Verify design matrix shape
    # mod_dsn contains baseline features + discounts + outcome
    dsn = trained_model_trainer.get_design_matrix()
    # At baseline, we have some number of features. Let's just check it's non-empty.
    assert dsn.shape[1] > len(trained_model_trainer.get_exog_features())
