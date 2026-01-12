import pytest
from ModelStudy_FinanceRocks.model_validation import ModelValidator

@pytest.fixture
def model_validator(trained_model_trainer):
    """Fixture for ModelValidator."""
    return ModelValidator(model_trainer=trained_model_trainer)

def test_validator_initialization(trained_model_trainer):
    """Test if the validator initializes correctly."""
    validator = ModelValidator(model_trainer=trained_model_trainer)
    assert validator.model_trainer == trained_model_trainer

def test_setup_ols_with_cv(model_validator):
    """Test cross-validation run."""
    cv_mean = model_validator.setup_OLS_with_CV(cv_folds=3)
    assert isinstance(cv_mean, float)
    assert hasattr(model_validator, 'ols_cv_scores')
    assert len(model_validator.ols_cv_scores) == 3

def test_feature_stability(model_validator):
    """Test feature stability analysis."""
    model_validator.setup_OLS_with_CV(cv_folds=3)
    assert model_validator.feature_stability is not None
    assert 'mean_coef' in model_validator.feature_stability.columns
    assert 'std_coef' in model_validator.feature_stability.columns
