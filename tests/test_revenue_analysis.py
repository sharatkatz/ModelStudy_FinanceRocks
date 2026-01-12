import pytest
import pandas as pd
from ModelStudy_FinanceRocks.revenue_analysis import RevenueAnalyzer

@pytest.fixture
def revenue_analyzer(trained_model_trainer):
    """Fixture for RevenueAnalyzer."""
    trainer = trained_model_trainer
    return RevenueAnalyzer(
        customer_data=trainer.preprocessor.customer_data,
        model_results=trainer.get_model_results(),
        mod_dsn=trainer.get_design_matrix(),
        exog_features=trainer.get_exog_features(),
        endog_feature=trainer.get_endog_feature()
    )

def test_revenue_analyzer_initialization(trained_model_trainer):
    """Test if analyzer initializes correctly."""
    trainer = trained_model_trainer
    analyzer = RevenueAnalyzer(
        customer_data=trainer.preprocessor.customer_data
    )
    assert analyzer.customer_data is not None

def test_direct_revenue_ranking(revenue_analyzer):
    """Test direct revenue ranking strategy."""
    top_n = 5
    results = revenue_analyzer.analyze_direct_revenue_ranking(top_n=top_n)
    assert len(results) == top_n
    assert 'total_revenue' in results.columns
    # Check if sorted
    assert results['total_revenue'].iloc[0] >= results['total_revenue'].iloc[-1]

def test_revenue_by_segment(revenue_analyzer):
    """Test revenue by segment strategy."""
    results = revenue_analyzer.analyze_revenue_by_segment()
    assert 'package' in results
    assert 'company_type' in results
    assert isinstance(results['package'], pd.DataFrame)

def test_feature_adopters(revenue_analyzer):
    """Test feature adopters analysis."""
    results = revenue_analyzer.analyze_feature_adopters(top_n=5)
    # Our synthetic data has 'add_' columns so this should return results
    if not results.empty:
        assert 'total_revenue' in results.columns
        assert 'addon_count' in results.columns

def test_high_activity_users(revenue_analyzer):
    """Test high activity users strategy."""
    results = revenue_analyzer.analyze_high_activity_users(top_n=5)
    assert isinstance(results, dict)
    # Check for expected usage metric keys from our synthetic data
    assert 'total_records_sum' in results or 'total_SI_PI_vouchers_sum' in results
