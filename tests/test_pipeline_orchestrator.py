import pytest
import os
from ModelStudy_FinanceRocks.pipeline_orchestrator import ModelingPipeline

def test_pipeline_orchestration(test_data_dir, parquet_file, test_plot_dir):
    """Test the full pipeline orchestration."""
    # Define a temporary log file path
    test_log_file = os.path.join(test_data_dir, "test_modeling_pipeline.log")
    
    pipeline = ModelingPipeline(
        file_path=test_data_dir,
        file_name="test_customer_data.parquet",
        log_file=test_log_file
    )
    # Redirect plot dir to our test plot dir
    pipeline.plot_dir = test_plot_dir
    
    # Run full pipeline with small parameters for speed
    results = pipeline.run_full_pipeline(
        exclude_high_vif=True,
        cv_folds=2,
        revenue_top_n=5,
        run_diagnostics=True,
        run_revenue_analysis=True
    )
    
    print("\nPipeline Results Summary:")
    import pprint
    pprint.pprint(results)
    
    # Assertions on results summary
    assert results['data_preparation']['completed'] is True, "Data preparation failed"
    assert results['model_training']['completed'] is True, "Model training failed"
    assert results['model_validation']['completed'] is True, "Model validation failed"
    assert results['model_diagnostics']['completed'] is True, "Model diagnostics failed"
    assert results['revenue_analysis']['completed'] is True, "Revenue analysis failed"
    
    # Check if log file was created
    assert os.path.exists(test_log_file), f"Log file not found at {test_log_file}"
