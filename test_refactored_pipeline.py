#!/usr/bin/python3

"""
Test script for refactored modeling pipeline.

This script tests all 6 modules to ensure they work correctly together.

Author: Sharat Sharma
Date: Jan-2026
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'ModelStudy_FinanceRocks'))

def test_imports():
    """Test that all modules can be imported."""
    print("=" * 80)
    print("TEST 1: Module Imports")
    print("=" * 80)
    
    try:
        from data_preparation import DataPreparationPipeline
        print("✓ data_preparation imported successfully")
    except Exception as e:
        print(f"✗ data_preparation import failed: {e}")
        return False
    
    try:
        from model_training import OLSModelTrainer
        print("✓ model_training imported successfully")
    except Exception as e:
        print(f"✗ model_training import failed: {e}")
        return False
    
    try:
        from model_validation import ModelValidator
        print("✓ model_validation imported successfully")
    except Exception as e:
        print(f"✗ model_validation import failed: {e}")
        return False
    
    try:
        from model_diagnostics import ModelDiagnostics
        print("✓ model_diagnostics imported successfully")
    except Exception as e:
        print(f"✗ model_diagnostics import failed: {e}")
        return False
    
    try:
        from revenue_analysis import RevenueAnalyzer
        print("✓ revenue_analysis imported successfully")
    except Exception as e:
        print(f"✗ revenue_analysis import failed: {e}")
        return False
    
    try:
        from pipeline_orchestrator import ModelingPipeline
        print("✓ pipeline_orchestrator imported successfully")
    except Exception as e:
        print(f"✗ pipeline_orchestrator import failed: {e}")
        return False
    
    print("\n✓ All modules imported successfully!\n")
    return True


def test_pipeline_initialization():
    """Test that the pipeline can be initialized."""
    print("=" * 80)
    print("TEST 2: Pipeline Initialization")
    print("=" * 80)
    
    try:
        from pipeline_orchestrator import ModelingPipeline
        
        pipeline = ModelingPipeline(
            file_path=None,
            file_name="customer_data.parquet",
            log_file="test_pipeline.log"
        )
        print("✓ Pipeline initialized successfully")
        print(f"  - Log file: {pipeline.log_file}")
        print(f"  - Plot directory: {pipeline.plot_dir}")
        print(f"  - Data file: {pipeline.file_name}")
        print()
        return True
        
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test running the full pipeline."""
    print("=" * 80)
    print("TEST 3: Full Pipeline Execution")
    print("=" * 80)
    
    try:
        from pipeline_orchestrator import ModelingPipeline
        
        print("Initializing pipeline...")
        pipeline = ModelingPipeline(log_file="test_full_pipeline.log")
        
        print("Running full pipeline...")
        print("(This may take several minutes...)\n")
        
        results = pipeline.run_full_pipeline(
            exclude_high_vif=True,
            cv_folds=5,
            revenue_top_n=20,
            run_diagnostics=True,
            run_revenue_analysis=True
        )
        
        print("\n" + "=" * 80)
        print("PIPELINE RESULTS SUMMARY")
        print("=" * 80)
        
        for phase, metrics in results.items():
            print(f"\n{phase.upper().replace('_', ' ')}:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        
        print("\n✓ Full pipeline executed successfully!")
        print(f"\nCheck the log file for details: test_full_pipeline.log")
        print(f"Check the plot directory for visualizations: {pipeline.plot_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Full pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_modules():
    """Test each module individually."""
    print("=" * 80)
    print("TEST 4: Individual Module Testing")
    print("=" * 80)
    
    try:
        from data_preparation import DataPreparationPipeline
        from model_training import OLSModelTrainer
        from model_validation import ModelValidator
        
        # Test 4.1: Data Preparation
        print("\n4.1: Testing DataPreparationPipeline...")
        data_prep = DataPreparationPipeline(log_file="test_data_prep.log")
        predictors_df, full_df = data_prep.prepare_modeling_dataset()
        print(f"  ✓ Data prepared: {predictors_df.shape} predictors, {full_df.shape} full dataset")
        
        # Test 4.2: Model Training
        print("\n4.2: Testing OLSModelTrainer...")
        trainer = OLSModelTrainer(data_prep=data_prep)
        trainer.setup_OLS()
        model_results = trainer.get_model_results()
        print(f"  ✓ Model trained: R² = {model_results.rsquared:.4f}")
        
        # Test 4.3: Model Validation
        print("\n4.3: Testing ModelValidator...")
        validator = ModelValidator(model_trainer=trainer)
        cv_score = validator.setup_OLS_with_CV(cv_folds=5)
        print(f"  ✓ Validation complete: CV R² = {cv_score:.4f}")
        
        print("\n✓ All individual modules tested successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Individual module testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("REFACTORED PIPELINE TEST SUITE")
    print("=" * 80)
    print()
    
    results = []
    
    # Test 1: Imports
    results.append(("Module Imports", test_imports()))
    
    # Test 2: Pipeline Initialization
    results.append(("Pipeline Initialization", test_pipeline_initialization()))
    
    # Test 3: Individual Modules
    results.append(("Individual Modules", test_individual_modules()))
    
    # Test 4: Full Pipeline (most comprehensive)
    results.append(("Full Pipeline", test_full_pipeline()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe refactored modules are working correctly.")
        print("You can now use the modular pipeline in your projects.")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 80)
        print("\nPlease review the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
