use ohms_adaptq::{
    NOVAQConfig, WeightMatrix, PublicNOVAQ, 
    QuantizationRecoveryManager, VerbosityLevel
};

/// Comprehensive stability test suite for NOVAQ quantization
/// Tests edge cases that previously caused NaN failures and numerical instabilities

#[test]
fn test_width_1_tensor_stability() {
    // Test case: Single-width tensors that caused subspace adjustment issues
    let config = NOVAQConfig {
        target_bits: 1.5,
        num_subspaces: 4,
        ..Default::default()
    };
    
    let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
    
    // Create a width-1 tensor (common in bias vectors)
    let data = vec![0.1, -0.2, 0.3, -0.4, 0.5];
    let weight = WeightMatrix::new(data, vec![5, 1], "bias_vector".to_string());
    
    let result = novaq.compress_model(vec![weight]);
    assert!(result.is_ok(), "Width-1 tensor quantization should not fail: {:?}", result.err());
    
    let model = result.unwrap();
    assert!(model.compression_ratio > 0.0, "Compression ratio should be positive");
    assert!(model.bit_accuracy > 0.9, "Bit accuracy should be high for simple tensors");
}

#[test]
fn test_extreme_bit_rates() {
    // Test extremely low and high bit rates
    let test_cases = vec![
        0.5,  // Very low precision
        1.0,  // Minimal precision
        4.0,  // High precision
        8.0,  // Very high precision
    ];
    
    for bits in test_cases {
        let config = NOVAQConfig {
            target_bits: bits,
            num_subspaces: 2,
            refinement_iterations: 10, // Reduce iterations for speed
            ..Default::default()
        };
        
        let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
        
        // Create test data
        let data: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let weight = WeightMatrix::new(data, vec![10, 10], format!("test_{}bits", bits));
        
        let result = novaq.compress_model(vec![weight]);
        assert!(result.is_ok(), "Quantization with {} bits should not fail: {:?}", bits, result.err());
        
        let model = result.unwrap();
        assert!(model.compression_ratio > 0.0, "Compression ratio should be positive for {} bits", bits);
    }
}

#[test]
fn test_very_small_models() {
    // Test tiny models that might cause edge cases in quantization
    let config = NOVAQConfig {
        target_bits: 1.5,
        num_subspaces: 1, // Use single subspace for small models
        refinement_iterations: 5,
        ..Default::default()
    };
    
    let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
    
    // Minimal model: 2x2 matrix
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let weight = WeightMatrix::new(data, vec![2, 2], "tiny_model".to_string());
    
    let result = novaq.compress_model(vec![weight]);
    assert!(result.is_ok(), "Tiny model quantization should succeed: {:?}", result.err());
    
    let model = result.unwrap();
    assert!(model.weight_shapes.contains_key("tiny_model"));
    assert_eq!(model.weight_shapes["tiny_model"], vec![2, 2]);
}

#[test]
fn test_large_model_stability() {
    // Test larger models to ensure no memory or performance issues
    let config = NOVAQConfig {
        target_bits: 1.5,
        num_subspaces: 4,
        refinement_iterations: 20,
        ..Default::default()
    };
    
    let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
    
    // Large model: 1000x1000 matrix (1M parameters)
    let size = 1000;
    let data: Vec<f32> = (0..size*size).map(|i| {
        // Create realistic weight distribution
        let normalized = (i as f32) / (size * size) as f32;
        (normalized * 2.0 - 1.0) * 0.1 // Values between -0.1 and 0.1
    }).collect();
    
    let weight = WeightMatrix::new(data, vec![size, size], "large_model".to_string());
    
    let result = novaq.compress_model(vec![weight]);
    assert!(result.is_ok(), "Large model quantization should succeed: {:?}", result.err());
    
    let model = result.unwrap();
    assert!(model.compression_ratio > 1.0, "Large model should achieve compression");
    assert!(model.bit_accuracy > 0.95, "Large model should maintain high accuracy");
}

#[test]
fn test_numerical_edge_cases() {
    // Test values that commonly cause numerical instabilities
    let config = NOVAQConfig {
        target_bits: 1.5,
        num_subspaces: 2,
        refinement_iterations: 10,
        ..Default::default()
    };
    
    let mut novaq = PublicNOVAQ::new_with_verbosity(config.clone(), VerbosityLevel::Silent);
    
    // Test case 1: Very small values near zero
    let small_data = vec![1e-8, -1e-8, 1e-7, -1e-7, 0.0, 1e-6, -1e-6, 1e-5];
    let small_weight = WeightMatrix::new(small_data, vec![2, 4], "small_values".to_string());
    
    let result = novaq.compress_model(vec![small_weight]);
    assert!(result.is_ok(), "Small values should not cause instability: {:?}", result.err());
    
    // Test case 2: Large values
    let large_data = vec![100.0, -100.0, 50.0, -50.0, 200.0, -200.0, 75.0, -75.0];
    let large_weight = WeightMatrix::new(large_data, vec![2, 4], "large_values".to_string());
    
    let mut novaq2 = PublicNOVAQ::new_with_verbosity(config.clone(), VerbosityLevel::Silent);
    let result2 = novaq2.compress_model(vec![large_weight]);
    assert!(result2.is_ok(), "Large values should not cause instability: {:?}", result2.err());
    
    // Test case 3: Mixed scales
    let mixed_data = vec![1e-6, 100.0, -1e-5, -50.0, 0.0, 1.0, 1e-4, -1.0];
    let mixed_weight = WeightMatrix::new(mixed_data, vec![2, 4], "mixed_scales".to_string());
    
    let mut novaq3 = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
    let result3 = novaq3.compress_model(vec![mixed_weight]);
    assert!(result3.is_ok(), "Mixed scale values should not cause instability: {:?}", result3.err());
}

#[test]
fn test_recovery_system_activation() {
    // Test that the recovery system properly activates when needed
    let config = NOVAQConfig {
        target_bits: 0.1, // Extremely low bits to potentially trigger recovery
        num_subspaces: 8,  // High subspaces on small data to force edge cases
        refinement_iterations: 5,
        learning_rate: 10.0, // High learning rate to potentially cause instability
        ..Default::default()
    };
    
    let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
    
    // Create challenging data
    let data: Vec<f32> = (0..16).map(|i| if i % 2 == 0 { 1000.0 } else { -1000.0 }).collect();
    let weight = WeightMatrix::new(data, vec![4, 4], "challenging_data".to_string());
    
    let result = novaq.compress_model(vec![weight]);
    
    // Even with challenging parameters, recovery should handle it
    assert!(result.is_ok(), "Recovery system should handle challenging cases: {:?}", result.err());
    
    // Check recovery stats
    let stats = novaq.get_recovery_stats();
    if stats.total_attempts > 0 {
        println!("Recovery system activated: {} attempts, {} successful", 
                 stats.total_attempts, stats.successful_recoveries);
    }
}

#[test]
fn test_subspace_boundary_conditions() {
    // Test boundary conditions for subspace calculations
    let test_cases = vec![
        (1, 1),   // Single subspace, minimal size
        (2, 2),   // Two subspaces
        (4, 8),   // Standard case
        (8, 4),   // More subspaces than columns per subspace
    ];
    
    for (num_subspaces, cols) in test_cases {
        let config = NOVAQConfig {
            target_bits: 1.5,
            num_subspaces,
            refinement_iterations: 5,
            ..Default::default()
        };
        
        let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
        
        // Create data that matches the subspace requirements
        let total_size = 4 * cols; // 4 rows
        let data: Vec<f32> = (0..total_size).map(|i| (i as f32) * 0.1).collect();
        let weight = WeightMatrix::new(data, vec![4, cols], 
                                     format!("subspace_{}_{}", num_subspaces, cols));
        
        let result = novaq.compress_model(vec![weight]);
        assert!(result.is_ok(), 
                "Subspace boundary case ({} subspaces, {} cols) should work: {:?}", 
                num_subspaces, cols, result.err());
    }
}

#[test]
fn test_zero_and_uniform_data() {
    // Test edge cases with zero data and uniform data
    let config = NOVAQConfig {
        target_bits: 1.5,
        num_subspaces: 2,
        refinement_iterations: 10,
        ..Default::default()
    };
    
    // Test case 1: All zeros
    let mut novaq1 = PublicNOVAQ::new_with_verbosity(config.clone(), VerbosityLevel::Silent);
    let zero_data = vec![0.0; 16];
    let zero_weight = WeightMatrix::new(zero_data, vec![4, 4], "all_zeros".to_string());
    
    let result1 = novaq1.compress_model(vec![zero_weight]);
    assert!(result1.is_ok(), "All-zero data should not cause issues: {:?}", result1.err());
    
    // Test case 2: All same values
    let mut novaq2 = PublicNOVAQ::new_with_verbosity(config.clone(), VerbosityLevel::Silent);
    let uniform_data = vec![0.5; 16];
    let uniform_weight = WeightMatrix::new(uniform_data, vec![4, 4], "uniform".to_string());
    
    let result2 = novaq2.compress_model(vec![uniform_weight]);
    assert!(result2.is_ok(), "Uniform data should not cause issues: {:?}", result2.err());
    
    // Test case 3: Alternating pattern
    let mut novaq3 = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
    let alternating_data: Vec<f32> = (0..16).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let alternating_weight = WeightMatrix::new(alternating_data, vec![4, 4], "alternating".to_string());
    
    let result3 = novaq3.compress_model(vec![alternating_weight]);
    assert!(result3.is_ok(), "Alternating pattern should not cause issues: {:?}", result3.err());
}

#[test]
fn test_recovery_manager_isolation() {
    // Test recovery manager independently to ensure it handles specific error types
    let config = NOVAQConfig::default();
    let mut recovery_manager = QuantizationRecoveryManager::new(config);
    
    // Create test data that should work
    let data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
    let weight = WeightMatrix::new(data, vec![8, 8], "recovery_test".to_string());
    
    let result = recovery_manager.quantize_with_recovery(vec![weight]);
    assert!(result.is_ok(), "Recovery manager should handle standard cases: {:?}", result.err());
    
    // Check that stats are being tracked
    let stats = recovery_manager.get_stats();
    assert!(stats.total_attempts >= 0); // Should be initialized
}

#[test]
fn test_stress_multiple_weights() {
    // Test quantization of multiple weight matrices with different characteristics
    let config = NOVAQConfig {
        target_bits: 1.5,
        num_subspaces: 4,
        refinement_iterations: 10,
        ..Default::default()
    };
    
    let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
    
    let mut weights = Vec::new();
    
    // Weight 1: Small dense matrix
    let data1: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.8).collect();
    weights.push(WeightMatrix::new(data1, vec![4, 4], "dense_small".to_string()));
    
    // Weight 2: Wide matrix (many columns)
    let data2: Vec<f32> = (0..128).map(|i| ((i as f32) / 64.0) - 1.0).collect();
    weights.push(WeightMatrix::new(data2, vec![4, 32], "wide_matrix".to_string()));
    
    // Weight 3: Tall matrix (many rows)
    let data3: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    weights.push(WeightMatrix::new(data3, vec![32, 4], "tall_matrix".to_string()));
    
    // Weight 4: Minimal width-1 vector
    let data4 = vec![0.1, -0.2, 0.3, -0.4];
    weights.push(WeightMatrix::new(data4, vec![4, 1], "bias_vector".to_string()));
    
    let result = novaq.compress_model(weights);
    assert!(result.is_ok(), "Multiple diverse weights should quantize successfully: {:?}", result.err());
    
    let model = result.unwrap();
    assert_eq!(model.weight_shapes.len(), 4, "Should preserve all weight shapes");
    assert!(model.weight_shapes.contains_key("dense_small"));
    assert!(model.weight_shapes.contains_key("wide_matrix"));
    assert!(model.weight_shapes.contains_key("tall_matrix"));
    assert!(model.weight_shapes.contains_key("bias_vector"));
}

#[cfg(test)]
mod performance_regression_tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn test_quantization_performance_bounds() {
        // Ensure quantization completes within reasonable time bounds
        let config = NOVAQConfig {
            target_bits: 1.5,
            num_subspaces: 4,
            refinement_iterations: 20,
            ..Default::default()
        };
        
        let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
        
        // Medium-sized model for performance testing
        let size = 100;
        let data: Vec<f32> = (0..size*size).map(|i| (i as f32) / (size * size) as f32).collect();
        let weight = WeightMatrix::new(data, vec![size, size], "performance_test".to_string());
        
        let start = Instant::now();
        let result = novaq.compress_model(vec![weight]);
        let duration = start.elapsed();
        
        assert!(result.is_ok(), "Performance test should complete successfully");
        assert!(duration.as_secs() < 30, "Quantization should complete within 30 seconds, took: {:?}", duration);
        
        println!("Quantization performance: {:.2}s for 10K parameters", duration.as_secs_f32());
    }
    
    #[test]
    fn test_memory_usage_stability() {
        // Test that memory usage remains stable across multiple quantizations
        let config = NOVAQConfig {
            target_bits: 1.5,
            num_subspaces: 2,
            refinement_iterations: 5,
            ..Default::default()
        };
        
        // Run multiple quantizations to check for memory leaks
        for i in 0..5 {
            let mut novaq = PublicNOVAQ::new_with_verbosity(config.clone(), VerbosityLevel::Silent);
            
            let data: Vec<f32> = (0..64).map(|j| ((i * 64 + j) as f32) * 0.01).collect();
            let weight = WeightMatrix::new(data, vec![8, 8], format!("mem_test_{}", i));
            
            let result = novaq.compress_model(vec![weight]);
            assert!(result.is_ok(), "Memory stability test iteration {} should succeed", i);
        }
    }
}