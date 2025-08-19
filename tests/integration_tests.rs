use ohms_adaptq::{
    NOVAQEngine, NOVAQConfig, WeightMatrix, PublicNOVAQ, VerbosityLevel
};

/// Integration tests for complete NOVAQ quantization pipeline
/// Tests end-to-end functionality with real-world scenarios

#[test]
fn test_complete_quantization_pipeline() {
    // Test the complete pipeline from model input to compressed output
    let config = NOVAQConfig {
        target_bits: 1.5,
        num_subspaces: 4,
        refinement_iterations: 25,
        ..Default::default()
    };
    
    let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Standard);
    
    // Create realistic model weights (simulating a small transformer layer)
    let mut weights = Vec::new();
    
    // Attention weights
    let attention_size = 64;
    let attention_data: Vec<f32> = (0..attention_size*attention_size).map(|i| {
        let row = i / attention_size;
        let col = i % attention_size;
        if row == col { 1.0 } else { (i as f32 * 0.01).sin() * 0.1 }
    }).collect();
    weights.push(WeightMatrix::new(attention_data, vec![attention_size, attention_size], "attention_weights".to_string()));
    
    // Feed-forward weights
    let ff_data: Vec<f32> = (0..64*128).map(|i| {
        ((i as f32) / 1000.0).tanh() * 0.5
    }).collect();
    weights.push(WeightMatrix::new(ff_data, vec![64, 128], "feedforward_weights".to_string()));
    
    // Bias vectors
    let bias_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.001 - 0.032).collect();
    weights.push(WeightMatrix::new(bias_data, vec![64, 1], "bias_vector".to_string()));
    
    // Perform quantization
    let result = novaq.compress_model(weights);
    assert!(result.is_ok(), "Complete pipeline should succeed: {:?}", result.err());
    
    let model = result.unwrap();
    
    // Verify model properties
    assert_eq!(model.weight_shapes.len(), 3, "Should preserve all weight matrices");
    assert!(model.compression_ratio > 2.0, "Should achieve good compression");
    assert!(model.bit_accuracy > 0.95, "Should maintain high accuracy");
    
    // Verify specific shapes are preserved
    assert_eq!(model.weight_shapes["attention_weights"], vec![64, 64]);
    assert_eq!(model.weight_shapes["feedforward_weights"], vec![64, 128]);
    assert_eq!(model.weight_shapes["bias_vector"], vec![64, 1]);
    
    println!("✅ Complete pipeline test: {:.1}x compression, {:.3} accuracy", 
             model.compression_ratio, model.bit_accuracy);
}

#[test]
fn test_model_reconstruction_accuracy() {
    // Test that we can reconstruct weights with acceptable accuracy
    let config = NOVAQConfig {
        target_bits: 2.0, // Higher bits for better reconstruction
        num_subspaces: 4,
        refinement_iterations: 30,
        ..Default::default()
    };
    
    let mut novaq = PublicNOVAQ::new_with_verbosity(config.clone(), VerbosityLevel::Silent);
    
    // Create test data with known patterns
    let size = 32;
    let original_data: Vec<f32> = (0..size*size).map(|i| {
        let x = (i % size) as f32 / size as f32;
        let y = (i / size) as f32 / size as f32;
        (x * 3.14159).sin() * (y * 3.14159).cos() * 0.5
    }).collect();
    
    let original_weight = WeightMatrix::new(original_data.clone(), vec![size, size], "reconstruction_test".to_string());
    
    // Quantize the model
    let result = novaq.compress_model(vec![original_weight.clone()]);
    assert!(result.is_ok(), "Quantization should succeed");
    
    let model = result.unwrap();
    
    // Test reconstruction
    let engine = NOVAQEngine::new(config);
    let reconstructed_result = engine.reconstruct_weights(&model, "reconstruction_test");
    assert!(reconstructed_result.is_ok(), "Reconstruction should succeed");
    
    let reconstructed = reconstructed_result.unwrap();
    
    // Verify reconstruction accuracy
    assert_eq!(reconstructed.data.len(), original_data.len(), "Reconstructed size should match");
    assert_eq!(reconstructed.shape, vec![size, size], "Reconstructed shape should match");
    
    // Calculate reconstruction error
    let mse: f32 = original_data.iter()
        .zip(reconstructed.data.iter())
        .map(|(orig, recon)| (orig - recon).powi(2))
        .sum::<f32>() / original_data.len() as f32;
    
    let rmse = mse.sqrt();
    assert!(rmse < 0.1, "Reconstruction RMSE should be low: {:.4}", rmse);
    
    println!("✅ Reconstruction test: RMSE = {:.4}", rmse);
}

#[test]
fn test_different_model_architectures() {
    // Test quantization with different neural network architectures
    let architectures = vec![
        ("linear", vec![(100, 50), (50, 10)]),
        ("conv_like", vec![(32, 64), (64, 128), (128, 256)]),
        ("attention", vec![(512, 512), (512, 2048), (2048, 512)]),
        ("embedding", vec![(10000, 256)]),
    ];
    
    for (arch_name, layers) in architectures {
        let config = NOVAQConfig {
            target_bits: 1.5,
            num_subspaces: 4,
            refinement_iterations: 15,
            ..Default::default()
        };
        
        let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
        let mut weights = Vec::new();
        
        for (layer_idx, (rows, cols)) in layers.iter().enumerate() {
            let total_params = rows * cols;
            
            // Generate realistic weight distributions
            let data: Vec<f32> = (0..total_params).map(|i| {
                let fan_in = *cols as f32;
                let fan_out = *rows as f32;
                let limit = (6.0 / (fan_in + fan_out)).sqrt(); // Xavier initialization
                
                let normalized = (i as f32) / total_params as f32;
                ((normalized * 2.0 - 1.0) * limit) + (layer_idx as f32 * 0.001)
            }).collect();
            
            weights.push(WeightMatrix::new(data, vec![*rows, *cols], 
                                         format!("{}_{}", arch_name, layer_idx)));
        }
        
        let result = novaq.compress_model(weights);
        assert!(result.is_ok(), "Architecture {} should quantize successfully: {:?}", 
                arch_name, result.err());
        
        let model = result.unwrap();
        assert_eq!(model.weight_shapes.len(), layers.len(), 
                  "Architecture {} should preserve all layers", arch_name);
        assert!(model.compression_ratio > 1.0, 
               "Architecture {} should achieve compression", arch_name);
        
        println!("✅ Architecture {}: {:.1}x compression, {} layers", 
                 arch_name, model.compression_ratio, layers.len());
    }
}

#[test]
fn test_progressive_compression_rates() {
    // Test different compression targets and their quality impact
    let compression_targets = vec![
        (0.5, "ultra_low"),
        (1.0, "low"),
        (1.5, "standard"),
        (2.0, "high"),
        (4.0, "lossless"),
    ];
    
    // Standard test model
    let size = 64;
    let original_data: Vec<f32> = (0..size*size).map(|i| {
        ((i as f32) / 100.0).sin() * 0.1
    }).collect();
    
    for (target_bits, mode) in compression_targets {
        let config = NOVAQConfig {
            target_bits,
            num_subspaces: 4,
            refinement_iterations: 20,
            ..Default::default()
        };
        
        let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
        let weight = WeightMatrix::new(original_data.clone(), vec![size, size], 
                                     format!("compression_test_{}", mode));
        
        let result = novaq.compress_model(vec![weight]);
        assert!(result.is_ok(), "Compression test {} should succeed", mode);
        
        let model = result.unwrap();
        
        println!("✅ {} compression ({:.1} bits): {:.1}x ratio, {:.3} accuracy", 
                 mode, target_bits, model.compression_ratio, model.bit_accuracy);
        
        // Higher bit rates should give better accuracy
        if target_bits >= 2.0 {
            assert!(model.bit_accuracy > 0.98, "High bit rate should give high accuracy");
        } else if target_bits >= 1.0 {
            assert!(model.bit_accuracy > 0.90, "Medium bit rate should give good accuracy");
        }
        
        // Should always achieve some compression
        assert!(model.compression_ratio > 1.0, "Should achieve compression at {} bits", target_bits);
    }
}

#[test]
fn test_progress_tracking_integration() {
    // Test that progress tracking integrates properly with quantization
    let config = NOVAQConfig {
        target_bits: 1.5,
        num_subspaces: 4,
        refinement_iterations: 20,
        ..Default::default()
    };
    
    // Test all verbosity levels
    let verbosity_levels = vec![
        VerbosityLevel::Silent,
        VerbosityLevel::Minimal,
        VerbosityLevel::Standard,
        VerbosityLevel::Detailed,
    ];
    
    for verbosity in verbosity_levels {
        let mut novaq = PublicNOVAQ::new_with_verbosity(config.clone(), verbosity);
        
        let data: Vec<f32> = (0..64*64).map(|i| (i as f32) * 0.001).collect();
        let weight = WeightMatrix::new(data, vec![64, 64], "progress_test".to_string());
        
        let result = novaq.compress_model(vec![weight]);
        assert!(result.is_ok(), "Progress tracking test should succeed with verbosity {:?}", verbosity);
        
        let model = result.unwrap();
        assert!(model.compression_ratio > 1.0, "Should achieve compression");
    }
    
    println!("✅ Progress tracking integration: All verbosity levels working");
}

#[test]
fn test_recovery_system_integration() {
    // Test recovery system with challenging scenarios
    let challenging_configs = vec![
        NOVAQConfig {
            target_bits: 0.3,  // Very low bits
            num_subspaces: 8,   // Many subspaces
            refinement_iterations: 5,
            learning_rate: 5.0, // High learning rate
            ..Default::default()
        },
        NOVAQConfig {
            target_bits: 1.0,
            num_subspaces: 1,   // Single subspace
            refinement_iterations: 100, // Many iterations
            learning_rate: 0.00001, // Very low learning rate
            ..Default::default()
        },
    ];
    
    for (i, config) in challenging_configs.into_iter().enumerate() {
        let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
        
        // Create challenging data
        let data: Vec<f32> = (0..64).map(|j| {
            if j % 4 == 0 { 1000.0 }
            else if j % 4 == 1 { -1000.0 }
            else if j % 4 == 2 { 0.00001 }
            else { -0.00001 }
        }).collect();
        
        let weight = WeightMatrix::new(data, vec![8, 8], format!("recovery_test_{}", i));
        
        let result = novaq.compress_model(vec![weight]);
        assert!(result.is_ok(), "Recovery integration test {} should succeed: {:?}", i, result.err());
        
        // Check if recovery was used
        let stats = novaq.get_recovery_stats();
        println!("✅ Recovery test {}: {} attempts, {} successful", 
                 i, stats.total_attempts, stats.successful_recoveries);
    }
}

#[test]
fn test_serialization_roundtrip() {
    // Test that quantized models can be serialized and deserialized correctly
    let config = NOVAQConfig {
        target_bits: 1.5,
        num_subspaces: 4,
        refinement_iterations: 15,
        ..Default::default()
    };
    
    let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
    
    // Create test model
    let data: Vec<f32> = (0..32*32).map(|i| ((i as f32) / 100.0).sin()).collect();
    let weight = WeightMatrix::new(data, vec![32, 32], "serialization_test".to_string());
    
    let result = novaq.compress_model(vec![weight]);
    assert!(result.is_ok(), "Model creation should succeed");
    
    let original_model = result.unwrap();
    
    // Test serialization
    let serialized = bincode::serialize(&original_model);
    assert!(serialized.is_ok(), "Model serialization should succeed");
    
    let serialized_data = serialized.unwrap();
    assert!(serialized_data.len() > 0, "Serialized data should not be empty");
    
    // Test deserialization
    let deserialized: Result<ohms_adaptq::NOVAQModel, _> = bincode::deserialize(&serialized_data);
    assert!(deserialized.is_ok(), "Model deserialization should succeed");
    
    let deserialized_model = deserialized.unwrap();
    
    // Verify integrity
    assert_eq!(deserialized_model.weight_shapes, original_model.weight_shapes);
    assert!((deserialized_model.compression_ratio - original_model.compression_ratio).abs() < 1e-6);
    assert!((deserialized_model.bit_accuracy - original_model.bit_accuracy).abs() < 1e-6);
    
    println!("✅ Serialization roundtrip: {} bytes, {:.1}x compression preserved", 
             serialized_data.len(), deserialized_model.compression_ratio);
}

#[test]
fn test_edge_case_model_sizes() {
    // Test various edge case model sizes
    let edge_cases = vec![
        (1, 1, "minimal"),
        (1, 100, "wide_vector"),
        (100, 1, "tall_vector"),
        (2, 2, "tiny_matrix"),
        (3, 5, "small_rectangular"),
        (7, 11, "prime_dimensions"),
    ];
    
    for (rows, cols, case_name) in edge_cases {
        let config = NOVAQConfig {
            target_bits: 1.5,
            num_subspaces: (cols / 2).max(1), // Adapt subspaces to matrix width
            refinement_iterations: 10,
            ..Default::default()
        };
        
        let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
        
        let total_size = rows * cols;
        let data: Vec<f32> = (0..total_size).map(|i| (i as f32) * 0.1).collect();
        let weight = WeightMatrix::new(data, vec![rows, cols], case_name.to_string());
        
        let result = novaq.compress_model(vec![weight]);
        assert!(result.is_ok(), "Edge case {} ({}x{}) should succeed: {:?}", 
                case_name, rows, cols, result.err());
        
        let model = result.unwrap();
        assert_eq!(model.weight_shapes[case_name], vec![rows, cols]);
        
        println!("✅ Edge case {}: {}x{} matrix, {:.1}x compression", 
                 case_name, rows, cols, model.compression_ratio);
    }
}

#[test]
fn test_validation_and_stats() {
    // Test model validation and statistics functionality
    let config = NOVAQConfig {
        target_bits: 1.5,
        num_subspaces: 4,
        refinement_iterations: 20,
        ..Default::default()
    };
    
    let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
    
    // Create test model
    let data: Vec<f32> = (0..50*50).map(|i| ((i as f32) / 50.0).tanh()).collect();
    let weight = WeightMatrix::new(data, vec![50, 50], "validation_test".to_string());
    
    let result = novaq.compress_model(vec![weight]);
    assert!(result.is_ok(), "Model creation should succeed");
    
    let model = result.unwrap();
    
    // Test validation
    let validation = novaq.validate_model(&model);
    assert!(validation.is_ok(), "Model validation should succeed");
    
    let validation_report = validation.unwrap();
    assert!(validation_report.compression_ratio > 1.0, "Validation should report compression");
    assert!(validation_report.bit_accuracy > 0.9, "Validation should report good accuracy");
    assert!(validation_report.quality_score > 0.9, "Quality score should be high");
    
    // Test compression statistics
    let stats = novaq.get_compression_stats(&model);
    assert_eq!(stats.target_bits, 1.5, "Stats should match config");
    assert_eq!(stats.num_subspaces, 4, "Stats should match config");
    assert!(stats.compression_ratio > 1.0, "Stats should show compression");
    
    println!("✅ Validation: {:.1}x compression, {:.3} quality score, {} issues", 
             validation_report.compression_ratio, 
             validation_report.quality_score,
             validation_report.issues.len());
}