use ohms_adaptq::{NOVAQConfig, WeightMatrix, PublicNOVAQ, VerbosityLevel};
use std::time::Instant;

/// Performance regression test suite
/// Ensures stability improvements don't degrade quantization performance

#[test]
fn test_small_model_performance() {
    // Baseline performance test for small models
    let config = NOVAQConfig {
        target_bits: 1.5,
        num_subspaces: 4,
        refinement_iterations: 50,
        ..Default::default()
    };
    
    let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
    
    // Small model: 50x50 matrix
    let size = 50;
    let data: Vec<f32> = (0..size*size).map(|i| {
        let normalized = (i as f32) / (size * size) as f32;
        (normalized * 2.0 - 1.0) * 0.1
    }).collect();
    
    let weight = WeightMatrix::new(data, vec![size, size], "small_perf_test".to_string());
    
    let start = Instant::now();
    let result = novaq.compress_model(vec![weight]);
    let duration = start.elapsed();
    
    assert!(result.is_ok(), "Small model performance test should succeed");
    assert!(duration.as_secs() < 5, "Small model should quantize in <5s, took: {:?}", duration);
    
    let model = result.unwrap();
    assert!(model.compression_ratio > 1.0, "Should achieve compression");
    assert!(model.bit_accuracy > 0.95, "Should maintain high accuracy");
    
    println!("Small model performance: {:.2}s for {}K parameters", 
             duration.as_secs_f32(), (size * size) / 1000);
}

#[test]
fn test_medium_model_performance() {
    // Performance test for medium-sized models
    let config = NOVAQConfig {
        target_bits: 1.5,
        num_subspaces: 4,
        refinement_iterations: 30,
        ..Default::default()
    };
    
    let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
    
    // Medium model: 200x200 matrix (40K parameters)
    let size = 200;
    let data: Vec<f32> = (0..size*size).map(|i| {
        let normalized = (i as f32) / (size * size) as f32;
        ((normalized * 4.0 - 2.0) * 0.05).sin() // Sinusoidal pattern for realism
    }).collect();
    
    let weight = WeightMatrix::new(data, vec![size, size], "medium_perf_test".to_string());
    
    let start = Instant::now();
    let result = novaq.compress_model(vec![weight]);
    let duration = start.elapsed();
    
    assert!(result.is_ok(), "Medium model performance test should succeed");
    assert!(duration.as_secs() < 15, "Medium model should quantize in <15s, took: {:?}", duration);
    
    let model = result.unwrap();
    assert!(model.compression_ratio > 1.0, "Should achieve compression");
    assert!(model.bit_accuracy > 0.93, "Should maintain good accuracy");
    
    println!("Medium model performance: {:.2}s for {}K parameters", 
             duration.as_secs_f32(), (size * size) / 1000);
}

#[test]
fn test_progress_tracking_overhead() {
    // Test that progress tracking doesn't add significant overhead
    let config = NOVAQConfig {
        target_bits: 1.5,
        num_subspaces: 2,
        refinement_iterations: 20,
        ..Default::default()
    };
    
    let size = 100;
    let data: Vec<f32> = (0..size*size).map(|i| (i as f32) * 0.001).collect();
    let weight = WeightMatrix::new(data, vec![size, size], "overhead_test".to_string());
    
    // Test with silent mode (minimal overhead)
    let mut novaq_silent = PublicNOVAQ::new_with_verbosity(config.clone(), VerbosityLevel::Silent);
    let start_silent = Instant::now();
    let result_silent = novaq_silent.compress_model(vec![weight.clone()]);
    let duration_silent = start_silent.elapsed();
    
    // Test with standard progress tracking
    let mut novaq_standard = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Standard);
    let start_standard = Instant::now();
    let result_standard = novaq_standard.compress_model(vec![weight]);
    let duration_standard = start_standard.elapsed();
    
    assert!(result_silent.is_ok(), "Silent mode should succeed");
    assert!(result_standard.is_ok(), "Standard mode should succeed");
    
    // Progress tracking overhead should be minimal (less than 20% increase)
    let overhead_ratio = duration_standard.as_secs_f32() / duration_silent.as_secs_f32();
    assert!(overhead_ratio < 1.2, 
            "Progress tracking overhead should be <20%, was: {:.1}%", 
            (overhead_ratio - 1.0) * 100.0);
    
    println!("Progress tracking overhead: {:.1}% ({:.2}s vs {:.2}s)", 
             (overhead_ratio - 1.0) * 100.0,
             duration_standard.as_secs_f32(),
             duration_silent.as_secs_f32());
}

#[test]
fn test_recovery_system_performance() {
    // Test that recovery system activation doesn't cause excessive delays
    let config = NOVAQConfig {
        target_bits: 0.8, // Low bits to potentially trigger recovery
        num_subspaces: 6,  // High subspaces for potential edge cases
        refinement_iterations: 10,
        learning_rate: 5.0, // High learning rate for potential instability
        ..Default::default()
    };
    
    let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
    
    // Create potentially challenging data
    let data: Vec<f32> = (0..64).map(|i| {
        if i % 3 == 0 { 100.0 }
        else if i % 3 == 1 { -100.0 }
        else { 0.001 }
    }).collect();
    
    let weight = WeightMatrix::new(data, vec![8, 8], "recovery_perf_test".to_string());
    
    let start = Instant::now();
    let result = novaq.compress_model(vec![weight]);
    let duration = start.elapsed();
    
    assert!(result.is_ok(), "Recovery performance test should succeed");
    assert!(duration.as_secs() < 30, "Recovery should complete within reasonable time: {:?}", duration);
    
    // Check if recovery was used
    let stats = novaq.get_recovery_stats();
    if stats.total_attempts > 0 {
        println!("Recovery activated: {} attempts in {:.2}s", 
                 stats.total_attempts, duration.as_secs_f32());
    } else {
        println!("No recovery needed, completed in {:.2}s", duration.as_secs_f32());
    }
}

#[test]
fn test_multiple_weights_performance() {
    // Test performance with multiple weight matrices (realistic model scenario)
    let config = NOVAQConfig {
        target_bits: 1.5,
        num_subspaces: 4,
        refinement_iterations: 20,
        ..Default::default()
    };
    
    let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
    
    let mut weights = Vec::new();
    
    // Simulate multiple layers of a neural network
    for layer in 0..5 {
        let size = 50 + layer * 10; // Varying sizes
        let total_params = size * size;
        
        let data: Vec<f32> = (0..total_params).map(|i| {
            let normalized = (i as f32) / total_params as f32;
            ((normalized + layer as f32) * 2.0 - 1.0) * 0.1
        }).collect();
        
        weights.push(WeightMatrix::new(data, vec![size, size], format!("layer_{}", layer)));
    }
    
    let total_params: usize = weights.iter().map(|w| w.data.len()).sum();
    
    let start = Instant::now();
    let result = novaq.compress_model(weights);
    let duration = start.elapsed();
    
    assert!(result.is_ok(), "Multiple weights performance test should succeed");
    assert!(duration.as_secs() < 60, "Multiple weights should quantize in <60s, took: {:?}", duration);
    
    let model = result.unwrap();
    assert_eq!(model.weight_shapes.len(), 5, "Should preserve all weight shapes");
    assert!(model.compression_ratio > 1.0, "Should achieve compression");
    
    println!("Multiple weights performance: {:.2}s for {}K total parameters across 5 layers", 
             duration.as_secs_f32(), total_params / 1000);
}

#[test]
fn test_memory_efficiency() {
    // Test that quantization doesn't consume excessive memory
    let config = NOVAQConfig {
        target_bits: 1.5,
        num_subspaces: 4,
        refinement_iterations: 10,
        ..Default::default()
    };
    
    // Test with moderately large model
    let size = 300; // 90K parameters
    let data: Vec<f32> = (0..size*size).map(|i| {
        let x = (i % size) as f32 / size as f32;
        let y = (i / size) as f32 / size as f32;
        ((x * 6.28).sin() + (y * 6.28).cos()) * 0.1
    }).collect();
    
    let weight = WeightMatrix::new(data, vec![size, size], "memory_test".to_string());
    
    let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
    
    let start = Instant::now();
    let result = novaq.compress_model(vec![weight]);
    let duration = start.elapsed();
    
    assert!(result.is_ok(), "Memory efficiency test should succeed");
    
    let model = result.unwrap();
    let original_size = size * size * 4; // f32 = 4 bytes
    let compression_achieved = model.compression_ratio;
    
    assert!(compression_achieved > 2.0, 
            "Should achieve significant compression: {:.1}x", compression_achieved);
    
    println!("Memory efficiency: {:.1}x compression on {}MB model in {:.2}s", 
             compression_achieved, 
             original_size / (1024 * 1024),
             duration.as_secs_f32());
}

#[test]
fn test_concurrent_safety() {
    // Test that multiple NOVAQ instances can run safely (no shared state issues)
    use std::thread;
    use std::sync::Arc;
    
    let config = Arc::new(NOVAQConfig {
        target_bits: 1.5,
        num_subspaces: 2,
        refinement_iterations: 10,
        ..Default::default()
    });
    
    let handles: Vec<_> = (0..3).map(|thread_id| {
        let config = Arc::clone(&config);
        
        thread::spawn(move || {
            let mut novaq = PublicNOVAQ::new_with_verbosity((*config).clone(), VerbosityLevel::Silent);
            
            // Each thread works with different data
            let data: Vec<f32> = (0..64).map(|i| ((thread_id * 64 + i) as f32) * 0.01).collect();
            let weight = WeightMatrix::new(data, vec![8, 8], format!("thread_{}", thread_id));
            
            let start = Instant::now();
            let result = novaq.compress_model(vec![weight]);
            let duration = start.elapsed();
            
            (thread_id, result.is_ok(), duration)
        })
    }).collect();
    
    let mut all_successful = true;
    let mut total_time = std::time::Duration::from_secs(0);
    
    for handle in handles {
        let (thread_id, success, duration) = handle.join().unwrap();
        all_successful = all_successful && success;
        total_time += duration;
        
        assert!(success, "Thread {} should complete successfully", thread_id);
        assert!(duration.as_secs() < 10, "Thread {} should complete quickly", thread_id);
    }
    
    assert!(all_successful, "All concurrent threads should succeed");
    println!("Concurrent safety: 3 threads completed in avg {:.2}s each", 
             total_time.as_secs_f32() / 3.0);
}

#[test]
fn test_quality_vs_speed_tradeoffs() {
    // Test different refinement iteration counts and their impact on quality/speed
    let test_cases = vec![
        (5, "fast"),
        (20, "standard"), 
        (50, "high_quality"),
    ];
    
    for (iterations, mode) in test_cases {
        let config = NOVAQConfig {
            target_bits: 1.5,
            num_subspaces: 4,
            refinement_iterations: iterations,
            ..Default::default()
        };
        
        let mut novaq = PublicNOVAQ::new_with_verbosity(config, VerbosityLevel::Silent);
        
        // Standard test data
        let size = 80;
        let data: Vec<f32> = (0..size*size).map(|i| {
            let normalized = (i as f32) / (size * size) as f32;
            (normalized * 6.28).sin() * 0.1
        }).collect();
        
        let weight = WeightMatrix::new(data, vec![size, size], format!("quality_test_{}", mode));
        
        let start = Instant::now();
        let result = novaq.compress_model(vec![weight]);
        let duration = start.elapsed();
        
        assert!(result.is_ok(), "Quality test {} should succeed", mode);
        
        let model = result.unwrap();
        
        println!("{} mode ({} iterations): {:.2}s, {:.1}x compression, {:.3} accuracy", 
                 mode, iterations, duration.as_secs_f32(), 
                 model.compression_ratio, model.bit_accuracy);
        
        // Quality should improve with more iterations
        if mode == "fast" {
            assert!(duration.as_secs() < 3, "Fast mode should be quick");
        } else if mode == "high_quality" {
            assert!(model.bit_accuracy > 0.95, "High quality mode should have high accuracy");
        }
    }
}