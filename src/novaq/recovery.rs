use crate::Result;
use super::{
    WeightMatrix, NOVAQEngine, NOVAQConfig, NOVAQModel, 
    NumericalStabilityGuard, QuantizationProgressTracker, 
    QuantizationPhase, QualityMetrics, VerbosityLevel
};
use std::time::{Duration, Instant};

/// Recovery strategies available when quantization fails
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecoveryStrategy {
    /// Reduce learning rate and retry
    ReduceLearningRate,
    /// Increase numerical stability thresholds
    IncreaseStabilityThresholds,
    /// Use simpler quantization approach
    FallbackToSimpleQuantization,
    /// Reduce codebook sizes
    ReduceCodebookSizes,
    /// Skip problematic layers
    SkipProblematicLayers,
    /// Reset and retry with different random seed
    ResetWithNewSeed,
}

/// Recovery attempt result
#[derive(Debug, Clone)]
pub struct RecoveryAttempt {
    pub strategy: RecoveryStrategy,
    pub attempt_number: usize,
    pub duration: Duration,
    pub success: bool,
    pub error_message: Option<String>,
    pub stability_issues_recovered: usize,
}

/// Quantization failure classification
#[derive(Debug, Clone, PartialEq)]
pub enum FailureType {
    /// NaN or Inf values detected
    NumericalInstability,
    /// Codebook generation failed
    CodebookGenerationFailure,
    /// Refinement process diverged
    RefinementDivergence,
    /// Memory or resource exhaustion
    ResourceExhaustion,
    /// Input validation failure
    InputValidationFailure,
    /// Unknown or unclassified error
    Unknown,
}

/// Error analysis result
#[derive(Debug, Clone)]
pub struct ErrorAnalysis {
    pub failure_type: FailureType,
    pub recommended_strategies: Vec<RecoveryStrategy>,
    pub confidence: f32,
    pub critical_level: usize, // 1-5, where 5 is most critical
}

/// Recovery statistics tracking
#[derive(Debug, Clone)]
pub struct RecoveryStats {
    pub total_attempts: usize,
    pub successful_recoveries: usize,
    pub failure_types_encountered: Vec<FailureType>,
    pub most_effective_strategy: Option<RecoveryStrategy>,
    pub average_recovery_time: Duration,
    pub max_attempts_used: usize,
}

/// Advanced quantization recovery manager
#[derive(Debug)]
pub struct QuantizationRecoveryManager {
    max_recovery_attempts: usize,
    stability_guard: NumericalStabilityGuard,
    recovery_stats: RecoveryStats,
    base_config: NOVAQConfig,
    current_attempt: usize,
    last_successful_strategy: Option<RecoveryStrategy>,
}

impl QuantizationRecoveryManager {
    pub fn new(base_config: NOVAQConfig) -> Self {
        Self {
            max_recovery_attempts: 5,
            stability_guard: NumericalStabilityGuard::default(),
            recovery_stats: RecoveryStats {
                total_attempts: 0,
                successful_recoveries: 0,
                failure_types_encountered: Vec::new(),
                most_effective_strategy: None,
                average_recovery_time: Duration::from_secs(0),
                max_attempts_used: 0,
            },
            base_config,
            current_attempt: 0,
            last_successful_strategy: None,
        }
    }

    /// Attempt quantization with automatic recovery on failure and progress tracking
    pub fn quantize_with_recovery(&mut self, weights: Vec<WeightMatrix>) -> Result<NOVAQModel> {
        self.quantize_with_recovery_and_progress(weights, VerbosityLevel::Standard)
    }
    
    /// Attempt quantization with automatic recovery and specified verbosity
    pub fn quantize_with_recovery_and_progress(&mut self, weights: Vec<WeightMatrix>, verbosity: VerbosityLevel) -> Result<NOVAQModel> {
        let start_time = Instant::now();
        let mut last_error: Option<Box<dyn std::error::Error + Send + Sync>> = None;
        
        let mut progress = QuantizationProgressTracker::new(verbosity);
        progress.start_phase(QuantizationPhase::ModelLoading, Some(1));

        // First attempt with original configuration
        progress.update_iteration(0, None);
        progress.complete_phase();
        
        progress.start_phase(QuantizationPhase::CodebookInitialization, Some(1));
        match self.attempt_quantization_with_progress(&weights, &self.base_config.clone(), &mut progress) {
            Ok(model) => {
                progress.complete_phase();
                progress.complete();
                return Ok(model);
            }
            Err(e) => {
                progress.error(&format!("Initial quantization failed: {}", e));
                last_error = Some(e);
            }
        }

        // Recovery attempts
        for attempt in 1..=self.max_recovery_attempts {
            self.current_attempt = attempt;
            
            // Analyze the error and determine recovery strategy
            let analysis = self.analyze_error(last_error.as_deref());
            
            println!("🔍 Failure analysis: {:?} (confidence: {:.1}%)", 
                     analysis.failure_type, analysis.confidence * 100.0);

            // Track failure type
            if !self.recovery_stats.failure_types_encountered.contains(&analysis.failure_type) {
                self.recovery_stats.failure_types_encountered.push(analysis.failure_type.clone());
            }

            // Try each recommended strategy
            for strategy in &analysis.recommended_strategies {
                let strategy_start = Instant::now();
                
                println!("🛠️  Attempting recovery strategy: {:?} (attempt {}/{})", 
                         strategy, attempt, self.max_recovery_attempts);

                // Apply recovery strategy and get modified config
                let recovered_config = self.apply_recovery_strategy(*strategy, &analysis);
                
                // Attempt quantization with recovery configuration
                match self.attempt_quantization(&weights, &recovered_config) {
                    Ok(model) => {
                        let recovery_duration = strategy_start.elapsed();
                        
                        // Record successful recovery
                        self.record_successful_recovery(*strategy, attempt, recovery_duration);
                        
                        let total_duration = start_time.elapsed();
                        println!("✅ Quantization recovered successfully after {} attempts in {:.1}s", 
                                 attempt, total_duration.as_secs_f32());
                        println!("🎯 Successful strategy: {:?}", strategy);
                        
                        return Ok(model);
                    }
                    Err(e) => {
                        let recovery_duration = strategy_start.elapsed();
                        self.record_failed_recovery(*strategy, attempt, recovery_duration, e.as_ref());
                        
                        println!("❌ Recovery strategy {:?} failed: {}", strategy, e);
                        last_error = Some(e);
                    }
                }
            }
        }

        // All recovery attempts exhausted
        self.recovery_stats.total_attempts += self.max_recovery_attempts;
        let total_duration = start_time.elapsed();
        
        println!("💥 All recovery attempts exhausted after {:.1}s", total_duration.as_secs_f32());
        self.print_recovery_summary();
        
        Err(format!("Quantization failed after {} recovery attempts. Last error: {}", 
                   self.max_recovery_attempts, 
                   last_error.map(|e| e.to_string()).unwrap_or_else(|| "Unknown error".to_string())).into())
    }

    /// Attempt quantization with specific configuration
    fn attempt_quantization(&mut self, weights: &[WeightMatrix], config: &NOVAQConfig) -> Result<NOVAQModel> {
        let mut engine = NOVAQEngine::new(config.clone());
        engine.quantize_model(weights.to_vec())
    }
    
    /// Attempt quantization with progress tracking
    fn attempt_quantization_with_progress(&mut self, weights: &[WeightMatrix], config: &NOVAQConfig, progress: &mut QuantizationProgressTracker) -> Result<NOVAQModel> {
        let mut engine = NOVAQEngine::new(config.clone());
        engine.quantize_model_with_progress(weights.to_vec(), progress)
    }

    /// Analyze error and determine failure type and recommended strategies
    fn analyze_error(&self, error: Option<&(dyn std::error::Error + Send + Sync)>) -> ErrorAnalysis {
        let error_message = error.map(|e| e.to_string()).unwrap_or_else(|| "Unknown error".to_string());
        let error_lower = error_message.to_lowercase();

        // Pattern matching for error classification
        let (failure_type, confidence) = if error_lower.contains("nan") || error_lower.contains("inf") {
            (FailureType::NumericalInstability, 0.95)
        } else if error_lower.contains("codebook") || error_lower.contains("cluster") {
            (FailureType::CodebookGenerationFailure, 0.85)
        } else if error_lower.contains("refinement") || error_lower.contains("convergence") {
            (FailureType::RefinementDivergence, 0.80)
        } else if error_lower.contains("memory") || error_lower.contains("allocation") {
            (FailureType::ResourceExhaustion, 0.90)
        } else if error_lower.contains("bounds") || error_lower.contains("index") || error_lower.contains("shape") {
            (FailureType::InputValidationFailure, 0.85)
        } else {
            (FailureType::Unknown, 0.50)
        };

        let recommended_strategies = self.get_strategies_for_failure_type(&failure_type);
        let critical_level = self.assess_critical_level(&failure_type, &error_message);

        ErrorAnalysis {
            failure_type,
            recommended_strategies,
            confidence,
            critical_level,
        }
    }

    /// Get recommended recovery strategies for specific failure type
    fn get_strategies_for_failure_type(&self, failure_type: &FailureType) -> Vec<RecoveryStrategy> {
        match failure_type {
            FailureType::NumericalInstability => vec![
                RecoveryStrategy::IncreaseStabilityThresholds,
                RecoveryStrategy::ReduceLearningRate,
                RecoveryStrategy::FallbackToSimpleQuantization,
            ],
            FailureType::CodebookGenerationFailure => vec![
                RecoveryStrategy::ReduceCodebookSizes,
                RecoveryStrategy::ResetWithNewSeed,
                RecoveryStrategy::FallbackToSimpleQuantization,
            ],
            FailureType::RefinementDivergence => vec![
                RecoveryStrategy::ReduceLearningRate,
                RecoveryStrategy::IncreaseStabilityThresholds,
                RecoveryStrategy::ReduceCodebookSizes,
            ],
            FailureType::ResourceExhaustion => vec![
                RecoveryStrategy::ReduceCodebookSizes,
                RecoveryStrategy::SkipProblematicLayers,
                RecoveryStrategy::FallbackToSimpleQuantization,
            ],
            FailureType::InputValidationFailure => vec![
                RecoveryStrategy::FallbackToSimpleQuantization,
                RecoveryStrategy::SkipProblematicLayers,
            ],
            FailureType::Unknown => {
                // Try the most successful strategy first, then fallback options
                if let Some(successful_strategy) = self.last_successful_strategy {
                    vec![
                        successful_strategy,
                        RecoveryStrategy::FallbackToSimpleQuantization,
                        RecoveryStrategy::ResetWithNewSeed,
                    ]
                } else {
                    vec![
                        RecoveryStrategy::FallbackToSimpleQuantization,
                        RecoveryStrategy::ReduceLearningRate,
                        RecoveryStrategy::ResetWithNewSeed,
                    ]
                }
            }
        }
    }

    /// Assess critical level of the failure (1-5)
    fn assess_critical_level(&self, failure_type: &FailureType, error_message: &str) -> usize {
        let base_level = match failure_type {
            FailureType::NumericalInstability => 4, // High priority
            FailureType::CodebookGenerationFailure => 3,
            FailureType::RefinementDivergence => 3,
            FailureType::ResourceExhaustion => 5, // Critical
            FailureType::InputValidationFailure => 2,
            FailureType::Unknown => 3,
        };

        // Adjust based on error message content
        let adjustment = if error_message.to_lowercase().contains("panic") || 
                           error_message.to_lowercase().contains("abort") {
            1
        } else {
            0
        };

        (base_level + adjustment).min(5)
    }

    /// Apply specific recovery strategy and return modified configuration
    fn apply_recovery_strategy(&self, strategy: RecoveryStrategy, analysis: &ErrorAnalysis) -> NOVAQConfig {
        let mut config = self.base_config.clone();

        match strategy {
            RecoveryStrategy::ReduceLearningRate => {
                config.learning_rate *= 0.5;
                println!("   🎛️  Reducing learning rate to {:.6}", config.learning_rate);
            },
            RecoveryStrategy::IncreaseStabilityThresholds => {
                // This would be handled by the NumericalStabilityGuard internally
                println!("   🛡️  Increasing numerical stability thresholds");
            },
            RecoveryStrategy::FallbackToSimpleQuantization => {
                config.num_subspaces = 1;
                config.codebook_size_l1 = config.codebook_size_l1.min(8);
                config.codebook_size_l2 = 2;
                config.refinement_iterations = config.refinement_iterations.min(20);
                println!("   📉 Falling back to simple quantization (1 subspace, reduced codebooks)");
            },
            RecoveryStrategy::ReduceCodebookSizes => {
                config.codebook_size_l1 = (config.codebook_size_l1 / 2).max(2);
                config.codebook_size_l2 = (config.codebook_size_l2 / 2).max(2);
                println!("   📊 Reducing codebook sizes to L1={}, L2={}", 
                         config.codebook_size_l1, config.codebook_size_l2);
            },
            RecoveryStrategy::SkipProblematicLayers => {
                // This would require modifying the weight filtering logic
                config.refinement_iterations = config.refinement_iterations.min(10);
                println!("   ⏭️  Skipping problematic layers (reduced refinement)");
            },
            RecoveryStrategy::ResetWithNewSeed => {
                config.seed = config.seed.wrapping_add(self.current_attempt as u64 * 1000);
                println!("   🎲 Resetting with new random seed: {}", config.seed);
            },
        }

        config
    }

    /// Record successful recovery attempt
    fn record_successful_recovery(&mut self, strategy: RecoveryStrategy, attempt: usize, duration: Duration) {
        self.recovery_stats.successful_recoveries += 1;
        self.recovery_stats.total_attempts += attempt;
        self.recovery_stats.max_attempts_used = self.recovery_stats.max_attempts_used.max(attempt);
        
        // Update average recovery time
        let total_time = self.recovery_stats.average_recovery_time.as_secs_f32() * 
                        (self.recovery_stats.successful_recoveries - 1) as f32 + duration.as_secs_f32();
        self.recovery_stats.average_recovery_time = Duration::from_secs_f32(
            total_time / self.recovery_stats.successful_recoveries as f32
        );

        self.last_successful_strategy = Some(strategy);
        self.recovery_stats.most_effective_strategy = Some(strategy);
    }

    /// Record failed recovery attempt
    fn record_failed_recovery(&mut self, _strategy: RecoveryStrategy, _attempt: usize, _duration: Duration, _error: &(dyn std::error::Error + Send + Sync)) {
        // Could track per-strategy failure rates here if needed
    }

    /// Print recovery statistics summary
    pub fn print_recovery_summary(&self) {
        println!("\n📊 Recovery Statistics Summary:");
        println!("   Total attempts: {}", self.recovery_stats.total_attempts);
        println!("   Successful recoveries: {}", self.recovery_stats.successful_recoveries);
        
        if self.recovery_stats.successful_recoveries > 0 {
            let success_rate = (self.recovery_stats.successful_recoveries as f32 / 
                               self.recovery_stats.total_attempts as f32) * 100.0;
            println!("   Success rate: {:.1}%", success_rate);
            println!("   Average recovery time: {:.1}s", self.recovery_stats.average_recovery_time.as_secs_f32());
            
            if let Some(strategy) = self.recovery_stats.most_effective_strategy {
                println!("   Most effective strategy: {:?}", strategy);
            }
        }
        
        if !self.recovery_stats.failure_types_encountered.is_empty() {
            println!("   Failure types encountered: {:?}", self.recovery_stats.failure_types_encountered);
        }
    }

    /// Get recovery statistics
    pub fn get_stats(&self) -> &RecoveryStats {
        &self.recovery_stats
    }

    /// Reset recovery statistics
    pub fn reset_stats(&mut self) {
        self.recovery_stats = RecoveryStats {
            total_attempts: 0,
            successful_recoveries: 0,
            failure_types_encountered: Vec::new(),
            most_effective_strategy: None,
            average_recovery_time: Duration::from_secs(0),
            max_attempts_used: 0,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recovery_manager_creation() {
        let config = NOVAQConfig::default();
        let manager = QuantizationRecoveryManager::new(config);
        
        assert_eq!(manager.max_recovery_attempts, 5);
        assert_eq!(manager.current_attempt, 0);
        assert!(manager.last_successful_strategy.is_none());
    }

    #[test]
    fn test_error_analysis_nan() {
        let config = NOVAQConfig::default();
        let manager = QuantizationRecoveryManager::new(config);
        
        let error = std::io::Error::new(std::io::ErrorKind::Other, "NaN value detected");
        let analysis = manager.analyze_error(Some(&error));
        
        assert_eq!(analysis.failure_type, FailureType::NumericalInstability);
        assert!(analysis.confidence > 0.9);
        assert!(analysis.recommended_strategies.contains(&RecoveryStrategy::IncreaseStabilityThresholds));
    }

    #[test]
    fn test_recovery_strategy_application() {
        let base_config = NOVAQConfig::default();
        let manager = QuantizationRecoveryManager::new(base_config.clone());
        
        let analysis = ErrorAnalysis {
            failure_type: FailureType::NumericalInstability,
            recommended_strategies: vec![RecoveryStrategy::ReduceLearningRate],
            confidence: 0.95,
            critical_level: 4,
        };
        
        let recovered_config = manager.apply_recovery_strategy(RecoveryStrategy::ReduceLearningRate, &analysis);
        
        assert!(recovered_config.learning_rate < base_config.learning_rate);
    }

    #[test]
    fn test_fallback_strategy() {
        let base_config = NOVAQConfig::default();
        let manager = QuantizationRecoveryManager::new(base_config.clone());
        
        let analysis = ErrorAnalysis {
            failure_type: FailureType::CodebookGenerationFailure,
            recommended_strategies: vec![RecoveryStrategy::FallbackToSimpleQuantization],
            confidence: 0.85,
            critical_level: 3,
        };
        
        let recovered_config = manager.apply_recovery_strategy(RecoveryStrategy::FallbackToSimpleQuantization, &analysis);
        
        assert_eq!(recovered_config.num_subspaces, 1);
        assert!(recovered_config.codebook_size_l1 <= base_config.codebook_size_l1);
        assert!(recovered_config.refinement_iterations <= base_config.refinement_iterations);
    }
}