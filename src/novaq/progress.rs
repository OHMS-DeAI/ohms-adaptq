use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use std::time::{Duration, Instant};
use nu_ansi_term::Color::{Green, Blue, Yellow, Red, Cyan};

/// Comprehensive quantization progress tracking system
/// Replaces verbose iteration logs with clean, meaningful progress indicators
#[derive(Debug)]
pub struct QuantizationProgressTracker {
    multi_progress: MultiProgress,
    main_progress: ProgressBar,
    phase_progress: ProgressBar,
    current_phase: QuantizationPhase,
    start_time: Instant,
    total_phases: u64,
    completed_phases: u64,
    quality_metrics: QualityMetrics,
    verbosity_level: VerbosityLevel,
}

#[derive(Debug, Clone, Copy)]
pub enum QuantizationPhase {
    ModelLoading,
    CodebookInitialization,
    Level1Refinement,
    Level2Refinement,
    QualityValidation,
    ModelSaving,
    Complete,
}

#[derive(Debug, Clone, Copy)]
pub enum VerbosityLevel {
    Silent,     // No output except errors
    Minimal,    // Progress bars only
    Standard,   // Progress bars + phase summaries
    Detailed,   // Standard + quality metrics
}

#[derive(Debug, Clone)]
pub struct QualityMetrics {
    pub mse: f32,
    pub accuracy: f32,
    pub compression_ratio: f32,
    pub recovery_count: u32,
    pub nan_issues: u32,
    pub inf_issues: u32,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            mse: 0.0,
            accuracy: 0.0,
            compression_ratio: 0.0,
            recovery_count: 0,
            nan_issues: 0,
            inf_issues: 0,
        }
    }
}

impl QuantizationProgressTracker {
    pub fn new(verbosity: VerbosityLevel) -> Self {
        let multi_progress = MultiProgress::new();
        
        // Main progress bar for overall quantization
        let main_progress = multi_progress.add(ProgressBar::new(6));
        main_progress.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("#>-")
        );
        main_progress.set_message("Initializing NOVAQ Quantization");
        
        // Phase-specific progress bar
        let phase_progress = multi_progress.add(ProgressBar::new(100));
        phase_progress.set_style(
            ProgressStyle::default_bar()
                .template("  └─ {spinner:.yellow} [{bar:30.yellow/dim}] {percent}% {msg}")
                .unwrap()
                .progress_chars("█▉▊▋▌▍▎▏ ")
        );
        
        Self {
            multi_progress,
            main_progress,
            phase_progress,
            current_phase: QuantizationPhase::ModelLoading,
            start_time: Instant::now(),
            total_phases: 6,
            completed_phases: 0,
            quality_metrics: QualityMetrics::default(),
            verbosity_level: verbosity,
        }
    }
    
    /// Start a new quantization phase
    pub fn start_phase(&mut self, phase: QuantizationPhase, max_iterations: Option<u64>) {
        self.current_phase = phase;
        
        match self.verbosity_level {
            VerbosityLevel::Silent => return,
            _ => {}
        }
        
        let (phase_name, description) = self.get_phase_info(phase);
        
        // Update main progress
        self.main_progress.set_position(self.completed_phases);
        self.main_progress.set_message(format!("{} {}", phase_name, description));
        
        // Reset and configure phase progress
        let iterations = max_iterations.unwrap_or(100);
        self.phase_progress.reset();
        self.phase_progress.set_length(iterations);
        self.phase_progress.set_message(description.to_string());
        
        if matches!(self.verbosity_level, VerbosityLevel::Standard | VerbosityLevel::Detailed) {
            println!("{}", Blue.bold().paint(format!("🔄 Starting: {}", phase_name)));
        }
    }
    
    /// Update iteration progress within current phase
    pub fn update_iteration(&mut self, iteration: u64, metrics: Option<&QualityMetrics>) {
        if matches!(self.verbosity_level, VerbosityLevel::Silent) {
            return;
        }
        
        self.phase_progress.set_position(iteration);
        
        if let Some(m) = metrics {
            self.quality_metrics = m.clone();
            
            // Update message with key metrics
            let message = match self.current_phase {
                QuantizationPhase::Level1Refinement | QuantizationPhase::Level2Refinement => {
                    if m.recovery_count > 0 {
                        format!("MSE: {:.4}, Acc: {:.2}% [Recovered: {}]", m.mse, m.accuracy * 100.0, m.recovery_count)
                    } else {
                        format!("MSE: {:.4}, Accuracy: {:.2}%", m.mse, m.accuracy * 100.0)
                    }
                },
                _ => format!("Quality: {:.2}%", m.accuracy * 100.0),
            };
            self.phase_progress.set_message(message);
        }
    }
    
    /// Complete current phase
    pub fn complete_phase(&mut self) {
        self.completed_phases += 1;
        
        match self.verbosity_level {
            VerbosityLevel::Silent => return,
            _ => {}
        }
        
        self.phase_progress.finish_and_clear();
        
        let (phase_name, _) = self.get_phase_info(self.current_phase);
        let elapsed = self.start_time.elapsed();
        
        if matches!(self.verbosity_level, VerbosityLevel::Standard | VerbosityLevel::Detailed) {
            let status_icon = if self.quality_metrics.recovery_count > 0 { "🛡️" } else { "✅" };
            println!("{} Completed: {} ({})", 
                     status_icon, 
                     phase_name, 
                     self.format_duration(elapsed));
                     
            if matches!(self.verbosity_level, VerbosityLevel::Detailed) {
                self.print_phase_metrics();
            }
        }
        
        self.main_progress.set_position(self.completed_phases);
    }
    
    /// Complete entire quantization process
    pub fn complete(&mut self) {
        match self.verbosity_level {
            VerbosityLevel::Silent => return,
            _ => {}
        }
        
        self.phase_progress.finish_and_clear();
        self.main_progress.finish_and_clear();
        
        let total_time = self.start_time.elapsed();
        
        println!();
        println!("{}", Green.bold().paint("🎉 NOVAQ Quantization Complete!"));
        println!();
        
        if matches!(self.verbosity_level, VerbosityLevel::Standard | VerbosityLevel::Detailed) {
            println!("{}", Cyan.bold().paint("Summary:"));
            println!("  Duration: {}", self.format_duration(total_time));
            println!("  Final Accuracy: {:.2}%", self.quality_metrics.accuracy * 100.0);
            
            if self.quality_metrics.compression_ratio > 0.0 {
                println!("  Compression Ratio: {:.1}x", self.quality_metrics.compression_ratio);
            }
            
            if self.quality_metrics.recovery_count > 0 {
                println!("  {} Recoveries: {} (NaN: {}, Inf: {})", 
                         Yellow.paint("🛡️"),
                         self.quality_metrics.recovery_count,
                         self.quality_metrics.nan_issues,
                         self.quality_metrics.inf_issues);
            } else {
                println!("  {} No numerical issues detected", Green.paint("✨"));
            }
            println!();
        }
    }
    
    /// Report error and cleanup
    pub fn error(&mut self, error_msg: &str) {
        self.phase_progress.abandon_with_message(format!("❌ {}", error_msg));
        self.main_progress.abandon_with_message("❌ Quantization failed");
        
        if !matches!(self.verbosity_level, VerbosityLevel::Silent) {
            println!("{}", Red.bold().paint(format!("❌ Error: {}", error_msg)));
        }
    }
    
    /// Set verbosity level
    pub fn set_verbosity(&mut self, level: VerbosityLevel) {
        self.verbosity_level = level;
    }
    
    /// Check if we should show detailed output
    pub fn should_log_iteration(&self, iteration: u64) -> bool {
        match self.verbosity_level {
            VerbosityLevel::Silent => false,
            VerbosityLevel::Minimal => false,
            VerbosityLevel::Standard => iteration % 25 == 0,
            VerbosityLevel::Detailed => iteration % 10 == 0,
        }
    }
    
    fn get_phase_info(&self, phase: QuantizationPhase) -> (&str, &str) {
        match phase {
            QuantizationPhase::ModelLoading => ("Model Loading", "Loading and preprocessing model weights"),
            QuantizationPhase::CodebookInitialization => ("Codebook Init", "Initializing L1 and L2 codebooks"),
            QuantizationPhase::Level1Refinement => ("L1 Refinement", "Optimizing level-1 codebook centroids"),
            QuantizationPhase::Level2Refinement => ("L2 Refinement", "Optimizing level-2 residual codebooks"),
            QuantizationPhase::QualityValidation => ("Quality Check", "Validating compression quality"),
            QuantizationPhase::ModelSaving => ("Model Saving", "Serializing compressed model"),
            QuantizationPhase::Complete => ("Complete", "Quantization finished successfully"),
        }
    }
    
    fn format_duration(&self, duration: Duration) -> String {
        let total_secs = duration.as_secs();
        if total_secs < 60 {
            format!("{:.1}s", duration.as_secs_f32())
        } else if total_secs < 3600 {
            format!("{}m {}s", total_secs / 60, total_secs % 60)
        } else {
            format!("{}h {}m {}s", total_secs / 3600, (total_secs % 3600) / 60, total_secs % 60)
        }
    }
    
    fn print_phase_metrics(&self) {
        if self.quality_metrics.accuracy > 0.0 {
            println!("    Accuracy: {:.3}%", self.quality_metrics.accuracy * 100.0);
        }
        if self.quality_metrics.mse > 0.0 {
            println!("    MSE Loss: {:.6}", self.quality_metrics.mse);
        }
        if self.quality_metrics.recovery_count > 0 {
            println!("    Numerical Recoveries: {}", self.quality_metrics.recovery_count);
        }
    }
}

/// Create progress tracker with environment-based verbosity
pub fn create_progress_tracker() -> QuantizationProgressTracker {
    let verbosity = match std::env::var("NOVAQ_VERBOSITY").as_deref() {
        Ok("silent") => VerbosityLevel::Silent,
        Ok("minimal") => VerbosityLevel::Minimal,
        Ok("detailed") => VerbosityLevel::Detailed,
        _ => VerbosityLevel::Standard,
    };
    
    QuantizationProgressTracker::new(verbosity)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_progress_tracker_creation() {
        let tracker = QuantizationProgressTracker::new(VerbosityLevel::Standard);
        assert_eq!(tracker.completed_phases, 0);
        assert_eq!(tracker.total_phases, 6);
    }
    
    #[test]
    fn test_phase_transitions() {
        let mut tracker = QuantizationProgressTracker::new(VerbosityLevel::Minimal);
        
        tracker.start_phase(QuantizationPhase::CodebookInitialization, Some(50));
        assert!(matches!(tracker.current_phase, QuantizationPhase::CodebookInitialization));
        
        tracker.complete_phase();
        assert_eq!(tracker.completed_phases, 1);
    }
    
    #[test]
    fn test_metrics_update() {
        let mut tracker = QuantizationProgressTracker::new(VerbosityLevel::Standard);
        let metrics = QualityMetrics {
            mse: 0.001,
            accuracy: 0.95,
            compression_ratio: 4.2,
            recovery_count: 2,
            nan_issues: 1,
            inf_issues: 1,
        };
        
        tracker.update_iteration(10, Some(&metrics));
        assert!((tracker.quality_metrics.accuracy - 0.95).abs() < f32::EPSILON);
        assert_eq!(tracker.quality_metrics.recovery_count, 2);
    }
    
    #[test]
    fn test_verbosity_levels() {
        let silent_tracker = QuantizationProgressTracker::new(VerbosityLevel::Silent);
        assert!(!silent_tracker.should_log_iteration(10));
        
        let detailed_tracker = QuantizationProgressTracker::new(VerbosityLevel::Detailed);
        assert!(detailed_tracker.should_log_iteration(10));
        assert!(!detailed_tracker.should_log_iteration(15));
    }
}