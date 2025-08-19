use std::f32;

/// Comprehensive numerical stability guard for NOVAQ quantization operations
/// Prevents NaN/Inf propagation and provides safe alternatives for problematic calculations
#[derive(Debug, Clone)]
pub struct NumericalStabilityGuard {
    pub epsilon: f32,
    pub max_value: f32,
    pub min_value: f32,
    pub nan_count: usize,
    pub inf_count: usize,
    pub recovery_count: usize,
}

impl Default for NumericalStabilityGuard {
    fn default() -> Self {
        Self {
            epsilon: 1e-8,
            max_value: 1e6,
            min_value: -1e6,
            nan_count: 0,
            inf_count: 0,
            recovery_count: 0,
        }
    }
}

impl NumericalStabilityGuard {
    pub fn new(epsilon: f32, max_value: f32, min_value: f32) -> Self {
        Self {
            epsilon,
            max_value,
            min_value,
            nan_count: 0,
            inf_count: 0,
            recovery_count: 0,
        }
    }

    /// Validate and sanitize a single floating-point value
    pub fn sanitize_value(&mut self, value: f32) -> f32 {
        if value.is_nan() {
            self.nan_count += 1;
            self.recovery_count += 1;
            0.0 // Replace NaN with 0
        } else if value.is_infinite() {
            self.inf_count += 1;
            self.recovery_count += 1;
            if value.is_sign_positive() {
                self.max_value
            } else {
                self.min_value
            }
        } else if value > self.max_value {
            self.recovery_count += 1;
            self.max_value
        } else if value < self.min_value {
            self.recovery_count += 1;
            self.min_value
        } else {
            value
        }
    }

    /// Sanitize an entire vector of values
    pub fn sanitize_vector(&mut self, values: &mut [f32]) {
        for value in values.iter_mut() {
            *value = self.sanitize_value(*value);
        }
    }

    /// Safe division with zero protection
    pub fn safe_divide(&mut self, numerator: f32, denominator: f32) -> f32 {
        if denominator.abs() < self.epsilon {
            self.recovery_count += 1;
            if numerator.abs() < self.epsilon {
                0.0 // 0/0 -> 0
            } else {
                // Non-zero/0 -> sign-preserving large value
                if numerator.is_sign_positive() {
                    self.max_value
                } else {
                    self.min_value
                }
            }
        } else {
            let result = numerator / denominator;
            self.sanitize_value(result)
        }
    }

    /// Safe square root
    pub fn safe_sqrt(&mut self, value: f32) -> f32 {
        if value < 0.0 {
            self.recovery_count += 1;
            0.0 // sqrt of negative -> 0
        } else {
            let result = value.sqrt();
            self.sanitize_value(result)
        }
    }

    /// Safe logarithm
    pub fn safe_log(&mut self, value: f32) -> f32 {
        if value <= 0.0 {
            self.recovery_count += 1;
            f32::NEG_INFINITY.max(self.min_value) // log of non-positive -> very negative
        } else {
            let result = value.ln();
            self.sanitize_value(result)
        }
    }

    /// Safe exponential
    pub fn safe_exp(&mut self, value: f32) -> f32 {
        if value > 100.0 {
            self.recovery_count += 1;
            self.max_value // Prevent overflow
        } else if value < -100.0 {
            self.recovery_count += 1;
            0.0 // Very negative exp -> 0
        } else {
            let result = value.exp();
            self.sanitize_value(result)
        }
    }

    /// Validate and fix MSE calculation
    pub fn safe_mse(&mut self, original: &[f32], reconstructed: &[f32]) -> f32 {
        if original.len() != reconstructed.len() || original.is_empty() {
            self.recovery_count += 1;
            return 1.0; // Default MSE for invalid inputs
        }

        let mut sum = 0.0;
        let mut valid_count = 0;

        for (&orig, &recon) in original.iter().zip(reconstructed.iter()) {
            let orig_clean = self.sanitize_value(orig);
            let recon_clean = self.sanitize_value(recon);
            
            let diff = orig_clean - recon_clean;
            let squared_diff = diff * diff;
            let clean_squared = self.sanitize_value(squared_diff);
            
            if clean_squared.is_finite() {
                sum += clean_squared;
                valid_count += 1;
            }
        }

        if valid_count == 0 {
            self.recovery_count += 1;
            1.0 // No valid values -> default MSE
        } else {
            let mse = self.safe_divide(sum, valid_count as f32);
            if mse < 0.0 {
                self.recovery_count += 1;
                0.0
            } else {
                mse
            }
        }
    }

    /// Safe mean calculation for centroid updates
    pub fn safe_mean(&mut self, values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sum = 0.0;
        let mut valid_count = 0;

        for &value in values {
            let clean_value = self.sanitize_value(value);
            if clean_value.is_finite() {
                sum += clean_value;
                valid_count += 1;
            }
        }

        if valid_count == 0 {
            self.recovery_count += 1;
            0.0
        } else {
            self.safe_divide(sum, valid_count as f32)
        }
    }

    /// Safe weighted update for learning rate applications
    pub fn safe_weighted_update(&mut self, old_value: f32, new_value: f32, learning_rate: f32) -> f32 {
        let clean_old = self.sanitize_value(old_value);
        let clean_new = self.sanitize_value(new_value);
        let clean_lr = self.sanitize_value(learning_rate.clamp(0.0, 1.0));
        
        let weight_old = 1.0 - clean_lr;
        let result = clean_old * weight_old + clean_new * clean_lr;
        self.sanitize_value(result)
    }

    /// Check if values are stable
    pub fn is_stable(&self, values: &[f32]) -> bool {
        for &value in values {
            if !value.is_finite() || value.abs() > self.max_value {
                return false;
            }
        }
        true
    }

    /// Reset counters
    pub fn reset_counters(&mut self) {
        self.nan_count = 0;
        self.inf_count = 0;
        self.recovery_count = 0;
    }

    /// Get stability statistics
    pub fn get_stats(&self) -> StabilityStats {
        StabilityStats {
            nan_count: self.nan_count,
            inf_count: self.inf_count,
            recovery_count: self.recovery_count,
        }
    }

    /// Comprehensive vector validation and cleaning
    pub fn validate_and_clean_vector(&mut self, values: &mut Vec<f32>) -> bool {
        let mut had_issues = false;
        
        for value in values.iter_mut() {
            let original = *value;
            *value = self.sanitize_value(original);
            if (*value - original).abs() > self.epsilon {
                had_issues = true;
            }
        }
        
        had_issues
    }

    /// Adaptive epsilon based on value scale
    pub fn adaptive_epsilon(&self, scale: f32) -> f32 {
        (self.epsilon * scale.abs()).max(self.epsilon)
    }
}

#[derive(Debug, Clone)]
pub struct StabilityStats {
    pub nan_count: usize,
    pub inf_count: usize,
    pub recovery_count: usize,
}

impl StabilityStats {
    pub fn has_issues(&self) -> bool {
        self.nan_count > 0 || self.inf_count > 0 || self.recovery_count > 0
    }

    pub fn print_summary(&self) {
        if self.has_issues() {
            println!("Numerical stability recoveries: {} (NaN: {}, Inf: {})", 
                     self.recovery_count, self.nan_count, self.inf_count);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_nan() {
        let mut guard = NumericalStabilityGuard::default();
        let result = guard.sanitize_value(f32::NAN);
        assert_eq!(result, 0.0);
        assert_eq!(guard.nan_count, 1);
    }

    #[test]
    fn test_sanitize_infinity() {
        let mut guard = NumericalStabilityGuard::default();
        let pos_inf = guard.sanitize_value(f32::INFINITY);
        let neg_inf = guard.sanitize_value(f32::NEG_INFINITY);
        
        assert_eq!(pos_inf, guard.max_value);
        assert_eq!(neg_inf, guard.min_value);
        assert_eq!(guard.inf_count, 2);
    }

    #[test]
    fn test_safe_divide() {
        let mut guard = NumericalStabilityGuard::default();
        
        // Normal division
        assert_eq!(guard.safe_divide(6.0, 2.0), 3.0);
        
        // Division by zero
        let result = guard.safe_divide(5.0, 0.0);
        assert!(result > 0.0);
        assert_eq!(guard.recovery_count, 1);
        
        // Zero by zero
        guard.reset_counters();
        let result = guard.safe_divide(0.0, 0.0);
        assert_eq!(result, 0.0);
        assert_eq!(guard.recovery_count, 1);
    }

    #[test]
    fn test_safe_mse() {
        let mut guard = NumericalStabilityGuard::default();
        
        let original = vec![1.0, 2.0, 3.0];
        let reconstructed = vec![1.1, 1.9, 3.1];
        
        let mse = guard.safe_mse(&original, &reconstructed);
        assert!(mse >= 0.0);
        assert!(mse.is_finite());
    }

    #[test]
    fn test_safe_mse_with_nan() {
        let mut guard = NumericalStabilityGuard::default();
        
        let original = vec![1.0, f32::NAN, 3.0];
        let reconstructed = vec![1.1, 2.0, f32::INFINITY];
        
        let mse = guard.safe_mse(&original, &reconstructed);
        assert!(mse >= 0.0);
        assert!(mse.is_finite());
        assert!(guard.recovery_count > 0);
    }

    #[test]
    fn test_safe_mean() {
        let mut guard = NumericalStabilityGuard::default();
        
        // Normal case
        let values = vec![1.0, 2.0, 3.0];
        let mean = guard.safe_mean(&values);
        assert_eq!(mean, 2.0);
        
        // With NaN
        let values_with_nan = vec![1.0, f32::NAN, 3.0];
        let mean = guard.safe_mean(&values_with_nan);
        assert!(mean.is_finite());
        assert!(guard.recovery_count > 0);
    }

    #[test]
    fn test_safe_weighted_update() {
        let mut guard = NumericalStabilityGuard::default();
        
        let old = 1.0;
        let new = 3.0;
        let lr = 0.5;
        
        let result = guard.safe_weighted_update(old, new, lr);
        assert_eq!(result, 2.0); // (1.0 * 0.5) + (3.0 * 0.5)
        
        // With NaN inputs
        let result = guard.safe_weighted_update(f32::NAN, 3.0, 0.5);
        assert!(result.is_finite());
        assert!(guard.recovery_count > 0);
    }

    #[test]
    fn test_is_stable() {
        let guard = NumericalStabilityGuard::default();
        
        assert!(guard.is_stable(&vec![1.0, 2.0, 3.0]));
        assert!(!guard.is_stable(&vec![1.0, f32::NAN, 3.0]));
        assert!(!guard.is_stable(&vec![1.0, f32::INFINITY, 3.0]));
        assert!(!guard.is_stable(&vec![1.0, 1e10, 3.0])); // Exceeds max_value
    }
}