use crate::Result;
use super::{WeightMatrix, NormalizationMetadata};

/// Distribution Normalizer implementing Stage 1 of NOVAQ
/// 
/// Mathematical formulation from NOVAQ paper:
/// W_hat_{i,:} = (W_{i,:} - μ_i) / s_i
/// 
/// where:
/// μ_i = (1/d) * Σ_j W_{i,j}  (per-channel mean)
/// s_i = σ_i / Δ if σ_i in top p%, else 1  (outlier scaling)
#[derive(Debug)]
pub struct DistributionNormalizer {
    outlier_threshold: f32,
    seed: u64,
}

impl DistributionNormalizer {
    pub fn new(outlier_threshold: f32, seed: u64) -> Self {
        Self {
            outlier_threshold,
            seed,
        }
    }
    
    /// Normalize weight matrix by eliminating per-channel means and rescaling outlier channels
    pub fn normalize(&self, weights: &mut WeightMatrix) -> Result<NormalizationMetadata> {
        let rows = weights.rows();
        let cols = weights.cols();
        
        let mut channel_means = Vec::with_capacity(rows);
        let mut channel_variances = Vec::with_capacity(rows);
        
        // Calculate per-channel statistics
        for i in 0..rows {
            let row = weights.get_row(i);
            
            // Calculate mean: μ_i = (1/d) * Σ_j W_{i,j}
            let mean = row.iter().sum::<f32>() / cols as f32;
            channel_means.push(mean);
            
            // Calculate variance for outlier detection
            let variance = row.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / cols as f32;
            channel_variances.push(variance);
        }
        
        // Identify outlier channels (top-p% by variance)
        let outlier_channels = self.identify_outlier_channels(&channel_variances);
        
        // Calculate scaling factors
        let mut channel_scales = vec![1.0; rows];
        let delta = self.calculate_delta(&channel_variances);
        
        for &channel_idx in &outlier_channels {
            let sigma = channel_variances[channel_idx].sqrt();
            channel_scales[channel_idx] = sigma / delta;
        }
        
        // Apply normalization: W_hat_{i,:} = (W_{i,:} - μ_i) / s_i
        for i in 0..rows {
            let mean = channel_means[i];
            let scale = channel_scales[i];
            let row = weights.get_row_mut(i);
            
            for j in 0..cols {
                row[j] = (row[j] - mean) / scale;
            }
        }
        
        Ok(NormalizationMetadata {
            channel_means,
            channel_scales,
            outlier_channels,
        })
    }
    
    /// Denormalize weights using stored metadata
    pub fn denormalize(&self, weights: &mut [f32], metadata: &NormalizationMetadata) -> Result<()> {
        let rows = metadata.channel_means.len();
        let cols = weights.len() / rows;
        
        for i in 0..rows {
            let mean = metadata.channel_means[i];
            let scale = metadata.channel_scales[i];
            
            let start = i * cols;
            let end = start + cols;
            
            for j in start..end {
                weights[j] = weights[j] * scale + mean;
            }
        }
        
        Ok(())
    }
    
    /// Identify channels with top-p% variance as outliers
    fn identify_outlier_channels(&self, variances: &[f32]) -> Vec<usize> {
        let mut indexed_variances: Vec<(usize, f32)> = variances.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        
        // Sort by variance in descending order
        indexed_variances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top-p% channels
        let num_outliers = ((variances.len() as f32) * self.outlier_threshold).ceil() as usize;
        let num_outliers = num_outliers.max(1); // At least one outlier
        
        indexed_variances.into_iter()
            .take(num_outliers)
            .map(|(idx, _)| idx)
            .collect()
    }
    
    /// Calculate delta parameter for outlier scaling
    /// Uses median variance as a robust estimator
    fn calculate_delta(&self, variances: &[f32]) -> f32 {
        let mut sorted_variances = variances.to_vec();
        sorted_variances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let median_idx = sorted_variances.len() / 2;
        let median_variance = if sorted_variances.len() % 2 == 0 {
            (sorted_variances[median_idx - 1] + sorted_variances[median_idx]) / 2.0
        } else {
            sorted_variances[median_idx]
        };
        
        median_variance.sqrt().max(1e-8) // Avoid division by zero
    }
}

/// Calculate channel-wise statistics for analysis
pub struct ChannelStatistics {
    pub means: Vec<f32>,
    pub variances: Vec<f32>,
    pub std_devs: Vec<f32>,
    pub outlier_ratio: f32,
}

impl ChannelStatistics {
    pub fn compute(weights: &WeightMatrix, outlier_threshold: f32) -> Self {
        let rows = weights.rows();
        let cols = weights.cols();
        
        let mut means = Vec::with_capacity(rows);
        let mut variances = Vec::with_capacity(rows);
        let mut std_devs = Vec::with_capacity(rows);
        
        for i in 0..rows {
            let row = weights.get_row(i);
            
            let mean = row.iter().sum::<f32>() / cols as f32;
            means.push(mean);
            
            let variance = row.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / cols as f32;
            variances.push(variance);
            std_devs.push(variance.sqrt());
        }
        
        Self {
            means,
            variances,
            std_devs,
            outlier_ratio: outlier_threshold,
        }
    }
    
    pub fn print_summary(&self) {
        println!("Channel Statistics Summary:");
        println!("  Total channels: {}", self.means.len());
        println!("  Mean of means: {:.6}", self.means.iter().sum::<f32>() / self.means.len() as f32);
        println!("  Mean variance: {:.6}", self.variances.iter().sum::<f32>() / self.variances.len() as f32);
        println!("  Max variance: {:.6}", self.variances.iter().fold(0.0f32, |a, &b| a.max(b)));
        println!("  Min variance: {:.6}", self.variances.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
        println!("  Outlier threshold: {:.1}%", self.outlier_ratio * 100.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_normalization_basic() {
        let data = vec![
            1.0, 2.0, 3.0,  // Row 0: mean = 2.0
            4.0, 5.0, 6.0,  // Row 1: mean = 5.0
        ];
        let mut weights = WeightMatrix::new(data, vec![2, 3], "test".to_string());
        
        let normalizer = DistributionNormalizer::new(0.5, 42);
        let metadata = normalizer.normalize(&mut weights).unwrap();
        
        assert_eq!(metadata.channel_means.len(), 2);
        assert_eq!(metadata.channel_scales.len(), 2);
        assert!((metadata.channel_means[0] - 2.0).abs() < 1e-6);
        assert!((metadata.channel_means[1] - 5.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_outlier_identification() {
        let variances = vec![1.0, 100.0, 1.5, 2.0, 150.0]; // Indices 1 and 4 are outliers
        let normalizer = DistributionNormalizer::new(0.4, 42); // Top 40% = 2 channels
        let outliers = normalizer.identify_outlier_channels(&variances);
        
        assert_eq!(outliers.len(), 2);
        assert!(outliers.contains(&1)); // Variance 100.0
        assert!(outliers.contains(&4)); // Variance 150.0
    }
    
    #[test]
    fn test_denormalization() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut weights = WeightMatrix::new(data.clone(), vec![2, 3], "test".to_string());
        
        let normalizer = DistributionNormalizer::new(0.5, 42);
        let metadata = normalizer.normalize(&mut weights).unwrap();
        
        // Denormalize
        let mut denormalized = weights.data.clone();
        normalizer.denormalize(&mut denormalized, &metadata).unwrap();
        
        // Should be close to original
        for (orig, denorm) in data.iter().zip(denormalized.iter()) {
            assert!((orig - denorm).abs() < 1e-5, "Original: {}, Denormalized: {}", orig, denorm);
        }
    }
    
    #[test]
    fn test_delta_calculation() {
        let variances = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // Medians: 9.0, sqrt = 3.0
        let normalizer = DistributionNormalizer::new(0.2, 42);
        let delta = normalizer.calculate_delta(&variances);
        
        assert!((delta - 3.0).abs() < 1e-6);
    }
}