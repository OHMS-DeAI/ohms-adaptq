use crate::Result;
use super::{WeightMatrix, VectorCodebooks, QuantizationIndices, CodebookEntry, NumericalStabilityGuard};

/// Fallback quantization strategies for edge cases
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationStrategy {
    /// Standard multi-subspace quantization
    MultiSubspace,
    /// Single-subspace with enhanced codebook size
    SingleSubspaceEnhanced,
    /// Direct scalar quantization for very small tensors
    ScalarQuantization,
    /// Uniform quantization with dithering
    UniformDithered,
}

/// Enhanced subspace configuration that handles edge cases robustly
#[derive(Debug, Clone)]
pub struct SubspaceConfig {
    pub effective_subspaces: usize,
    pub subspace_size: usize,
    pub strategy: QuantizationStrategy,
    pub codebook_size_l1: usize,
    pub codebook_size_l2: usize,
    pub min_vectors_per_cluster: usize,
}

/// Robust subspace strategy selector that prevents mathematical instabilities
#[derive(Debug)]
pub struct SubspaceStrategy {
    pub original_subspaces: usize,
    pub original_codebook_l1: usize,
    pub original_codebook_l2: usize,
    pub min_subspace_size: usize,
    pub min_vectors_per_cluster: usize,
}

impl SubspaceStrategy {
    pub fn new(
        original_subspaces: usize,
        original_codebook_l1: usize,
        original_codebook_l2: usize,
    ) -> Self {
        Self {
            original_subspaces,
            original_codebook_l1,
            original_codebook_l2,
            min_subspace_size: 2, // Minimum for mathematical stability
            min_vectors_per_cluster: 3, // Minimum for meaningful clustering
        }
    }

    /// Determine optimal subspace configuration based on tensor dimensions and constraints
    pub fn determine_config(
        &self,
        rows: usize,
        cols: usize,
    ) -> SubspaceConfig {
        // Handle degenerate cases first
        if rows == 0 || cols == 0 {
            return SubspaceConfig {
                effective_subspaces: 1,
                subspace_size: 1,
                strategy: QuantizationStrategy::ScalarQuantization,
                codebook_size_l1: 2,
                codebook_size_l2: 2,
                min_vectors_per_cluster: 1,
            };
        }

        // For very small tensors, use scalar quantization
        if cols == 1 || rows < self.min_vectors_per_cluster {
            return self.create_scalar_config(rows, cols);
        }

        // For small tensors that can't support multiple subspaces
        if cols < self.original_subspaces * self.min_subspace_size {
            return self.create_single_subspace_config(rows, cols);
        }

        // Find optimal subspace division for normal cases
        self.create_multi_subspace_config(rows, cols)
    }

    /// Create configuration for scalar quantization (very small tensors)
    fn create_scalar_config(&self, rows: usize, cols: usize) -> SubspaceConfig {
        let codebook_size = if rows >= 8 {
            self.original_codebook_l1.min(16)
        } else {
            (rows / 2).max(2).min(8)
        };

        SubspaceConfig {
            effective_subspaces: 1,
            subspace_size: cols,
            strategy: QuantizationStrategy::ScalarQuantization,
            codebook_size_l1: codebook_size,
            codebook_size_l2: 2, // Minimal residual codebook
            min_vectors_per_cluster: (rows / codebook_size).max(1),
        }
    }

    /// Create configuration for single subspace with enhanced codebook
    fn create_single_subspace_config(&self, rows: usize, cols: usize) -> SubspaceConfig {
        // Use larger codebook to compensate for lack of subspaces
        let enhanced_l1_size = if cols >= 4 {
            (self.original_codebook_l1 * 2).min(64)
        } else {
            self.original_codebook_l1
        };

        let enhanced_l2_size = if cols >= 4 {
            (self.original_codebook_l2 * 2).min(16)
        } else {
            self.original_codebook_l2
        };

        SubspaceConfig {
            effective_subspaces: 1,
            subspace_size: cols,
            strategy: QuantizationStrategy::SingleSubspaceEnhanced,
            codebook_size_l1: enhanced_l1_size,
            codebook_size_l2: enhanced_l2_size,
            min_vectors_per_cluster: (rows / enhanced_l1_size).max(1),
        }
    }

    /// Create configuration for multi-subspace quantization (normal case)
    fn create_multi_subspace_config(&self, rows: usize, cols: usize) -> SubspaceConfig {
        let mut best_config = SubspaceConfig {
            effective_subspaces: 1,
            subspace_size: cols,
            strategy: QuantizationStrategy::SingleSubspaceEnhanced,
            codebook_size_l1: self.original_codebook_l1,
            codebook_size_l2: self.original_codebook_l2,
            min_vectors_per_cluster: rows / self.original_codebook_l1,
        };

        // Try different subspace counts to find the best fit
        for num_subspaces in (2..=self.original_subspaces).rev() {
            if cols % num_subspaces == 0 {
                let subspace_size = cols / num_subspaces;
                
                // Ensure subspace size meets minimum requirements
                if subspace_size >= self.min_subspace_size {
                    // Calculate minimum vectors needed per cluster
                    let vectors_per_cluster = rows / self.original_codebook_l1;
                    
                    if vectors_per_cluster >= self.min_vectors_per_cluster {
                        best_config = SubspaceConfig {
                            effective_subspaces: num_subspaces,
                            subspace_size,
                            strategy: QuantizationStrategy::MultiSubspace,
                            codebook_size_l1: self.original_codebook_l1,
                            codebook_size_l2: self.original_codebook_l2,
                            min_vectors_per_cluster: vectors_per_cluster,
                        };
                        break;
                    }
                }
            }
        }

        best_config
    }

    /// Print configuration details for debugging
    pub fn print_config(&self, config: &SubspaceConfig, rows: usize, cols: usize) {
        match config.strategy {
            QuantizationStrategy::ScalarQuantization => {
                println!("🔧 NOVAQ: Using scalar quantization for {}×{} tensor (very small)", rows, cols);
                println!("   Codebook size: {} entries", config.codebook_size_l1);
            },
            QuantizationStrategy::SingleSubspaceEnhanced => {
                println!("🔧 NOVAQ: Using enhanced single-subspace for {}×{} tensor", rows, cols);
                println!("   Enhanced codebook: L1={}, L2={}", config.codebook_size_l1, config.codebook_size_l2);
            },
            QuantizationStrategy::MultiSubspace => {
                if config.effective_subspaces != self.original_subspaces {
                    println!("🔧 NOVAQ: Adjusted subspaces from {} to {} for {}×{} tensor", 
                             self.original_subspaces, config.effective_subspaces, rows, cols);
                }
                println!("   Subspace size: {}, Codebooks: L1={}, L2={}", 
                         config.subspace_size, config.codebook_size_l1, config.codebook_size_l2);
            },
            QuantizationStrategy::UniformDithered => {
                println!("🔧 NOVAQ: Using uniform dithered quantization for {}×{} tensor", rows, cols);
            },
        }
    }
}

/// Fallback quantizer for edge cases where standard subspace quantization fails
#[derive(Debug)]
pub struct FallbackQuantizer {
    stability_guard: NumericalStabilityGuard,
}

impl FallbackQuantizer {
    pub fn new() -> Self {
        Self {
            stability_guard: NumericalStabilityGuard::default(),
        }
    }

    /// Perform scalar quantization for very small tensors
    pub fn scalar_quantize(
        &mut self,
        weights: &WeightMatrix,
        config: &SubspaceConfig,
    ) -> Result<(VectorCodebooks, QuantizationIndices)> {
        let rows = weights.rows();
        let cols = weights.cols();

        // Create single codebook for entire vector
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(rows);
        for row_idx in 0..rows {
            let row = weights.get_row(row_idx).to_vec();
            vectors.push(row);
        }

        // Simple k-means for single codebook
        let l1_codebook = self.build_simple_codebook(&vectors, config.codebook_size_l1)?;
        
        // Minimal residual codebook (mostly zeros for scalar case)
        let zero_vector = vec![0.0; cols];
        let l2_codebook = vec![
            CodebookEntry { centroid: zero_vector.clone(), usage_count: 0 },
            CodebookEntry { centroid: zero_vector, usage_count: 0 },
        ];

        // Assign indices
        let mut level1_indices = vec![vec![0u8]; rows];
        let level2_indices = vec![vec![0u8]; rows]; // All zeros for residual

        for (row_idx, vector) in vectors.iter().enumerate() {
            let (nearest_idx, _) = self.find_nearest_centroid(vector, &l1_codebook);
            level1_indices[row_idx][0] = nearest_idx as u8;
        }

        let codebooks = VectorCodebooks {
            level1_codebooks: vec![l1_codebook],
            level2_codebooks: vec![l2_codebook],
            subspace_size: cols,
        };

        let indices = QuantizationIndices {
            level1_indices,
            level2_indices,
        };

        Ok((codebooks, indices))
    }

    /// Build a simple k-means codebook with stability protection
    fn build_simple_codebook(
        &mut self,
        vectors: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<CodebookEntry>> {
        if vectors.is_empty() {
            return Err("Cannot build codebook from empty vector set".into());
        }

        let vector_dim = vectors[0].len();
        let n_vectors = vectors.len();
        let effective_k = k.min(n_vectors);

        // Initialize centroids with actual data points to avoid extreme values
        let mut centroids = Vec::with_capacity(effective_k);
        for i in 0..effective_k {
            let sample_idx = (i * n_vectors / effective_k).min(n_vectors - 1);
            centroids.push(vectors[sample_idx].clone());
        }

        let mut assignments = vec![0; n_vectors];
        let max_iterations = 50; // Reduced iterations for small tensors

        // Simplified k-means with stability protection
        for _iteration in 0..max_iterations {
            // Assignment step
            for (vec_idx, vector) in vectors.iter().enumerate() {
                let (nearest_idx, _) = self.find_nearest_centroid_raw(vector, &centroids);
                assignments[vec_idx] = nearest_idx;
            }

            // Update step with stability protection
            for cluster_idx in 0..effective_k {
                let cluster_vectors: Vec<&Vec<f32>> = vectors.iter()
                    .enumerate()
                    .filter(|(idx, _)| assignments[*idx] == cluster_idx)
                    .map(|(_, vec)| vec)
                    .collect();

                if !cluster_vectors.is_empty() {
                    for dim in 0..vector_dim {
                        let values: Vec<f32> = cluster_vectors.iter().map(|v| v[dim]).collect();
                        centroids[cluster_idx][dim] = self.stability_guard.safe_mean(&values);
                    }
                }
            }
        }

        // Create codebook entries
        let mut codebook = Vec::with_capacity(effective_k);
        for cluster_idx in 0..effective_k {
            let usage_count = assignments.iter().filter(|&&idx| idx == cluster_idx).count();
            
            // Ensure centroid is stable
            self.stability_guard.sanitize_vector(&mut centroids[cluster_idx]);
            
            codebook.push(CodebookEntry {
                centroid: centroids[cluster_idx].clone(),
                usage_count,
            });
        }

        Ok(codebook)
    }

    /// Find nearest centroid in raw centroid list
    fn find_nearest_centroid_raw(&mut self, vector: &[f32], centroids: &[Vec<f32>]) -> (usize, f32) {
        let mut min_distance = f32::INFINITY;
        let mut nearest_idx = 0;

        for (idx, centroid) in centroids.iter().enumerate() {
            let distance = self.euclidean_distance(vector, centroid);
            if distance < min_distance {
                min_distance = distance;
                nearest_idx = idx;
            }
        }

        (nearest_idx, min_distance)
    }

    /// Find nearest centroid in codebook entries
    fn find_nearest_centroid(&mut self, vector: &[f32], codebook: &[CodebookEntry]) -> (usize, f32) {
        let centroids: Vec<Vec<f32>> = codebook.iter().map(|entry| entry.centroid.clone()).collect();
        self.find_nearest_centroid_raw(vector, &centroids)
    }

    /// Calculate stable Euclidean distance
    fn euclidean_distance(&mut self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let diff = self.stability_guard.sanitize_value(x - y);
            sum += diff * diff;
        }
        self.stability_guard.safe_sqrt(sum)
    }
}

impl Default for FallbackQuantizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subspace_strategy_creation() {
        let strategy = SubspaceStrategy::new(4, 16, 4);
        assert_eq!(strategy.original_subspaces, 4);
        assert_eq!(strategy.original_codebook_l1, 16);
        assert_eq!(strategy.original_codebook_l2, 4);
    }

    #[test]
    fn test_scalar_config_for_small_tensor() {
        let strategy = SubspaceStrategy::new(4, 16, 4);
        let config = strategy.determine_config(8, 1); // 8x1 tensor
        
        assert_eq!(config.strategy, QuantizationStrategy::ScalarQuantization);
        assert_eq!(config.effective_subspaces, 1);
        assert_eq!(config.subspace_size, 1);
        assert!(config.codebook_size_l1 <= 8); // Should be reasonable for small tensor
    }

    #[test]
    fn test_single_subspace_config() {
        let strategy = SubspaceStrategy::new(4, 16, 4);
        let config = strategy.determine_config(100, 3); // 100x3 tensor
        
        assert_eq!(config.strategy, QuantizationStrategy::SingleSubspaceEnhanced);
        assert_eq!(config.effective_subspaces, 1);
        assert_eq!(config.subspace_size, 3);
        assert!(config.codebook_size_l1 >= 16); // Should be enhanced
    }

    #[test]
    fn test_multi_subspace_config() {
        let strategy = SubspaceStrategy::new(4, 16, 4);
        let config = strategy.determine_config(1000, 16); // 1000x16 tensor
        
        assert_eq!(config.strategy, QuantizationStrategy::MultiSubspace);
        assert!(config.effective_subspaces > 1);
        assert_eq!(config.subspace_size, 16 / config.effective_subspaces);
    }

    #[test]
    fn test_degenerate_cases() {
        let strategy = SubspaceStrategy::new(4, 16, 4);
        
        // Zero rows
        let config = strategy.determine_config(0, 10);
        assert_eq!(config.strategy, QuantizationStrategy::ScalarQuantization);
        
        // Zero cols
        let config = strategy.determine_config(10, 0);
        assert_eq!(config.strategy, QuantizationStrategy::ScalarQuantization);
    }

    #[test]
    fn test_fallback_quantizer() {
        let mut quantizer = FallbackQuantizer::new();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let weights = WeightMatrix::new(data, vec![2, 2], "test".to_string());
        
        let config = SubspaceConfig {
            effective_subspaces: 1,
            subspace_size: 2,
            strategy: QuantizationStrategy::ScalarQuantization,
            codebook_size_l1: 2,
            codebook_size_l2: 2,
            min_vectors_per_cluster: 1,
        };
        
        let result = quantizer.scalar_quantize(&weights, &config);
        assert!(result.is_ok());
        
        let (codebooks, indices) = result.unwrap();
        assert_eq!(codebooks.level1_codebooks.len(), 1);
        assert_eq!(indices.level1_indices.len(), 2);
    }
}