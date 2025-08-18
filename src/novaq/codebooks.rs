use crate::Result;
use super::{WeightMatrix, VectorCodebooks, QuantizationIndices, CodebookEntry};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Multi-stage Vector Codebook Builder implementing Stage 2 of NOVAQ
/// 
/// Mathematical formulation:
/// For each subspace k:
/// 1. Coarse quantization: b^(1)_{i,k} = argmin_c ||v_{i,k} - C^(1)_{c,k}||²
/// 2. Residual calculation: r_{i,k} = v_{i,k} - C^(1)_{b^(1)_{i,k},k}
/// 3. Residual quantization: b^(2)_{i,k} = argmin_c ||r_{i,k} - C^(2)_{c,k}||²
/// 
/// Final reconstruction: W_{i,:} = Σ_{k=1}^N (C^(1)_{b^(1)_{i,k},k} + C^(2)_{b^(2)_{i,k},k})
#[derive(Debug)]
pub struct CodebookBuilder {
    num_subspaces: usize,
    codebook_size_l1: usize,
    codebook_size_l2: usize,
    rng: ChaCha8Rng,
    max_kmeans_iterations: usize,
    convergence_threshold: f32,
}

impl CodebookBuilder {
    pub fn new(num_subspaces: usize, codebook_size_l1: usize, codebook_size_l2: usize, seed: u64) -> Self {
        Self {
            num_subspaces,
            codebook_size_l1,
            codebook_size_l2,
            rng: ChaCha8Rng::seed_from_u64(seed),
            max_kmeans_iterations: 100,
            convergence_threshold: 1e-6,
        }
    }
    
    /// Build multi-stage vector codebooks for given weight matrix
    pub fn build_codebooks(&mut self, weights: &WeightMatrix) -> Result<(VectorCodebooks, QuantizationIndices)> {
        let rows = weights.rows();
        let cols = weights.cols();
        
        // Calculate subspace size
        let subspace_size = cols / self.num_subspaces;
        if cols % self.num_subspaces != 0 {
            return Err("Weight matrix width must be divisible by number of subspaces".into());
        }
        
        let mut level1_codebooks = Vec::with_capacity(self.num_subspaces);
        let mut level2_codebooks = Vec::with_capacity(self.num_subspaces);
        let mut level1_indices = vec![vec![0u8; self.num_subspaces]; rows];
        let mut level2_indices = vec![vec![0u8; self.num_subspaces]; rows];
        
        // Process each subspace
        for subspace_idx in 0..self.num_subspaces {
            let start_col = subspace_idx * subspace_size;
            let end_col = start_col + subspace_size;
            
            // Extract subspace vectors
            let mut subspace_vectors = Vec::with_capacity(rows);
            for row_idx in 0..rows {
                let row = weights.get_row(row_idx);
                subspace_vectors.push(row[start_col..end_col].to_vec());
            }
            
            // Stage 1: Build first-level codebook using k-means
            let l1_codebook = self.build_kmeans_codebook(&subspace_vectors, self.codebook_size_l1)?;
            
            // Stage 2: Calculate residuals and build second-level codebook
            let mut residuals = Vec::with_capacity(rows);
            for (row_idx, vector) in subspace_vectors.iter().enumerate() {
                // Find nearest centroid in L1 codebook
                let (nearest_idx, _) = self.find_nearest_centroid(vector, &l1_codebook);
                level1_indices[row_idx][subspace_idx] = nearest_idx as u8;
                
                // Calculate residual: r_{i,k} = v_{i,k} - C^(1)_{b^(1)_{i,k},k}
                let residual = self.calculate_residual(vector, &l1_codebook[nearest_idx].centroid);
                residuals.push(residual);
            }
            
            // Build second-level codebook for residuals
            let l2_codebook = self.build_kmeans_codebook(&residuals, self.codebook_size_l2)?;
            
            // Assign residuals to L2 codebook
            for (row_idx, residual) in residuals.iter().enumerate() {
                let (nearest_idx, _) = self.find_nearest_centroid(residual, &l2_codebook);
                level2_indices[row_idx][subspace_idx] = nearest_idx as u8;
            }
            
            level1_codebooks.push(l1_codebook);
            level2_codebooks.push(l2_codebook);
        }
        
        let codebooks = VectorCodebooks {
            level1_codebooks,
            level2_codebooks,
            subspace_size,
        };
        
        let indices = QuantizationIndices {
            level1_indices,
            level2_indices,
        };
        
        Ok((codebooks, indices))
    }
    
    /// Reconstruct weights from codebooks and indices
    pub fn reconstruct_weights(
        &self,
        codebooks: &VectorCodebooks,
        indices: &QuantizationIndices,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        let mut reconstructed = vec![0.0; rows * cols];
        
        for row_idx in 0..rows {
            for subspace_idx in 0..self.num_subspaces {
                let start_col = subspace_idx * codebooks.subspace_size;
                
                // Get codebook entries
                let l1_idx = indices.level1_indices[row_idx][subspace_idx] as usize;
                let l2_idx = indices.level2_indices[row_idx][subspace_idx] as usize;
                
                let l1_centroid = &codebooks.level1_codebooks[subspace_idx][l1_idx].centroid;
                let l2_centroid = &codebooks.level2_codebooks[subspace_idx][l2_idx].centroid;
                
                // Reconstruct: v = C^(1) + C^(2)
                for (offset, (&l1_val, &l2_val)) in l1_centroid.iter().zip(l2_centroid.iter()).enumerate() {
                    let col_idx = start_col + offset;
                    let flat_idx = row_idx * cols + col_idx;
                    reconstructed[flat_idx] = l1_val + l2_val;
                }
            }
        }
        
        Ok(reconstructed)
    }
    
    /// Build codebook using k-means clustering
    fn build_kmeans_codebook(&mut self, vectors: &[Vec<f32>], k: usize) -> Result<Vec<CodebookEntry>> {
        if vectors.is_empty() {
            return Err("Cannot build codebook from empty vector set".into());
        }
        
        let vector_dim = vectors[0].len();
        let n_vectors = vectors.len();
        
        // Handle case where k >= number of vectors
        let effective_k = k.min(n_vectors);
        
        // Initialize centroids randomly
        let mut centroids = Vec::with_capacity(effective_k);
        for _ in 0..effective_k {
            let mut centroid = vec![0.0; vector_dim];
            for dim in 0..vector_dim {
                centroid[dim] = self.rng.gen_range(-1.0..1.0);
            }
            centroids.push(centroid);
        }
        
        let mut assignments = vec![0; n_vectors];
        let mut prev_centroids = centroids.clone();
        
        // K-means iterations
        for iteration in 0..self.max_kmeans_iterations {
            // Assignment step: assign each vector to nearest centroid
            for (vec_idx, vector) in vectors.iter().enumerate() {
                let (nearest_idx, _) = self.find_nearest_centroid_raw(vector, &centroids);
                assignments[vec_idx] = nearest_idx;
            }
            
            // Update step: recompute centroids
            for cluster_idx in 0..effective_k {
                let cluster_vectors: Vec<&Vec<f32>> = vectors.iter()
                    .enumerate()
                    .filter(|(idx, _)| assignments[*idx] == cluster_idx)
                    .map(|(_, vec)| vec)
                    .collect();
                
                if !cluster_vectors.is_empty() {
                    // Compute mean of assigned vectors
                    for dim in 0..vector_dim {
                        let sum: f32 = cluster_vectors.iter().map(|v| v[dim]).sum();
                        centroids[cluster_idx][dim] = sum / cluster_vectors.len() as f32;
                    }
                } else {
                    // Reinitialize empty cluster
                    for dim in 0..vector_dim {
                        centroids[cluster_idx][dim] = self.rng.gen_range(-1.0..1.0);
                    }
                }
            }
            
            // Check convergence
            let converged = self.check_convergence(&prev_centroids, &centroids);
            if converged {
                break;
            }
            
            prev_centroids = centroids.clone();
        }
        
        // Create codebook entries with usage counts
        let mut codebook = Vec::with_capacity(effective_k);
        for cluster_idx in 0..effective_k {
            let usage_count = assignments.iter().filter(|&&idx| idx == cluster_idx).count();
            codebook.push(CodebookEntry {
                centroid: centroids[cluster_idx].clone(),
                usage_count,
            });
        }
        
        Ok(codebook)
    }
    
    /// Find nearest centroid and return index and distance
    fn find_nearest_centroid(&self, vector: &[f32], codebook: &[CodebookEntry]) -> (usize, f32) {
        let centroids: Vec<Vec<f32>> = codebook.iter().map(|entry| entry.centroid.clone()).collect();
        self.find_nearest_centroid_raw(vector, &centroids)
    }
    
    /// Find nearest centroid in raw centroid list
    fn find_nearest_centroid_raw(&self, vector: &[f32], centroids: &[Vec<f32>]) -> (usize, f32) {
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
    
    /// Calculate residual vector
    fn calculate_residual(&self, vector: &[f32], centroid: &[f32]) -> Vec<f32> {
        vector.iter().zip(centroid.iter()).map(|(&v, &c)| v - c).collect()
    }
    
    /// Calculate Euclidean distance between two vectors
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum::<f32>().sqrt()
    }
    
    /// Check convergence of k-means algorithm
    fn check_convergence(&self, prev_centroids: &[Vec<f32>], current_centroids: &[Vec<f32>]) -> bool {
        for (prev, current) in prev_centroids.iter().zip(current_centroids.iter()) {
            let distance = self.euclidean_distance(prev, current);
            if distance > self.convergence_threshold {
                return false;
            }
        }
        true
    }
}

/// Calculate quantization error metrics
pub struct QuantizationMetrics {
    pub mse: f32,
    pub psnr: f32,
    pub max_error: f32,
    pub mean_error: f32,
}

impl QuantizationMetrics {
    pub fn calculate(original: &[f32], reconstructed: &[f32]) -> Self {
        assert_eq!(original.len(), reconstructed.len());
        
        let n = original.len() as f32;
        let mut mse = 0.0;
        let mut max_error: f32 = 0.0;
        let mut total_error = 0.0;
        
        for (&orig, &recon) in original.iter().zip(reconstructed.iter()) {
            let error = (orig - recon).abs();
            let squared_error = error * error;
            
            mse += squared_error;
            max_error = max_error.max(error);
            total_error += error;
        }
        
        mse /= n;
        let mean_error = total_error / n;
        
        // Calculate PSNR (Peak Signal-to-Noise Ratio)
        let max_val = original.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        let psnr = if mse > 0.0 {
            20.0 * (max_val / mse.sqrt()).log10()
        } else {
            f32::INFINITY
        };
        
        Self {
            mse,
            psnr,
            max_error,
            mean_error,
        }
    }
    
    pub fn print_summary(&self) {
        println!("Quantization Metrics:");
        println!("  MSE: {:.6}", self.mse);
        println!("  PSNR: {:.2} dB", self.psnr);
        println!("  Max error: {:.6}", self.max_error);
        println!("  Mean error: {:.6}", self.mean_error);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_codebook_builder_creation() {
        let builder = CodebookBuilder::new(4, 16, 4, 42);
        assert_eq!(builder.num_subspaces, 4);
        assert_eq!(builder.codebook_size_l1, 16);
        assert_eq!(builder.codebook_size_l2, 4);
    }
    
    #[test]
    fn test_build_codebooks_basic() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0,  // 4 columns for 2 subspaces of size 2
            5.0, 6.0, 7.0, 8.0,
            1.1, 2.1, 3.1, 4.1,
            5.1, 6.1, 7.1, 8.1,
        ];
        let weights = WeightMatrix::new(data, vec![4, 4], "test".to_string());
        
        let mut builder = CodebookBuilder::new(2, 4, 2, 42);
        let result = builder.build_codebooks(&weights);
        
        assert!(result.is_ok());
        let (codebooks, indices) = result.unwrap();
        
        assert_eq!(codebooks.level1_codebooks.len(), 2);
        assert_eq!(codebooks.level2_codebooks.len(), 2);
        assert_eq!(codebooks.subspace_size, 2);
        assert_eq!(indices.level1_indices.len(), 4);
        assert_eq!(indices.level2_indices.len(), 4);
    }
    
    #[test]
    fn test_reconstruction() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ];
        let weights = WeightMatrix::new(data.clone(), vec![2, 4], "test".to_string());
        
        let mut builder = CodebookBuilder::new(2, 2, 2, 42);
        let (codebooks, indices) = builder.build_codebooks(&weights).unwrap();
        
        let reconstructed = builder.reconstruct_weights(&codebooks, &indices, 2, 4).unwrap();
        
        assert_eq!(reconstructed.len(), 8);
        
        // Calculate reconstruction error
        let metrics = QuantizationMetrics::calculate(&data, &reconstructed);
        println!("Reconstruction MSE: {:.6}", metrics.mse);
        
        // Should have reasonable reconstruction quality
        assert!(metrics.mse < 10.0); // Adjust threshold as needed
    }
    
    #[test]
    fn test_euclidean_distance() {
        let builder = CodebookBuilder::new(2, 2, 2, 42);
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        let distance = builder.euclidean_distance(&a, &b);
        let expected = ((3.0_f32).powi(2) + (3.0_f32).powi(2) + (3.0_f32).powi(2)).sqrt();
        
        assert!((distance - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_residual_calculation() {
        let builder = CodebookBuilder::new(2, 2, 2, 42);
        let vector = vec![5.0, 7.0, 9.0];
        let centroid = vec![2.0, 3.0, 4.0];
        
        let residual = builder.calculate_residual(&vector, &centroid);
        let expected = vec![3.0, 4.0, 5.0];
        
        assert_eq!(residual, expected);
    }
}