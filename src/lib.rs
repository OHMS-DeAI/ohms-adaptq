pub mod quantization;
pub mod manifest;
pub mod verification;

pub use quantization::*;
pub use manifest::*;
pub use verification::*;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Debug, Clone)]
pub struct ApqConfig {
    pub target_bits: u8,
    pub calibration_size: usize,
    pub chunk_size: usize,
    pub fidelity_threshold: f32,
    pub seed: Option<u64>,
}

impl Default for ApqConfig {
    fn default() -> Self {
        Self {
            target_bits: 4,
            calibration_size: 512,
            chunk_size: 2 * 1024 * 1024, // 2 MiB
            fidelity_threshold: 0.1,
            seed: Some(42),
        }
    }
}