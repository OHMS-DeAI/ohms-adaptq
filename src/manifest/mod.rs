// Manifest Module - NOVAQ Deployment Artifacts
// Handles NOVAQ compressed model artifacts and deployment manifests

use crate::novaq::NOVAQModel;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use chrono;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub model_id: String,
    pub version: String,
    pub compression_type: String, // "novaq"
    pub chunks: Vec<ChunkInfo>,
    pub digest: String,
    pub novaq_metadata: NOVAQMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NOVAQMetadata {
    pub target_bits: f32,           // 1.5 for NOVAQ
    pub num_subspaces: usize,       // Multi-stage codebooks
    pub compression_ratio: f32,     // 93-100x
    pub bit_accuracy: f32,          // >99%
    pub quality_score: f32,         // Overall quality metric
    pub codebook_size_l1: usize,    // Level 1 codebook size
    pub codebook_size_l2: usize,    // Level 2 codebook size
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkInfo {
    pub id: String,
    pub offset: usize,
    pub size: usize,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMeta {
    pub family: String,
    pub arch: String,
    pub tokenizer_id: String,
    pub vocab_size: usize,
    pub ctx_window: usize,
    pub license: String,
    pub quantization: String, // "novaq"
}

pub struct ManifestBuilder;

impl ManifestBuilder {
    /// Create manifest from NOVAQ result
    pub fn from_novaq_model(model: &NOVAQModel, model_id: &str, version: &str) -> crate::Result<Manifest> {
        // Serialize the model to get the actual data size
        let model_data = bincode::serialize(model)?;
        
        let chunk = ChunkInfo {
            id: "novaq_compressed".to_string(),
            offset: 0,
            size: model_data.len(),
            sha256: hex::encode(Sha256::digest(&model_data)),
        };

        let novaq_metadata = NOVAQMetadata {
            target_bits: model.config.target_bits,
            num_subspaces: model.config.num_subspaces,
            compression_ratio: model.compression_ratio,
            bit_accuracy: model.bit_accuracy,
            quality_score: (model.compression_ratio / 100.0 + model.bit_accuracy) / 2.0,
            codebook_size_l1: model.config.codebook_size_l1,
            codebook_size_l2: model.config.codebook_size_l2,
        };

        let manifest_data = serde_json::to_string(&chunk)?;
        let digest = hex::encode(Sha256::digest(manifest_data.as_bytes()));

        Ok(Manifest {
            model_id: model_id.to_string(),
            version: version.to_string(),
            compression_type: "novaq".to_string(),
            chunks: vec![chunk],
            digest,
            novaq_metadata,
        })
    }

    pub fn build_model_meta(
        family: &str,
        arch: &str,
        tokenizer_id: &str,
        vocab_size: usize,
        ctx_window: usize,
        license: &str,
    ) -> ModelMeta {
        ModelMeta {
            family: family.to_string(),
            arch: arch.to_string(),
            tokenizer_id: tokenizer_id.to_string(),
            vocab_size,
            ctx_window,
            license: license.to_string(),
            quantization: "novaq".to_string(),
        }
    }

    pub fn verify_manifest_integrity(manifest: &Manifest, actual_chunks: &[Vec<u8>]) -> bool {
        if manifest.chunks.len() != actual_chunks.len() {
            return false;
        }

        for (chunk_info, actual_data) in manifest.chunks.iter().zip(actual_chunks.iter()) {
            if chunk_info.size != actual_data.len() {
                return false;
            }

            let actual_hash = hex::encode(Sha256::digest(actual_data));
            if chunk_info.sha256 != actual_hash {
                return false;
            }
        }

        true
    }

    /// Get compression statistics from manifest
    pub fn get_stats(manifest: &Manifest) -> String {
        let meta = &manifest.novaq_metadata;
        format!(
            "NOVAQ Stats:\n\
             • Target bits: {}\n\
             • Subspaces: {}\n\
             • Compression: {:.1}x\n\
             • Bit accuracy: {:.3}%\n\
             • Quality score: {:.3}\n\
             • L1 codebook: {}\n\
             • L2 codebook: {}",
            meta.target_bits,
            meta.num_subspaces,
            meta.compression_ratio,
            meta.bit_accuracy * 100.0,
            meta.quality_score,
            meta.codebook_size_l1,
            meta.codebook_size_l2
        )
    }

    /// Create deployment manifest for OHMS platform
    pub fn create_deployment_manifest(
        model: &NOVAQModel,
        model_id: &str,
        admin_principal: &str,
        description: &str,
        ohms_canister: &str,
    ) -> crate::Result<serde_json::Value> {
        let model_data = bincode::serialize(model)?;
        
        Ok(serde_json::json!({
            "model_id": model_id,
            "admin_principal": admin_principal,
            "description": description,
            "ohms_canister": ohms_canister,
            "compression_type": "novaq",
            "compression_ratio": model.compression_ratio,
            "bit_accuracy": model.bit_accuracy,
            "quality_score": (model.compression_ratio / 100.0 + model.bit_accuracy) / 2.0,
            "submission_timestamp": chrono::Utc::now().timestamp(),
            "model_size_bytes": model_data.len(),
            "checksum": hex::encode(Sha256::digest(&model_data)),
            "novaq_config": {
                "target_bits": model.config.target_bits,
                "num_subspaces": model.config.num_subspaces,
                "codebook_size_l1": model.config.codebook_size_l1,
                "codebook_size_l2": model.config.codebook_size_l2,
                "refinement_iterations": model.config.refinement_iterations
            }
        }))
    }
}