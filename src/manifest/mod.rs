use crate::quantization::QuantizationResult;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub model_id: String,
    pub version: String,
    pub chunks: Vec<ChunkInfo>,
    pub digest: String,
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
}

pub struct ManifestBuilder;

impl ManifestBuilder {
    pub fn from_quantization_result(result: &QuantizationResult, version: &str) -> crate::Result<Manifest> {
        let mut chunks = Vec::new();
        let mut offset = 0;

        for chunk in &result.chunks {
            chunks.push(ChunkInfo {
                id: chunk.id.clone(),
                offset,
                size: chunk.size,
                sha256: chunk.sha256.clone(),
            });
            offset += chunk.size;
        }

        // Calculate manifest digest
        let manifest_data = serde_json::to_string(&chunks)?;
        let mut hasher = Sha256::new();
        hasher.update(manifest_data.as_bytes());
        let digest = hex::encode(hasher.finalize());

        Ok(Manifest {
            model_id: result.metadata.model_id.clone(),
            version: version.to_string(),
            chunks,
            digest,
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
        }
    }

    pub fn verify_manifest_integrity(manifest: &Manifest, actual_chunks: &[Vec<u8>]) -> bool {
        if manifest.chunks.len() != actual_chunks.len() {
            return false;
        }

        for (chunk_info, actual_data) in manifest.chunks.iter().zip(actual_chunks.iter()) {
            // Verify size
            if chunk_info.size != actual_data.len() {
                return false;
            }

            // Verify hash
            let mut hasher = Sha256::new();
            hasher.update(actual_data);
            let actual_hash = hex::encode(hasher.finalize());
            
            if chunk_info.sha256 != actual_hash {
                return false;
            }
        }

        true
    }
}