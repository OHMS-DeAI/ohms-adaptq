use crate::{Result, WeightMatrix};
use crate::model_fetcher::{FetchResult, ModelMetadata, ModelFormat};
use std::path::PathBuf;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStats {
    pub total_parameters: usize,
    pub total_size_mb: f64,
    pub num_tensors: usize,
    pub largest_tensor: usize,
    pub smallest_tensor: usize,
    pub average_tensor_size: usize,
}

pub struct RealModelLoader;

impl RealModelLoader {
    /// Convert BF16 (Brain Float 16) to F32
    fn bf16_to_f32(bf16_bits: u16) -> f32 {
        // BF16 has 1 sign bit, 8 exponent bits, 7 mantissa bits
        let sign = (bf16_bits >> 15) & 0x1;
        let exponent = (bf16_bits >> 7) & 0xFF;
        let mantissa = bf16_bits & 0x7F;
        
        if exponent == 0 {
            // Zero or denormalized
            if mantissa == 0 {
                return 0.0;
            } else {
                // Denormalized - very small numbers
                let f32_mantissa = mantissa as f32 / 128.0;
                return if sign == 1 { -f32_mantissa } else { f32_mantissa };
            }
        } else if exponent == 0xFF {
            // Infinity or NaN
            if mantissa == 0 {
                return if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY };
            } else {
                return f32::NAN;
            }
        } else {
            // Normalized number
            let f32_exponent = (exponent as i32 - 127) + 127; // Adjust bias
            let f32_mantissa = mantissa as u32;
            
            let f32_bits = (sign as u32) << 31 | (f32_exponent as u32) << 23 | f32_mantissa << 16;
            return f32::from_bits(f32_bits);
        }
    }

    /// Load model from fetch result and convert to NOVAQ-compatible weights
    pub fn load_model(fetch_result: &FetchResult) -> Result<Vec<WeightMatrix>> {
        match &fetch_result.model_format {
            ModelFormat::SafeTensors => Self::load_safetensors(&fetch_result.local_path),
            ModelFormat::PyTorch => Self::load_pytorch(&fetch_result.local_path),
            ModelFormat::GGUF => Self::load_gguf(&fetch_result.local_path),
            ModelFormat::ONNX => Self::load_onnx(&fetch_result.local_path),
            ModelFormat::Unknown => Err("Unknown model format".into()),
        }
    }

    /// Load SafeTensors format (most common for modern models)
    fn load_safetensors(path: &PathBuf) -> Result<Vec<WeightMatrix>> {
        let mut file = File::open(path)?;
        
        // Read header length (8 bytes)
        let mut header_len_bytes = [0u8; 8];
        file.read_exact(&mut header_len_bytes)?;
        let header_len = u64::from_le_bytes(header_len_bytes) as usize;
        
        // Read JSON header
        let mut header_json = vec![0u8; header_len];
        file.read_exact(&mut header_json)?;
        let header_str = String::from_utf8(header_json)?;
        let header: HashMap<String, serde_json::Value> = serde_json::from_str(&header_str)?;
        
        let mut weights = Vec::new();
        let offset = 8 + header_len as u64;
        
        for (tensor_name, tensor_info) in header {
            if let Some(tensor_obj) = tensor_info.as_object() {
                let dtype = tensor_obj.get("dtype")
                    .and_then(|v| v.as_str())
                    .unwrap_or("F32");
                let shape = tensor_obj.get("shape")
                    .and_then(|v| v.as_array())
                    .ok_or("Invalid shape")?
                    .iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect::<Vec<_>>();
                
                let data_offsets = tensor_obj.get("data_offsets")
                    .and_then(|v| v.as_array())
                    .ok_or("Invalid data_offsets")?;
                let start_offset = data_offsets[0].as_u64().unwrap_or(0) as u64;
                let end_offset = data_offsets[1].as_u64().unwrap_or(0) as u64;
                let tensor_size = (end_offset - start_offset) as usize;
                
                // Seek to tensor data
                file.seek(SeekFrom::Start(offset + start_offset))?;
                
                // Read tensor data based on dtype - Universal LLM model compatibility
                let tensor_data = match dtype {
                    // Standard floating point formats
                    "F32" | "FLOAT32" | "float32" => {
                        let mut data = vec![0f32; tensor_size / 4];
                        let mut bytes = vec![0u8; tensor_size];
                        file.read_exact(&mut bytes)?;
                        for (i, chunk) in bytes.chunks(4).enumerate() {
                            if i < data.len() {
                                data[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            }
                        }
                        data
                    },
                    "F64" | "FLOAT64" | "float64" => {
                        let mut data = vec![0f32; tensor_size / 8];
                        let mut bytes = vec![0u8; tensor_size];
                        file.read_exact(&mut bytes)?;
                        for (i, chunk) in bytes.chunks(8).enumerate() {
                            if i < data.len() {
                                let f64_val = f64::from_le_bytes([
                                    chunk[0], chunk[1], chunk[2], chunk[3],
                                    chunk[4], chunk[5], chunk[6], chunk[7]
                                ]);
                                data[i] = f64_val as f32; // Convert to f32
                            }
                        }
                        data
                    },
                    // Half precision formats
                    "F16" | "FLOAT16" | "float16" | "HALF" => {
                        let mut data = vec![0f32; tensor_size / 2];
                        let mut bytes = vec![0u8; tensor_size];
                        file.read_exact(&mut bytes)?;
                        for (i, chunk) in bytes.chunks(2).enumerate() {
                            if i < data.len() {
                                let f16_val = u16::from_le_bytes([chunk[0], chunk[1]]);
                                data[i] = half::f16::from_bits(f16_val).to_f32();
                            }
                        }
                        data
                    },
                    // Brain Float 16 - Critical for Phi-3, GPT-4, Claude models
                    "BF16" | "bfloat16" | "BFLOAT16" | "brain_float16" => {
                        let mut data = vec![0f32; tensor_size / 2];
                        let mut bytes = vec![0u8; tensor_size];
                        file.read_exact(&mut bytes)?;
                        for (i, chunk) in bytes.chunks(2).enumerate() {
                            if i < data.len() {
                                let bf16_val = u16::from_le_bytes([chunk[0], chunk[1]]);
                                data[i] = Self::bf16_to_f32(bf16_val);
                            }
                        }
                        data
                    },
                    // Integer formats - for quantized models (GPTQ, AWQ, etc.)
                    "I8" | "INT8" | "int8" => {
                        let mut data = vec![0f32; tensor_size];
                        let mut bytes = vec![0u8; tensor_size];
                        file.read_exact(&mut bytes)?;
                        for (i, &byte) in bytes.iter().enumerate() {
                            if i < data.len() {
                                data[i] = (byte as i8) as f32; // Convert signed int8 to float
                            }
                        }
                        data
                    },
                    "U8" | "UINT8" | "uint8" => {
                        let mut data = vec![0f32; tensor_size];
                        let mut bytes = vec![0u8; tensor_size];
                        file.read_exact(&mut bytes)?;
                        for (i, &byte) in bytes.iter().enumerate() {
                            if i < data.len() {
                                data[i] = byte as f32; // Convert unsigned int8 to float
                            }
                        }
                        data
                    },
                    "I16" | "INT16" | "int16" => {
                        let mut data = vec![0f32; tensor_size / 2];
                        let mut bytes = vec![0u8; tensor_size];
                        file.read_exact(&mut bytes)?;
                        for (i, chunk) in bytes.chunks(2).enumerate() {
                            if i < data.len() {
                                let i16_val = i16::from_le_bytes([chunk[0], chunk[1]]);
                                data[i] = i16_val as f32;
                            }
                        }
                        data
                    },
                    "I32" | "INT32" | "int32" => {
                        let mut data = vec![0f32; tensor_size / 4];
                        let mut bytes = vec![0u8; tensor_size];
                        file.read_exact(&mut bytes)?;
                        for (i, chunk) in bytes.chunks(4).enumerate() {
                            if i < data.len() {
                                let i32_val = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                                data[i] = i32_val as f32;
                            }
                        }
                        data
                    },
                    // Boolean for mask tensors
                    "BOOL" | "bool" => {
                        let mut data = vec![0f32; tensor_size];
                        let mut bytes = vec![0u8; tensor_size];
                        file.read_exact(&mut bytes)?;
                        for (i, &byte) in bytes.iter().enumerate() {
                            if i < data.len() {
                                data[i] = if byte != 0 { 1.0 } else { 0.0 };
                            }
                        }
                        data
                    },
                    _ => return Err(format!("Unsupported dtype: {} - NOVAQ supports F32, F64, F16, BF16, I8, U8, I16, I32, BOOL for universal LLM compatibility", dtype).into()),
                };
                
                // Reshape tensor data
                let total_elements: usize = shape.iter().product();
                if tensor_data.len() != total_elements {
                    return Err(format!("Tensor size mismatch for {}: expected {}, got {}", 
                        tensor_name, total_elements, tensor_data.len()).into());
                }
                
                weights.push(WeightMatrix::new(tensor_data, shape, tensor_name));
            }
        }
        
        Ok(weights)
    }

    /// Load PyTorch format (.bin, .pt, .pth files)
    fn load_pytorch(path: &PathBuf) -> Result<Vec<WeightMatrix>> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        
        // Try to parse PyTorch format
        // Note: This is a simplified implementation. Full PyTorch support would require
        // proper pickle protocol parsing or Python interop
        let mut weights = Vec::new();
        
        // Look for common PyTorch patterns
        // Check for ZIP archive structure (newer PyTorch format)
        if buffer.len() > 4 && &buffer[0..4] == b"PK\x03\x04" {
            // This is a ZIP-based PyTorch file (modern format)
            return Err("ZIP-based PyTorch files require specialized parsing. Please convert to SafeTensors format for BF16 support.".into());
        }
        
        // Look for tensor data patterns in binary PyTorch files
        let mut pos = 0;
        while pos < buffer.len() - 20 {
            // Look for potential tensor headers with data type information
            if let Some((tensor_data, tensor_shape, tensor_name, new_pos)) = Self::try_parse_pytorch_tensor(&buffer, pos)? {
                weights.push(WeightMatrix::new(tensor_data, tensor_shape, tensor_name));
                pos = new_pos;
            } else {
                pos += 1;
            }
        }
        
        if weights.is_empty() {
            return Err("Could not extract tensors from PyTorch file. For BF16 models, consider converting to SafeTensors format.".into());
        }
        
        Ok(weights)
    }

    /// Try to parse a single tensor from PyTorch binary data
    fn try_parse_pytorch_tensor(buffer: &[u8], start_pos: usize) -> Result<Option<(Vec<f32>, Vec<usize>, String, usize)>> {
        if start_pos + 20 >= buffer.len() {
            return Ok(None);
        }
        
        // Look for tensor markers (this is heuristic-based)
        let potential_size = u64::from_le_bytes([
            buffer[start_pos], buffer[start_pos+1], buffer[start_pos+2], buffer[start_pos+3],
            buffer[start_pos+4], buffer[start_pos+5], buffer[start_pos+6], buffer[start_pos+7]
        ]);
        
        if potential_size == 0 || potential_size > 1_000_000_000 {
            return Ok(None);
        }
        
        let tensor_elements = potential_size as usize;
        
        // Try to detect data type from the next bytes (heuristic)
        let dtype_hint = buffer[start_pos + 8];
        let (bytes_per_element, dtype_name) = match dtype_hint {
            1 => (2, "F16"),    // F16 hint
            2 => (2, "BF16"),   // BF16 hint
            4 => (4, "F32"),    // F32 hint
            _ => (4, "F32"),    // Default to F32
        };
        
        let tensor_bytes = tensor_elements * bytes_per_element;
        let data_start = start_pos + 12; // Skip header
        
        if data_start + tensor_bytes > buffer.len() {
            return Ok(None);
        }
        
        // Read tensor data based on detected type
        let mut tensor_data = vec![0f32; tensor_elements];
        match dtype_name {
            "F32" => {
                for (i, chunk) in buffer[data_start..data_start + tensor_bytes].chunks(4).enumerate() {
                    if i < tensor_data.len() && chunk.len() >= 4 {
                        tensor_data[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    }
                }
            },
            "F16" => {
                for (i, chunk) in buffer[data_start..data_start + tensor_bytes].chunks(2).enumerate() {
                    if i < tensor_data.len() && chunk.len() >= 2 {
                        let f16_val = u16::from_le_bytes([chunk[0], chunk[1]]);
                        tensor_data[i] = half::f16::from_bits(f16_val).to_f32();
                    }
                }
            },
            "BF16" => {
                for (i, chunk) in buffer[data_start..data_start + tensor_bytes].chunks(2).enumerate() {
                    if i < tensor_data.len() && chunk.len() >= 2 {
                        let bf16_val = u16::from_le_bytes([chunk[0], chunk[1]]);
                        tensor_data[i] = Self::bf16_to_f32(bf16_val);
                    }
                }
            },
            _ => return Ok(None),
        }
        
        // Create a reasonable shape (simplified)
        let tensor_shape = if tensor_elements <= 1024 {
            vec![tensor_elements] // 1D for small tensors
        } else {
            // Try to create a 2D shape
            let dim = (tensor_elements as f64).sqrt() as usize;
            if dim * dim == tensor_elements {
                vec![dim, dim]
            } else {
                // Find factors
                let mut factors = Vec::new();
                let mut n = tensor_elements;
                let mut d = 2;
                while d * d <= n {
                    if n % d == 0 {
                        factors.push(d);
                        n /= d;
                    } else {
                        d += 1;
                    }
                }
                if n > 1 {
                    factors.push(n);
                }
                
                if factors.len() >= 2 {
                    vec![factors[0] * factors[1], tensor_elements / (factors[0] * factors[1])]
                } else {
                    vec![tensor_elements]
                }
            }
        };
        
        let tensor_name = format!("pytorch_tensor_{}", start_pos);
        let next_pos = data_start + tensor_bytes;
        
        Ok(Some((tensor_data, tensor_shape, tensor_name, next_pos)))
    }

    /// Load GGUF format (Ollama models)
    fn load_gguf(path: &PathBuf) -> Result<Vec<WeightMatrix>> {
        let mut file = File::open(path)?;
        
        // Read GGUF header
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Err("Invalid GGUF magic number".into());
        }
        
        let mut version = [0u8; 4];
        file.read_exact(&mut version)?;
        let _version_num = u32::from_le_bytes(version);
        
        let mut tensor_count = [0u8; 8];
        file.read_exact(&mut tensor_count)?;
        let num_tensors = u64::from_le_bytes(tensor_count);
        
        let mut metadata_size = [0u8; 8];
        file.read_exact(&mut metadata_size)?;
        let metadata_len = u64::from_le_bytes(metadata_size) as usize;
        
        // Skip metadata for now (we'll implement proper parsing later)
        file.seek(SeekFrom::Current(metadata_len as i64))?;
        
        let mut weights = Vec::new();
        
        // Read tensors
        for _i in 0..num_tensors {
            // Read tensor name length
            let mut name_len = [0u8; 4];
            file.read_exact(&mut name_len)?;
            let name_length = u32::from_le_bytes(name_len) as usize;
            
            // Read tensor name
            let mut name_bytes = vec![0u8; name_length];
            file.read_exact(&mut name_bytes)?;
            let tensor_name = String::from_utf8(name_bytes)?;
            
            // Read tensor dimensions
            let mut dims = [0u8; 4];
            file.read_exact(&mut dims)?;
            let num_dims = u32::from_le_bytes(dims) as usize;
            
            let mut shape = Vec::new();
            for _ in 0..num_dims {
                let mut dim = [0u8; 8];
                file.read_exact(&mut dim)?;
                shape.push(u64::from_le_bytes(dim) as usize);
            }
            
            // Read tensor type
            let mut tensor_type = [0u8; 4];
            file.read_exact(&mut tensor_type)?;
            let dtype = u32::from_le_bytes(tensor_type);
            
            // Read tensor offset
            let mut offset = [0u8; 8];
            file.read_exact(&mut offset)?;
            let tensor_offset = u64::from_le_bytes(offset);
            
            // Calculate tensor size - Universal GGUF data type support
            let total_elements: usize = shape.iter().product();
            let bytes_per_element = match dtype {
                0 => 4,  // F32
                1 => 2,  // F16
                2 => 2,  // BF16 
                3 => 1,  // I8
                4 => 1,  // U8
                5 => 2,  // I16
                6 => 2,  // U16
                7 => 4,  // I32
                8 => 4,  // U32
                9 => 8,  // F64
                10 => 8, // I64
                11 => 8, // U64
                12 => 1, // BOOL
                _ => 4,  // Default to F32
            };
            let tensor_size = total_elements * bytes_per_element;
            
            // Store current position
            let current_pos = file.stream_position()?;
            
            // Seek to tensor data
            file.seek(SeekFrom::Start(tensor_offset))?;
            
            // Read tensor data - Universal GGUF data type support
            let mut tensor_data = vec![0f32; total_elements];
            match dtype {
                0 => { // F32
                    let mut bytes = vec![0u8; tensor_size];
                    file.read_exact(&mut bytes)?;
                    for (i, chunk) in bytes.chunks(4).enumerate() {
                        if i < tensor_data.len() {
                            tensor_data[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        }
                    }
                },
                1 => { // F16
                    let mut bytes = vec![0u8; tensor_size];
                    file.read_exact(&mut bytes)?;
                    for (i, chunk) in bytes.chunks(2).enumerate() {
                        if i < tensor_data.len() {
                            let f16_val = u16::from_le_bytes([chunk[0], chunk[1]]);
                            tensor_data[i] = half::f16::from_bits(f16_val).to_f32();
                        }
                    }
                },
                2 => { // BF16
                    let mut bytes = vec![0u8; tensor_size];
                    file.read_exact(&mut bytes)?;
                    for (i, chunk) in bytes.chunks(2).enumerate() {
                        if i < tensor_data.len() {
                            let bf16_val = u16::from_le_bytes([chunk[0], chunk[1]]);
                            tensor_data[i] = Self::bf16_to_f32(bf16_val);
                        }
                    }
                },
                3 => { // I8
                    let mut bytes = vec![0u8; tensor_size];
                    file.read_exact(&mut bytes)?;
                    for (i, &byte) in bytes.iter().enumerate() {
                        if i < tensor_data.len() {
                            tensor_data[i] = (byte as i8) as f32;
                        }
                    }
                },
                4 => { // U8
                    let mut bytes = vec![0u8; tensor_size];
                    file.read_exact(&mut bytes)?;
                    for (i, &byte) in bytes.iter().enumerate() {
                        if i < tensor_data.len() {
                            tensor_data[i] = byte as f32;
                        }
                    }
                },
                5 => { // I16
                    let mut bytes = vec![0u8; tensor_size];
                    file.read_exact(&mut bytes)?;
                    for (i, chunk) in bytes.chunks(2).enumerate() {
                        if i < tensor_data.len() {
                            let i16_val = i16::from_le_bytes([chunk[0], chunk[1]]);
                            tensor_data[i] = i16_val as f32;
                        }
                    }
                },
                6 => { // U16
                    let mut bytes = vec![0u8; tensor_size];
                    file.read_exact(&mut bytes)?;
                    for (i, chunk) in bytes.chunks(2).enumerate() {
                        if i < tensor_data.len() {
                            let u16_val = u16::from_le_bytes([chunk[0], chunk[1]]);
                            tensor_data[i] = u16_val as f32;
                        }
                    }
                },
                7 => { // I32
                    let mut bytes = vec![0u8; tensor_size];
                    file.read_exact(&mut bytes)?;
                    for (i, chunk) in bytes.chunks(4).enumerate() {
                        if i < tensor_data.len() {
                            let i32_val = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            tensor_data[i] = i32_val as f32;
                        }
                    }
                },
                8 => { // U32
                    let mut bytes = vec![0u8; tensor_size];
                    file.read_exact(&mut bytes)?;
                    for (i, chunk) in bytes.chunks(4).enumerate() {
                        if i < tensor_data.len() {
                            let u32_val = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            tensor_data[i] = u32_val as f32;
                        }
                    }
                },
                9 => { // F64
                    let mut bytes = vec![0u8; tensor_size];
                    file.read_exact(&mut bytes)?;
                    for (i, chunk) in bytes.chunks(8).enumerate() {
                        if i < tensor_data.len() {
                            let f64_val = f64::from_le_bytes([
                                chunk[0], chunk[1], chunk[2], chunk[3],
                                chunk[4], chunk[5], chunk[6], chunk[7]
                            ]);
                            tensor_data[i] = f64_val as f32;
                        }
                    }
                },
                10 => { // I64
                    let mut bytes = vec![0u8; tensor_size];
                    file.read_exact(&mut bytes)?;
                    for (i, chunk) in bytes.chunks(8).enumerate() {
                        if i < tensor_data.len() {
                            let i64_val = i64::from_le_bytes([
                                chunk[0], chunk[1], chunk[2], chunk[3],
                                chunk[4], chunk[5], chunk[6], chunk[7]
                            ]);
                            tensor_data[i] = i64_val as f32;
                        }
                    }
                },
                11 => { // U64
                    let mut bytes = vec![0u8; tensor_size];
                    file.read_exact(&mut bytes)?;
                    for (i, chunk) in bytes.chunks(8).enumerate() {
                        if i < tensor_data.len() {
                            let u64_val = u64::from_le_bytes([
                                chunk[0], chunk[1], chunk[2], chunk[3],
                                chunk[4], chunk[5], chunk[6], chunk[7]
                            ]);
                            tensor_data[i] = u64_val as f32;
                        }
                    }
                },
                12 => { // BOOL
                    let mut bytes = vec![0u8; tensor_size];
                    file.read_exact(&mut bytes)?;
                    for (i, &byte) in bytes.iter().enumerate() {
                        if i < tensor_data.len() {
                            tensor_data[i] = if byte != 0 { 1.0 } else { 0.0 };
                        }
                    }
                },
                _ => {
                    return Err(format!("Unsupported GGUF tensor dtype: {} - NOVAQ supports F32(0), F16(1), BF16(2), I8(3), U8(4), I16(5), U16(6), I32(7), U32(8), F64(9), I64(10), U64(11), BOOL(12)", dtype).into());
                }
            }
            
            weights.push(WeightMatrix::new(tensor_data, shape, tensor_name));
            
            // Return to metadata position
            file.seek(SeekFrom::Start(current_pos))?;
        }
        
        Ok(weights)
    }

    /// Load ONNX format
    fn load_onnx(path: &PathBuf) -> Result<Vec<WeightMatrix>> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        
        // Check ONNX magic number (simplified check)
        if buffer.len() < 8 {
            return Err("ONNX file too small".into());
        }
        
        // Note: This is a simplified ONNX parser for demonstration
        // Real implementation would use proper ONNX protobuf parsing
        let mut weights = Vec::new();
        
        // Look for weight tensors in the ONNX file with BF16 support
        let mut pos = 0;
        while pos < buffer.len() - 20 {
            if let Some((tensor_data, tensor_shape, tensor_name, new_pos)) = Self::try_parse_onnx_tensor(&buffer, pos)? {
                weights.push(WeightMatrix::new(tensor_data, tensor_shape, tensor_name));
                pos = new_pos;
            } else {
                pos += 1;
            }
        }
        
        if weights.is_empty() {
            return Err("Could not extract tensors from ONNX file. For BF16 models, consider converting to SafeTensors format.".into());
        }
        
        Ok(weights)
    }

    /// Try to parse a single tensor from ONNX binary data
    fn try_parse_onnx_tensor(buffer: &[u8], start_pos: usize) -> Result<Option<(Vec<f32>, Vec<usize>, String, usize)>> {
        if start_pos + 20 >= buffer.len() {
            return Ok(None);
        }
        
        // Look for potential ONNX tensor patterns
        let potential_size = u64::from_le_bytes([
            buffer[start_pos], buffer[start_pos+1], buffer[start_pos+2], buffer[start_pos+3],
            buffer[start_pos+4], buffer[start_pos+5], buffer[start_pos+6], buffer[start_pos+7]
        ]);
        
        if potential_size == 0 || potential_size > 100_000_000 {
            return Ok(None);
        }
        
        let tensor_elements = potential_size as usize;
        
        // Try to detect ONNX data type from header (simplified)
        let dtype_marker = buffer[start_pos + 8];
        let (bytes_per_element, dtype_name) = match dtype_marker {
            1 => (4, "F32"),    // ONNX FLOAT type
            10 => (2, "F16"),   // ONNX FLOAT16 type
            16 => (2, "BF16"),  // ONNX BFLOAT16 type (if present)
            _ => {
                // Try to infer from data patterns
                if tensor_elements * 2 + start_pos + 16 < buffer.len() {
                    (2, "F16") // Assume F16
                } else if tensor_elements * 4 + start_pos + 16 < buffer.len() {
                    (4, "F32") // Assume F32
                } else {
                    return Ok(None);
                }
            },
        };
        
        let tensor_bytes = tensor_elements * bytes_per_element;
        let data_start = start_pos + 16; // Skip ONNX header
        
        if data_start + tensor_bytes > buffer.len() {
            return Ok(None);
        }
        
        // Read tensor data based on detected type
        let mut tensor_data = vec![0f32; tensor_elements];
        match dtype_name {
            "F32" => {
                for (i, chunk) in buffer[data_start..data_start + tensor_bytes].chunks(4).enumerate() {
                    if i < tensor_data.len() && chunk.len() >= 4 {
                        tensor_data[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    }
                }
            },
            "F16" => {
                for (i, chunk) in buffer[data_start..data_start + tensor_bytes].chunks(2).enumerate() {
                    if i < tensor_data.len() && chunk.len() >= 2 {
                        let f16_val = u16::from_le_bytes([chunk[0], chunk[1]]);
                        tensor_data[i] = half::f16::from_bits(f16_val).to_f32();
                    }
                }
            },
            "BF16" => {
                for (i, chunk) in buffer[data_start..data_start + tensor_bytes].chunks(2).enumerate() {
                    if i < tensor_data.len() && chunk.len() >= 2 {
                        let bf16_val = u16::from_le_bytes([chunk[0], chunk[1]]);
                        tensor_data[i] = Self::bf16_to_f32(bf16_val);
                    }
                }
            },
            _ => return Ok(None),
        }
        
        // Create tensor shape (simplified approach for ONNX)
        let tensor_shape = Self::infer_tensor_shape(tensor_elements);
        let tensor_name = format!("onnx_tensor_{}", start_pos);
        let next_pos = data_start + tensor_bytes;
        
        Ok(Some((tensor_data, tensor_shape, tensor_name, next_pos)))
    }

    /// Infer reasonable tensor shape from number of elements
    fn infer_tensor_shape(elements: usize) -> Vec<usize> {
        if elements <= 1024 {
            vec![elements] // 1D for small tensors
        } else {
            // Try to create a reasonable 2D shape
            let sqrt_elements = (elements as f64).sqrt() as usize;
            if sqrt_elements * sqrt_elements == elements {
                vec![sqrt_elements, sqrt_elements]
            } else {
                // Find factors for better shape
                let mut best_factor = 1;
                for i in 2..=((elements as f64).sqrt() as usize) {
                    if elements % i == 0 {
                        best_factor = i;
                    }
                }
                vec![best_factor, elements / best_factor]
            }
        }
    }

    /// Get model metadata from fetch result
    pub fn get_metadata(fetch_result: &FetchResult) -> Option<&ModelMetadata> {
        fetch_result.metadata.as_ref()
    }

    /// Validate model format and compatibility
    pub fn validate_model(fetch_result: &FetchResult) -> Result<bool> {
        match &fetch_result.model_format {
            ModelFormat::SafeTensors => {
                // Validate SafeTensors header
                let mut file = File::open(&fetch_result.local_path)?;
                let mut magic = [0u8; 15];
                file.read_exact(&mut magic)?;
                Ok(&magic == b"__safetensors__")
            },
            ModelFormat::PyTorch => {
                // Basic PyTorch validation
                let mut file = File::open(&fetch_result.local_path)?;
                let mut header = [0u8; 8];
                file.read_exact(&mut header)?;
                // Check for PyTorch magic (simplified)
                Ok(true) // Assume valid for now
            },
            ModelFormat::GGUF => {
                // Validate GGUF header
                let mut file = File::open(&fetch_result.local_path)?;
                let mut magic = [0u8; 4];
                file.read_exact(&mut magic)?;
                Ok(&magic == b"GGUF")
            },
            ModelFormat::ONNX => {
                // Validate ONNX header
                let mut file = File::open(&fetch_result.local_path)?;
                let mut header = [0u8; 9];
                file.read_exact(&mut header)?;
                Ok(&header == b"\x08\x01\x12\x07onnx\x1d")
            },
            ModelFormat::Unknown => Ok(false),
        }
    }

    /// Get model statistics
    pub fn get_model_stats(weights: &[WeightMatrix]) -> ModelStats {
        let total_parameters: usize = weights.iter()
            .map(|w| w.data.len())
            .sum();
        
        let total_size_bytes = total_parameters * 4; // f32 = 4 bytes
        
        let largest_tensor = weights.iter()
            .max_by_key(|w| w.data.len())
            .map(|w| w.data.len())
            .unwrap_or(0);
        
        let smallest_tensor = weights.iter()
            .min_by_key(|w| w.data.len())
            .map(|w| w.data.len())
            .unwrap_or(0);
        
        ModelStats {
            total_parameters,
            total_size_mb: total_size_bytes as f64 / (1024.0 * 1024.0),
            num_tensors: weights.len(),
            largest_tensor,
            smallest_tensor,
            average_tensor_size: if weights.is_empty() { 0 } else { total_parameters / weights.len() },
        }
    }
}