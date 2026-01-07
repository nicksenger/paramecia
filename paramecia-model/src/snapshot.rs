//! Model state snapshotting for Qwen3-Next inference.
//!
//! Saves and restores complete inference state including:
//! - KV cache for full attention layers (quantized or float)
//! - DeltaNet recurrent state for linear attention layers
//! - Token history and position tracking
//! - Model configuration for validation

use crc32fast::Hasher;
use paramecia_core::quantized::GgmlDType;
use paramecia_core::{DType, Device, Result, Tensor};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

// Magic constants
const MAGIC: &[u8; 8] = b"QWENSNAP";
const MAGIC_END: &[u8; 8] = b"SNAPEND\0";
const VERSION: u32 = 1;

// Section type constants
const SECTION_CONFIG: u32 = 0x01;
const SECTION_TOKENS: u32 = 0x03;
const SECTION_LAYER_STATE: u32 = 0x04;

// Flags
const FLAG_HAS_QUANTIZED_KV: u32 = 1 << 0;
const FLAG_HAS_RECURRENT_STATE: u32 = 1 << 1;

/// Snapshot file header (64 bytes)
#[repr(C)]
#[derive(Debug, Clone)]
struct SnapshotHeader {
    magic: [u8; 8],
    version: u32,
    flags: u32,
    config_hash: u64,
    state_position: u64,
    num_layers: u32,
    timestamp: u64,
    reserved: [u8; 20], // Increased from 16 to 20 to make header exactly 64 bytes
}

impl SnapshotHeader {
    fn write<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&self.magic)?;
        writer.write_all(&self.version.to_le_bytes())?;
        writer.write_all(&self.flags.to_le_bytes())?;
        writer.write_all(&self.config_hash.to_le_bytes())?;
        writer.write_all(&self.state_position.to_le_bytes())?;
        writer.write_all(&self.num_layers.to_le_bytes())?;
        writer.write_all(&self.timestamp.to_le_bytes())?;
        writer.write_all(&self.reserved)?;
        Ok(())
    }

    fn read<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;

        if &magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid magic bytes: expected {:?}, got {:?}", MAGIC, magic),
            ));
        }

        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        let version = u32::from_le_bytes(buf);

        reader.read_exact(&mut buf)?;
        let flags = u32::from_le_bytes(buf);

        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8)?;
        let config_hash = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let state_position = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf)?;
        let num_layers = u32::from_le_bytes(buf);

        reader.read_exact(&mut buf8)?;
        let timestamp = u64::from_le_bytes(buf8);

        let mut reserved = [0u8; 20];
        reader.read_exact(&mut reserved)?;

        Ok(Self {
            magic,
            version,
            flags,
            config_hash,
            state_position,
            num_layers,
            timestamp,
            reserved,
        })
    }
}

/// Cached tensor (quantized or float)
#[derive(Debug, Clone)]
pub enum CachedTensor {
    Quantized {
        data: Vec<u8>,
        dtype: GgmlDType,
        shape: Vec<usize>,
        /// Layout metadata to recreate cache exactly
        bytes_per_row: usize,
        padded_head_dim: usize,
        max_seq_len: usize,
    },
    Float {
        data: Vec<u8>,
        dtype: DType,
        shape: Vec<usize>,
    },
}

impl CachedTensor {
    /// Create from a quantized buffer with layout metadata
    pub fn from_quantized(
        data: Vec<u8>,
        dtype: GgmlDType,
        shape: Vec<usize>,
        bytes_per_row: usize,
        padded_head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        Self::Quantized {
            data,
            dtype,
            shape,
            bytes_per_row,
            padded_head_dim,
            max_seq_len,
        }
    }

    /// Create from a float tensor
    pub fn from_tensor(tensor: &Tensor) -> Result<Self> {
        let dtype = tensor.dtype();
        let shape = tensor.dims().to_vec();

        // Convert to contiguous CPU tensor and flatten to 1D
        let tensor = tensor.to_device(&Device::Cpu)?.contiguous()?;
        let tensor = tensor.flatten_all()?;

        // Extract raw bytes
        let data = match dtype {
            DType::F16 => {
                let vec = tensor.to_vec1::<half::f16>()?;
                vec.iter().flat_map(|&x| x.to_le_bytes()).collect()
            }
            DType::BF16 => {
                let vec = tensor.to_vec1::<half::bf16>()?;
                vec.iter().flat_map(|&x| x.to_le_bytes()).collect()
            }
            DType::F32 => {
                let vec = tensor.to_vec1::<f32>()?;
                vec.iter().flat_map(|&x| x.to_le_bytes()).collect()
            }
            DType::F64 => {
                let vec = tensor.to_vec1::<f64>()?;
                vec.iter().flat_map(|&x| x.to_le_bytes()).collect()
            }
            _ => paramecia_core::bail!("Unsupported dtype for snapshotting: {:?}", dtype),
        };

        Ok(Self::Float { data, dtype, shape })
    }

    /// Restore to a tensor on the target device
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        match self {
            Self::Float { data, dtype, shape } => {
                // Reconstruct tensor from raw bytes
                let tensor = match dtype {
                    DType::F16 => {
                        let values: Vec<half::f16> = data
                            .chunks_exact(2)
                            .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]))
                            .collect();
                        Tensor::from_vec(values, &shape[..], &Device::Cpu)?
                    }
                    DType::BF16 => {
                        let values: Vec<half::bf16> = data
                            .chunks_exact(2)
                            .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]))
                            .collect();
                        Tensor::from_vec(values, &shape[..], &Device::Cpu)?
                    }
                    DType::F32 => {
                        let values: Vec<f32> = data
                            .chunks_exact(4)
                            .map(|chunk| {
                                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                            })
                            .collect();
                        Tensor::from_vec(values, &shape[..], &Device::Cpu)?
                    }
                    DType::F64 => {
                        let values: Vec<f64> = data
                            .chunks_exact(8)
                            .map(|chunk| {
                                f64::from_le_bytes([
                                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5],
                                    chunk[6], chunk[7],
                                ])
                            })
                            .collect();
                        Tensor::from_vec(values, &shape[..], &Device::Cpu)?
                    }
                    _ => paramecia_core::bail!("Unsupported dtype: {:?}", dtype),
                };

                // Move to target device
                tensor.to_device(device)
            }
            Self::Quantized { .. } => {
                paramecia_core::bail!("Quantized tensors must be handled separately")
            }
        }
    }
}

/// Layer state (full attention or linear attention)
#[derive(Debug, Clone)]
pub enum LayerState {
    FullAttention {
        k_cache: Option<CachedTensor>,
        v_cache: Option<CachedTensor>,
        seq_len: usize,
    },
    LinearAttention {
        ssm_state: Option<CachedTensor>,
        conv_state: Option<CachedTensor>,
        gate_cumsum_offset: Option<CachedTensor>,
    },
    Empty,
}

/// Complete snapshot state
#[derive(Debug)]
pub struct Snapshot {
    pub state_position: usize,
    pub tokens: Vec<u32>,
    pub layer_states: Vec<LayerState>,
    pub config: SnapshotConfig,
}

/// Model configuration for validation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SnapshotConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rope_freq_base: f64,
    pub n_rot: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub ssm_d_inner: usize,
    pub ssm_d_state: usize,
    pub recurrent_layers: Vec<bool>,
}

impl SnapshotConfig {
    /// Compute hash of critical configuration fields
    fn compute_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.num_layers.hash(&mut hasher);
        self.hidden_size.hash(&mut hasher);
        self.num_attention_heads.hash(&mut hasher);
        self.num_key_value_heads.hash(&mut hasher);
        self.head_dim.hash(&mut hasher);
        hasher.finish()
    }
}

/// Convert GgmlDType to GGML protocol u32 value
fn ggml_dtype_to_u32(dtype: GgmlDType) -> u32 {
    match dtype {
        GgmlDType::F32 => 0,
        GgmlDType::F16 => 1,
        GgmlDType::BF16 => 30,
        GgmlDType::Q4_0 => 2,
        GgmlDType::Q4_1 => 3,
        GgmlDType::Q5_0 => 6,
        GgmlDType::Q5_1 => 7,
        GgmlDType::Q8_0 => 8,
        GgmlDType::Q8_1 => 9,
        GgmlDType::Q2K => 10,
        GgmlDType::Q3K => 11,
        GgmlDType::Q4K => 12,
        GgmlDType::Q5K => 13,
        GgmlDType::Q6K => 14,
        GgmlDType::Q8K => 15,
    }
}

/// Write a section header
fn write_section_header<W: Write>(
    writer: &mut W,
    section_type: u32,
    section_size: u64,
) -> std::io::Result<()> {
    writer.write_all(&section_type.to_le_bytes())?;
    writer.write_all(&section_size.to_le_bytes())?;
    Ok(())
}

/// Read a section header
fn read_section_header<R: Read>(reader: &mut R) -> std::io::Result<(u32, u64)> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let section_type = u32::from_le_bytes(buf);

    let mut buf8 = [0u8; 8];
    reader.read_exact(&mut buf8)?;
    let section_size = u64::from_le_bytes(buf8);

    Ok((section_type, section_size))
}

/// Write a tensor to the file
fn write_tensor<W: Write>(writer: &mut W, tensor: &CachedTensor) -> std::io::Result<()> {
    match tensor {
        CachedTensor::Quantized {
            data,
            dtype,
            shape,
            bytes_per_row,
            padded_head_dim,
            max_seq_len,
        } => {
            // Type: 0 = quantized
            writer.write_all(&0u8.to_le_bytes())?;
            // GGML dtype (as u32)
            writer.write_all(&ggml_dtype_to_u32(*dtype).to_le_bytes())?;
            // Shape
            writer.write_all(&(shape.len() as u32).to_le_bytes())?;
            for &dim in shape {
                writer.write_all(&(dim as u64).to_le_bytes())?;
            }
            // Data length and bytes
            writer.write_all(&(data.len() as u64).to_le_bytes())?;
            writer.write_all(data)?;
            // Layout metadata (for exact cache recreation)
            writer.write_all(&(*bytes_per_row as u64).to_le_bytes())?;
            writer.write_all(&(*padded_head_dim as u64).to_le_bytes())?;
            writer.write_all(&(*max_seq_len as u64).to_le_bytes())?;
        }
        CachedTensor::Float { data, dtype, shape } => {
            // Type: 1 = float
            writer.write_all(&1u8.to_le_bytes())?;
            // DType (as u8)
            let dtype_u8 = match dtype {
                DType::F16 => 0u8,
                DType::BF16 => 1u8,
                DType::F32 => 2u8,
                DType::F64 => 3u8,
                _ => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Unsupported dtype: {:?}", dtype),
                    ))
                }
            };
            writer.write_all(&dtype_u8.to_le_bytes())?;
            writer.write_all(&[0u8; 3])?; // Padding
                                          // Shape
            writer.write_all(&(shape.len() as u32).to_le_bytes())?;
            for &dim in shape {
                writer.write_all(&(dim as u64).to_le_bytes())?;
            }
            // Data length and bytes
            writer.write_all(&(data.len() as u64).to_le_bytes())?;
            writer.write_all(data)?;
        }
    }
    Ok(())
}

/// Read a tensor from the file
fn read_tensor<R: Read>(reader: &mut R) -> std::io::Result<CachedTensor> {
    let mut buf1 = [0u8; 1];
    reader.read_exact(&mut buf1)?;
    let tensor_type = buf1[0];

    match tensor_type {
        0 => {
            // Quantized
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            let dtype_u32 = u32::from_le_bytes(buf);
            let dtype = match dtype_u32 {
                8 => GgmlDType::Q8_0,
                10 => GgmlDType::Q2K,
                11 => GgmlDType::Q3K,
                12 => GgmlDType::Q4K,
                13 => GgmlDType::Q5K,
                14 => GgmlDType::Q6K,
                _ => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Unsupported GGML dtype: {}", dtype_u32),
                    ))
                }
            };

            reader.read_exact(&mut buf)?;
            let shape_len = u32::from_le_bytes(buf) as usize;
            let mut shape = Vec::with_capacity(shape_len);
            for _ in 0..shape_len {
                let mut buf8 = [0u8; 8];
                reader.read_exact(&mut buf8)?;
                shape.push(u64::from_le_bytes(buf8) as usize);
            }

            let mut buf8 = [0u8; 8];
            reader.read_exact(&mut buf8)?;
            let data_len = u64::from_le_bytes(buf8) as usize;

            // Sanity check: reject unreasonably large tensors (>4GB)
            if data_len > 4_000_000_000 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Tensor data too large: {} bytes", data_len),
                ));
            }

            let mut data = vec![0u8; data_len];
            reader.read_exact(&mut data)?;

            // Read layout metadata
            let mut buf8 = [0u8; 8];
            reader.read_exact(&mut buf8)?;
            let bytes_per_row = u64::from_le_bytes(buf8) as usize;

            reader.read_exact(&mut buf8)?;
            let padded_head_dim = u64::from_le_bytes(buf8) as usize;

            reader.read_exact(&mut buf8)?;
            let max_seq_len = u64::from_le_bytes(buf8) as usize;

            Ok(CachedTensor::Quantized {
                data,
                dtype,
                shape,
                bytes_per_row,
                padded_head_dim,
                max_seq_len,
            })
        }
        1 => {
            // Float
            let mut buf1 = [0u8; 1];
            reader.read_exact(&mut buf1)?;
            let dtype_u8 = buf1[0];
            let dtype = match dtype_u8 {
                0 => DType::F16,
                1 => DType::BF16,
                2 => DType::F32,
                3 => DType::F64,
                _ => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Unsupported dtype: {}", dtype_u8),
                    ))
                }
            };

            // Skip padding
            let mut padding = [0u8; 3];
            reader.read_exact(&mut padding)?;

            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            let shape_len = u32::from_le_bytes(buf) as usize;
            let mut shape = Vec::with_capacity(shape_len);
            for _ in 0..shape_len {
                let mut buf8 = [0u8; 8];
                reader.read_exact(&mut buf8)?;
                shape.push(u64::from_le_bytes(buf8) as usize);
            }

            let mut buf8 = [0u8; 8];
            reader.read_exact(&mut buf8)?;
            let data_len = u64::from_le_bytes(buf8) as usize;

            // Sanity check: reject unreasonably large tensors (>4GB)
            if data_len > 4_000_000_000 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Tensor data too large: {} bytes", data_len),
                ));
            }

            let mut data = vec![0u8; data_len];
            reader.read_exact(&mut data)?;

            Ok(CachedTensor::Float { data, dtype, shape })
        }
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unknown tensor type: {}", tensor_type),
        )),
    }
}

/// Save snapshot to file
pub fn save_snapshot<P: AsRef<Path>>(snapshot: &Snapshot, path: P) -> Result<()> {
    let file = File::create(path.as_ref()).map_err(paramecia_core::Error::Io)?;
    let mut writer = BufWriter::new(file);
    let mut crc = Hasher::new();

    // Determine flags
    let mut flags = 0u32;
    for layer_state in &snapshot.layer_states {
        match layer_state {
            LayerState::FullAttention {
                k_cache: Some(CachedTensor::Quantized { .. }),
                ..
            } => {
                flags |= FLAG_HAS_QUANTIZED_KV;
            }
            LayerState::FullAttention { .. } => {}
            LayerState::LinearAttention { ssm_state, .. } => {
                if ssm_state.is_some() {
                    flags |= FLAG_HAS_RECURRENT_STATE;
                }
            }
            _ => {}
        }
    }

    // Write header
    let header = SnapshotHeader {
        magic: *MAGIC,
        version: VERSION,
        flags,
        config_hash: snapshot.config.compute_hash(),
        state_position: snapshot.state_position as u64,
        num_layers: snapshot.config.num_layers as u32,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        reserved: [0u8; 20],
    };

    let mut header_bytes = Vec::new();
    header
        .write(&mut header_bytes)
        .map_err(paramecia_core::Error::Io)?;
    writer
        .write_all(&header_bytes)
        .map_err(paramecia_core::Error::Io)?;
    crc.update(&header_bytes);

    // Write config section
    let config_json = serde_json::to_string(&snapshot.config)
        .map_err(|e| paramecia_core::Error::Msg(format!("Failed to serialize config: {}", e)))?;
    let config_bytes = config_json.as_bytes();

    let mut section_bytes = Vec::new();
    write_section_header(
        &mut section_bytes,
        SECTION_CONFIG,
        config_bytes.len() as u64,
    )
    .map_err(paramecia_core::Error::Io)?;
    section_bytes.extend_from_slice(config_bytes);

    writer
        .write_all(&section_bytes)
        .map_err(paramecia_core::Error::Io)?;
    crc.update(&section_bytes);

    // Write tokens section
    let mut tokens_bytes = Vec::new();
    write_section_header(
        &mut tokens_bytes,
        SECTION_TOKENS,
        (snapshot.tokens.len() * 4 + 4) as u64,
    )
    .map_err(paramecia_core::Error::Io)?;
    tokens_bytes.extend_from_slice(&(snapshot.tokens.len() as u32).to_le_bytes());
    for &token in &snapshot.tokens {
        tokens_bytes.extend_from_slice(&token.to_le_bytes());
    }

    writer
        .write_all(&tokens_bytes)
        .map_err(paramecia_core::Error::Io)?;
    crc.update(&tokens_bytes);

    // Write layer state sections
    for (layer_idx, layer_state) in snapshot.layer_states.iter().enumerate() {
        let mut layer_bytes = Vec::new();

        // Write layer index and type
        layer_bytes.extend_from_slice(&(layer_idx as u32).to_le_bytes());

        match layer_state {
            LayerState::FullAttention {
                k_cache,
                v_cache,
                seq_len,
            } => {
                layer_bytes.push(0u8); // Layer type: Full
                layer_bytes.extend_from_slice(&[0u8; 3]); // Padding

                // Has cache flag
                let has_cache = k_cache.is_some() && v_cache.is_some();
                layer_bytes.push(if has_cache { 1u8 } else { 0u8 });
                layer_bytes.extend_from_slice(&[0u8; 7]); // Padding

                // Seq len
                layer_bytes.extend_from_slice(&(*seq_len as u64).to_le_bytes());

                if has_cache {
                    write_tensor(&mut layer_bytes, k_cache.as_ref().unwrap())
                        .map_err(paramecia_core::Error::Io)?;
                    write_tensor(&mut layer_bytes, v_cache.as_ref().unwrap())
                        .map_err(paramecia_core::Error::Io)?;
                }
            }
            LayerState::LinearAttention {
                ssm_state,
                conv_state,
                gate_cumsum_offset,
            } => {
                layer_bytes.push(1u8); // Layer type: Linear
                layer_bytes.extend_from_slice(&[0u8; 3]); // Padding

                // Has state flag
                let has_state = ssm_state.is_some();
                layer_bytes.push(if has_state { 1u8 } else { 0u8 });
                layer_bytes.extend_from_slice(&[0u8; 7]); // Padding
                layer_bytes.extend_from_slice(&[0u8; 8]); // Placeholder for seq_len (not used)

                if has_state {
                    write_tensor(&mut layer_bytes, ssm_state.as_ref().unwrap())
                        .map_err(paramecia_core::Error::Io)?;

                    if let Some(conv) = conv_state {
                        layer_bytes.push(1u8); // Has conv state
                        write_tensor(&mut layer_bytes, conv).map_err(paramecia_core::Error::Io)?;
                    } else {
                        layer_bytes.push(0u8);
                    }

                    if let Some(gate) = gate_cumsum_offset {
                        layer_bytes.push(1u8); // Has gate offset
                        write_tensor(&mut layer_bytes, gate).map_err(paramecia_core::Error::Io)?;
                    } else {
                        layer_bytes.push(0u8);
                    }
                }
            }
            LayerState::Empty => {
                layer_bytes.push(2u8); // Layer type: Empty
                layer_bytes.extend_from_slice(&[0u8; 3]); // Padding
                layer_bytes.push(0u8); // No state
                layer_bytes.extend_from_slice(&[0u8; 7]); // Padding
                layer_bytes.extend_from_slice(&[0u8; 8]); // Placeholder for seq_len
            }
        }

        // Write section with layer data
        let mut section_bytes = Vec::new();
        write_section_header(
            &mut section_bytes,
            SECTION_LAYER_STATE,
            layer_bytes.len() as u64,
        )
        .map_err(paramecia_core::Error::Io)?;
        section_bytes.extend_from_slice(&layer_bytes);

        writer
            .write_all(&section_bytes)
            .map_err(paramecia_core::Error::Io)?;
        crc.update(&section_bytes);
    }

    // Write footer
    let crc32 = crc.finalize();
    writer
        .write_all(&crc32.to_le_bytes())
        .map_err(paramecia_core::Error::Io)?;
    writer
        .write_all(MAGIC_END)
        .map_err(paramecia_core::Error::Io)?;
    writer
        .write_all(&0u64.to_le_bytes()) // Total size placeholder
        .map_err(paramecia_core::Error::Io)?;
    writer
        .write_all(&[0u8; 8]) // Padding
        .map_err(paramecia_core::Error::Io)?;

    writer.flush().map_err(paramecia_core::Error::Io)?;

    Ok(())
}

/// Load snapshot from file
pub fn load_snapshot<P: AsRef<Path>>(path: P) -> Result<Snapshot> {
    let file = File::open(path.as_ref()).map_err(paramecia_core::Error::Io)?;
    let mut reader = BufReader::new(file);
    let mut crc = Hasher::new();

    // Read and validate header
    let header_start_pos = reader
        .stream_position()
        .map_err(paramecia_core::Error::Io)?;

    let header = SnapshotHeader::read(&mut reader).map_err(paramecia_core::Error::Io)?;

    if header.version != VERSION {
        paramecia_core::bail!("Unsupported snapshot version: {}", header.version);
    }

    // Update CRC with header
    reader
        .seek(SeekFrom::Start(header_start_pos))
        .map_err(paramecia_core::Error::Io)?;
    let mut header_bytes = vec![0u8; 64];
    reader
        .read_exact(&mut header_bytes)
        .map_err(paramecia_core::Error::Io)?;
    crc.update(&header_bytes);

    let mut config = None;
    let mut tokens = Vec::new();
    let mut layer_states = vec![LayerState::Empty; header.num_layers as usize];

    // Read sections - we expect: 1 config + 1 tokens + N layer sections
    let expected_sections = 2 + header.num_layers as usize;
    let mut sections_read = 0;

    while sections_read < expected_sections {
        // Read section header
        let (section_type, section_size) =
            read_section_header(&mut reader).map_err(paramecia_core::Error::Io)?;

        let section_start = reader
            .stream_position()
            .map_err(paramecia_core::Error::Io)?;

        // Update CRC with section header (seek back to read it again)
        reader
            .seek(SeekFrom::Start(section_start - 12))
            .map_err(paramecia_core::Error::Io)?;
        let mut section_header = vec![0u8; 12];
        reader
            .read_exact(&mut section_header)
            .map_err(paramecia_core::Error::Io)?;
        crc.update(&section_header);

        match section_type {
            SECTION_CONFIG => {
                let mut config_bytes = vec![0u8; section_size as usize];
                reader
                    .read_exact(&mut config_bytes)
                    .map_err(paramecia_core::Error::Io)?;
                crc.update(&config_bytes);

                let config_str = String::from_utf8(config_bytes).map_err(|e| {
                    paramecia_core::Error::Msg(format!("Invalid UTF-8 in config: {}", e))
                })?;
                config = Some(serde_json::from_str(&config_str).map_err(|e| {
                    paramecia_core::Error::Msg(format!("Failed to parse config: {}", e))
                })?);
            }
            SECTION_TOKENS => {
                let mut buf = [0u8; 4];
                reader
                    .read_exact(&mut buf)
                    .map_err(paramecia_core::Error::Io)?;
                crc.update(&buf);
                let num_tokens = u32::from_le_bytes(buf) as usize;

                tokens = Vec::with_capacity(num_tokens);
                for _ in 0..num_tokens {
                    reader
                        .read_exact(&mut buf)
                        .map_err(paramecia_core::Error::Io)?;
                    crc.update(&buf);
                    tokens.push(u32::from_le_bytes(buf));
                }
            }
            SECTION_LAYER_STATE => {
                // Read section data
                let mut section_data = vec![0u8; section_size as usize];
                reader
                    .read_exact(&mut section_data)
                    .map_err(paramecia_core::Error::Io)?;
                crc.update(&section_data);

                let mut layer_reader = &section_data[..];

                // Parse layer state
                let mut buf = [0u8; 4];
                layer_reader
                    .read_exact(&mut buf)
                    .map_err(paramecia_core::Error::Io)?;
                let layer_idx = u32::from_le_bytes(buf) as usize;

                let mut buf1 = [0u8; 1];
                layer_reader
                    .read_exact(&mut buf1)
                    .map_err(paramecia_core::Error::Io)?;
                let layer_type = buf1[0];

                // Skip padding
                let mut padding3 = [0u8; 3];
                layer_reader
                    .read_exact(&mut padding3)
                    .map_err(paramecia_core::Error::Io)?;

                layer_reader
                    .read_exact(&mut buf1)
                    .map_err(paramecia_core::Error::Io)?;
                let has_state = buf1[0] != 0;

                // Skip padding
                let mut padding7 = [0u8; 7];
                layer_reader
                    .read_exact(&mut padding7)
                    .map_err(paramecia_core::Error::Io)?;

                let mut buf8 = [0u8; 8];
                layer_reader
                    .read_exact(&mut buf8)
                    .map_err(paramecia_core::Error::Io)?;
                let seq_len = u64::from_le_bytes(buf8) as usize;

                let layer_state = match layer_type {
                    0 => {
                        // Full attention
                        if has_state {
                            let k_cache = read_tensor(&mut layer_reader)
                                .map_err(paramecia_core::Error::Io)?;
                            let v_cache = read_tensor(&mut layer_reader)
                                .map_err(paramecia_core::Error::Io)?;
                            LayerState::FullAttention {
                                k_cache: Some(k_cache),
                                v_cache: Some(v_cache),
                                seq_len,
                            }
                        } else {
                            LayerState::FullAttention {
                                k_cache: None,
                                v_cache: None,
                                seq_len: 0,
                            }
                        }
                    }
                    1 => {
                        // Linear attention
                        if has_state {
                            let ssm_state = read_tensor(&mut layer_reader)
                                .map_err(paramecia_core::Error::Io)?;

                            layer_reader
                                .read_exact(&mut buf1)
                                .map_err(paramecia_core::Error::Io)?;
                            let has_conv = buf1[0] != 0;
                            let conv_state = if has_conv {
                                Some(
                                    read_tensor(&mut layer_reader)
                                        .map_err(paramecia_core::Error::Io)?,
                                )
                            } else {
                                None
                            };

                            layer_reader
                                .read_exact(&mut buf1)
                                .map_err(paramecia_core::Error::Io)?;
                            let has_gate = buf1[0] != 0;
                            let gate_cumsum_offset = if has_gate {
                                Some(
                                    read_tensor(&mut layer_reader)
                                        .map_err(paramecia_core::Error::Io)?,
                                )
                            } else {
                                None
                            };

                            LayerState::LinearAttention {
                                ssm_state: Some(ssm_state),
                                conv_state,
                                gate_cumsum_offset,
                            }
                        } else {
                            LayerState::LinearAttention {
                                ssm_state: None,
                                conv_state: None,
                                gate_cumsum_offset: None,
                            }
                        }
                    }
                    2 => LayerState::Empty,
                    _ => paramecia_core::bail!("Unknown layer type: {}", layer_type),
                };

                layer_states[layer_idx] = layer_state;
            }
            _ => {
                // Skip unknown section
                reader
                    .seek(SeekFrom::Current(section_size as i64))
                    .map_err(paramecia_core::Error::Io)?;
            }
        }

        sections_read += 1;
    }

    // Read and validate footer
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(paramecia_core::Error::Io)?;
    let stored_crc = u32::from_le_bytes(buf);

    let computed_crc = crc.finalize();
    if stored_crc != computed_crc {
        paramecia_core::bail!(
            "CRC32 mismatch: expected 0x{:08x}, got 0x{:08x}",
            stored_crc,
            computed_crc
        );
    }

    let config =
        config.ok_or_else(|| paramecia_core::Error::Msg("Missing config section".to_string()))?;

    Ok(Snapshot {
        state_position: header.state_position as usize,
        tokens,
        layer_states,
        config,
    })
}

/// Validate snapshot compatibility with model config
pub fn validate_config(
    model_config: &SnapshotConfig,
    snapshot_config: &SnapshotConfig,
) -> Result<Vec<String>> {
    let mut warnings = Vec::new();

    // Critical fields must match
    if model_config.num_layers != snapshot_config.num_layers {
        paramecia_core::bail!(
            "Layer count mismatch: model has {}, snapshot has {}",
            model_config.num_layers,
            snapshot_config.num_layers
        );
    }

    if model_config.hidden_size != snapshot_config.hidden_size {
        paramecia_core::bail!(
            "Hidden size mismatch: model has {}, snapshot has {}",
            model_config.hidden_size,
            snapshot_config.hidden_size
        );
    }

    if model_config.num_attention_heads != snapshot_config.num_attention_heads {
        paramecia_core::bail!(
            "Attention heads mismatch: model has {}, snapshot has {}",
            model_config.num_attention_heads,
            snapshot_config.num_attention_heads
        );
    }

    if model_config.head_dim != snapshot_config.head_dim {
        paramecia_core::bail!(
            "Head dimension mismatch: model has {}, snapshot has {}",
            model_config.head_dim,
            snapshot_config.head_dim
        );
    }

    // Non-critical fields generate warnings
    if model_config.max_position_embeddings != snapshot_config.max_position_embeddings {
        warnings.push(format!(
            "Max position embeddings differ: model {}, snapshot {}",
            model_config.max_position_embeddings, snapshot_config.max_position_embeddings
        ));
    }

    if (model_config.rope_freq_base - snapshot_config.rope_freq_base).abs() > 1e-6 {
        warnings.push(format!(
            "RoPE freq base differs: model {}, snapshot {}",
            model_config.rope_freq_base, snapshot_config.rope_freq_base
        ));
    }

    Ok(warnings)
}
