use std::collections::HashMap;
#[cfg(not(target_family = "wasm"))]
use std::time::Instant;

use wgpu::util::DeviceExt;

const N_ROWS: u32 = 256;
const N_STATE: u32 = 16;
const N_INSTANCES_PER_ROW: u32 = 1 << N_LOG_INSTANCES_PER_ROW;
const N_LOG_INSTANCES_PER_ROW: u32 = 3;
const N_COLUMNS: u32 = N_INSTANCES_PER_ROW * N_COLUMNS_PER_REP;
const N_INTERACTION_COLUMNS: u32 = N_INSTANCES_PER_ROW * 4;
const N_HALF_FULL_ROUNDS: u32 = 4;
const FULL_ROUNDS: u32 = 2 * N_HALF_FULL_ROUNDS;
const N_PARTIAL_ROUNDS: u32 = 14;
const N_LANES: u32 = 16;
const N_EXTENDED_ROWS: u32 = N_ROWS * 4;
// const N_ORIGINAL_ROWS: u32 = N_ROWS;

const N_ORIGINAL_COLUMN_SIZE: u32 = N_LANES * N_ROWS;

const N_COLUMNS_PER_REP: u32 = N_STATE * (1 + FULL_ROUNDS) + N_PARTIAL_ROUNDS;
const GEN_TRACE_WORKGROUP_SIZE: u32 = N_ROWS * N_LANES / GEN_TRACE_THREADS_PER_WORKGROUP;
const GEN_TRACE_THREADS_PER_WORKGROUP: u32 = 256;
const INTERPOLATE_WORKGROUP_SIZE: u32 = 8;
#[allow(dead_code)]
const INTERPOLATE_THREADS_PER_WORKGROUP: u32 = 256;
const N_FLAT_MAX_ARRAY_SIZE: u32 = N_ROWS * N_LANES * N_COLUMNS;

pub const N_LINE_TWIDDLES_SIZE: u32 = N_EXTENDED_ROWS * N_LANES;
pub const N_LINE_TWIDDLES_FLAT_SIZE: u32 = N_LINE_TWIDDLES_SIZE * 2;
pub const N_CIRCLE_TWIDDLES_SIZE: u32 = N_LINE_TWIDDLES_SIZE * 2;
const N_ORIGINAL_TRACE_COLUMNS: u32 = 1 + N_COLUMNS + N_INTERACTION_COLUMNS + 3;
const N_CONSTRAINTS: u32 = 1144;

use crate::core::backend::cpu::circle::circle_twiddles_from_line_twiddles;
use crate::core::backend::gpu::qm31::{GpuCM31, GpuM31, GpuQM31};
use crate::core::backend::simd::column::BaseColumn;
#[allow(unused_imports)]
use crate::core::backend::simd::m31::PackedM31;
#[allow(unused_imports)]
use crate::core::backend::Column;
use crate::core::backend::CpuBackend;
use crate::core::fields::m31::BaseField;
#[allow(unused_imports)]
use crate::core::fields::m31::M31;
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly, PolyOps};
use crate::core::poly::utils::domain_line_twiddles_from_tree;
#[allow(unused_imports)]
use crate::examples::poseidon::LookupData;
use crate::examples::poseidon::PoseidonElements;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Twiddles {
    pub circle_twiddles: [GpuM31; N_CIRCLE_TWIDDLES_SIZE as usize],
    pub circle_twiddles_size: u32,
    pub line_twiddles_flat: [GpuM31; N_LINE_TWIDDLES_FLAT_SIZE as usize],
    pub line_twiddles_layer_count: u32,
    pub line_twiddles_sizes: [u32; N_LINE_TWIDDLES_SIZE as usize],
    pub line_twiddles_offsets: [u32; N_LINE_TWIDDLES_SIZE as usize],
    pub mod_inv: GpuM31,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuOriginalColumn {
    pub data: [GpuM31; N_ORIGINAL_COLUMN_SIZE as usize],
}

impl From<PoseidonElements> for GpuLookupElements {
    fn from(value: PoseidonElements) -> Self {
        GpuLookupElements {
            z: value.0.z.into(),
            alpha: value.0.alpha.into(),
            alpha_powers: value
                .0
                .alpha_powers
                .iter()
                .map(|&x| x.into())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuLookupElements {
    pub z: GpuQM31,
    pub alpha: GpuQM31,
    pub alpha_powers: [GpuQM31; N_STATE as usize],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct GpuGenTraceInput {
    pub log_size: u32,
    pub twiddles: Twiddles,
    pub lookup_elements: GpuLookupElements,
    pub denom_inv: [GpuM31; 4],
    pub random_coeff_powers: [GpuQM31; N_CONSTRAINTS as usize],
    pub trace_domain_log_size: u32,
    pub eval_domain_log_size: u32,
}

impl GpuGenTraceInput {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct GpuLookupData {
    initial_state: [[GpuBaseColumn; N_STATE as usize]; N_INSTANCES_PER_ROW as usize],
    final_state: [[GpuBaseColumn; N_STATE as usize]; N_INSTANCES_PER_ROW as usize],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuBaseColumn {
    data: [[GpuM31; N_LANES as usize]; N_ROWS as usize],
    length: u32,
}

impl From<GpuBaseColumn> for BaseColumn {
    fn from(value: GpuBaseColumn) -> Self {
        BaseColumn {
            data: value
                .data
                .iter()
                .map(|f| {
                    let mut array: [M31; N_LANES as usize] = [M31(0); N_LANES as usize];
                    for (i, v) in f.iter().enumerate() {
                        array[i] = M31(v.data);
                    }
                    PackedM31::from_array(array)
                })
                .collect(),
            length: value.length as usize,
        }
    }
}

#[derive(Clone, Debug, Copy)]
#[repr(C)]
pub struct GenTraceOutput {
    original_trace: [GpuOriginalColumn; N_ORIGINAL_TRACE_COLUMNS as usize],
    trace: [GpuBaseColumn; N_COLUMNS as usize],
    lookup_data: GpuLookupData,
}

#[allow(dead_code)]
pub struct InterpolateOutput {
    results: [GpuM31; N_FLAT_MAX_ARRAY_SIZE as usize],
}

#[derive(Clone, Debug)]
#[repr(C)]
struct GenTraceOutputVec {
    original_trace: Vec<CircleEvaluation<CpuBackend, BaseField>>,
    trace: Vec<BaseColumn>,
    lookup_data: LookupData,
}

#[allow(dead_code)]
struct InterpolateOutputVec {
    results: Vec<CirclePoly<CpuBackend>>,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuQM31Column {
    pub data: [GpuQM31; N_ORIGINAL_COLUMN_SIZE as usize],
    pub length: u32,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GenInteractionTraceOutput {
    pub interaction_trace_qm31: [GpuQM31Column; N_INSTANCES_PER_ROW as usize],
    pub interaction_trace_buffers: [GpuQM31Column; 4],
    pub total_sum: GpuQM31,
}

impl InterpolateOutputVec {
    #[allow(dead_code)]
    pub fn from_bytes(bytes: &[u8], log_n_rows: u32) -> Self {
        assert!(bytes.len() >= std::mem::size_of::<[u32; N_FLAT_MAX_ARRAY_SIZE as usize]>());

        let results_slice = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const u32,
                N_COLUMNS as usize * (1 << log_n_rows) as usize,
            )
        };

        let mut polys = Vec::new();
        for i in 0..N_COLUMNS {
            polys.push(CirclePoly::new(
                results_slice[i as usize * (1 << log_n_rows) as usize
                    ..(i as usize + 1) * (1 << log_n_rows) as usize]
                    .iter()
                    .map(|&x| M31(x))
                    .collect(),
            ));
        }

        Self { results: polys }
    }
}
#[allow(dead_code)]
impl GenTraceOutputVec {
    fn from_bytes(bytes: &[u8], log_n_rows: u32) -> Self {
        let size_original_trace =
            std::mem::size_of::<GpuOriginalColumn>() * (N_ORIGINAL_TRACE_COLUMNS as usize);
        let base_column_size = std::mem::size_of::<GpuBaseColumn>();
        let size_trace = base_column_size * (N_COLUMNS as usize);
        let lookup_data_size = std::mem::size_of::<GpuLookupData>();
        let total_size = size_original_trace + size_trace + lookup_data_size;
        assert!(
            bytes.len() >= total_size,
            "Not enough bytes: expected {} but got {}",
            total_size,
            bytes.len()
        );

        let circle_domain = CanonicCoset::new(log_n_rows).circle_domain();

        let original_trace: Vec<CircleEvaluation<CpuBackend, BaseField>> = bytes
            .chunks(std::mem::size_of::<GpuOriginalColumn>())
            .take(N_ORIGINAL_TRACE_COLUMNS as usize)
            .map(|chunk| {
                let gpu_original = unsafe { &*(chunk.as_ptr() as *const GpuOriginalColumn) };
                let values: Vec<M31> = gpu_original.data.iter().map(|x| M31(x.data)).collect();
                CircleEvaluation::new(circle_domain.clone(), values)
            })
            .collect();

        let trace_offset = size_original_trace;
        let trace_bytes = &bytes[trace_offset..trace_offset + size_trace];
        let trace: Vec<BaseColumn> = trace_bytes
            .chunks(base_column_size)
            .take(N_COLUMNS as usize)
            .map(|chunk| BaseColumn::from_bytes(chunk))
            .collect();

        let lookup_data_offset = size_original_trace + size_trace;
        let lookup_data_bytes = &bytes[lookup_data_offset..lookup_data_offset + lookup_data_size];
        let lookup_data = LookupData::from_bytes(lookup_data_bytes);

        Self {
            original_trace,
            trace,
            lookup_data,
        }
    }
}

pub trait ByteSerialize: Sized {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self) as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }

    fn from_bytes(bytes: &[u8]) -> &Self {
        assert!(bytes.len() >= std::mem::size_of::<Self>());
        unsafe { &*(bytes.as_ptr() as *const Self) }
    }
}

impl ByteSerialize for BaseColumn {}
impl ByteSerialize for GenTraceOutput {}
impl ByteSerialize for GpuLookupElements {}
impl ByteSerialize for GpuQM31Column {}
impl ByteSerialize for GenInteractionTraceOutput {}

#[allow(dead_code)]
struct WgpuInstance {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    staging_buffer: wgpu::Buffer,
    interpolate_staging_buffer: wgpu::Buffer,
    gen_interaction_trace_staging_buffer: wgpu::Buffer,
    encoder: wgpu::CommandEncoder,
}

fn create_gpu_input(log_size: u32, lookup_elements: &PoseidonElements) -> GpuGenTraceInput {
    let mut input = GpuGenTraceInput {
        log_size: 0,
        twiddles: Twiddles {
            circle_twiddles: [GpuM31 { data: 0 }; N_CIRCLE_TWIDDLES_SIZE as usize],
            circle_twiddles_size: 0,
            line_twiddles_flat: [GpuM31 { data: 0 }; N_LINE_TWIDDLES_FLAT_SIZE as usize],
            line_twiddles_layer_count: 0,
            line_twiddles_sizes: [0; N_LINE_TWIDDLES_SIZE as usize],
            line_twiddles_offsets: [0; N_LINE_TWIDDLES_SIZE as usize],
            mod_inv: GpuM31 { data: 0 },
        },
        lookup_elements: GpuLookupElements {
            z: lookup_elements.0.z.into(),
            alpha: lookup_elements.0.alpha.into(),
            alpha_powers: lookup_elements.0.alpha_powers.map(|p| p.into()),
        },
        denom_inv: [GpuM31 { data: 0 }; 4],
        random_coeff_powers: [GpuQM31 {
            a: GpuCM31 {
                a: GpuM31 { data: 0 },
                b: GpuM31 { data: 0 },
            },
            b: GpuCM31 {
                a: GpuM31 { data: 0 },
                b: GpuM31 { data: 0 },
            },
        }; N_CONSTRAINTS as usize],
        trace_domain_log_size: 0,
        eval_domain_log_size: 0,
    };
    input.log_size = log_size;

    let domain = CanonicCoset::new(log_size + 3).circle_domain();
    let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);

    // line twiddles
    let domain = CanonicCoset::new(log_size).circle_domain();
    let line_twiddles = domain_line_twiddles_from_tree(domain, &twiddles.itwiddles);
    input.twiddles.line_twiddles_layer_count = line_twiddles.len() as u32;
    for (i, twiddle) in line_twiddles.iter().enumerate() {
        input.twiddles.line_twiddles_sizes[i] = twiddle.len() as u32;
        input.twiddles.line_twiddles_offsets[i] = if i == 0 {
            0
        } else {
            input.twiddles.line_twiddles_offsets[i - 1] + input.twiddles.line_twiddles_sizes[i - 1]
        };
        for (j, twiddle) in twiddle.iter().enumerate() {
            input.twiddles.line_twiddles_flat
                [input.twiddles.line_twiddles_offsets[i] as usize + j] = GpuM31 {
                data: (*twiddle).into(),
            };
        }
    }

    // circle twiddles
    let circle_twiddles: Vec<GpuM31> = circle_twiddles_from_line_twiddles(line_twiddles[0])
        .map(|x| GpuM31 { data: x.into() })
        .collect();
    input.twiddles.circle_twiddles[..circle_twiddles.len()].copy_from_slice(&circle_twiddles);
    input.twiddles.circle_twiddles_size = circle_twiddles.len() as u32;

    let inv = BaseField::from_u32_unchecked(domain.size() as u32).inverse();
    input.twiddles.mod_inv = GpuM31 { data: inv.into() };

    input
}

async fn init(log_n_rows: u32, lookup_elements: &PoseidonElements) -> WgpuInstance {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    let adapter_limits = adapter.limits();
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::SHADER_INT64,
                required_limits: adapter_limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .unwrap();

    let input_data = create_gpu_input(log_n_rows, lookup_elements);

    // Create buffers
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: input_data.as_bytes(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    println!(
        "std::mem::size_of::<GenTraceOutput>: {}",
        std::mem::size_of::<GenTraceOutput>()
    );
    let buffer_size = std::mem::size_of::<GenTraceOutput>();
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: buffer_size as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let interpolate_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Interpolate Output Buffer"),
        size: std::mem::size_of::<InterpolateOutput>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let gen_interaction_trace_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Gen Interaction Trace Output Buffer"),
        size: std::mem::size_of::<GenInteractionTraceOutput>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Load shader
    let constant_shader = include_str!("integrated_module_constants.wgsl");
    let utils_shader = include_str!("../utils.wgsl");
    let qm31_shader = include_str!("../qm31.wgsl");
    let gen_trace_impl_shader = include_str!("gen_trace.wgsl");
    let gen_trace_shader = format!(
        "{}\n
        {}\n    
        {}\n
        {}",
        constant_shader, utils_shader, qm31_shader, gen_trace_impl_shader,
    );

    let interpolate_impl_shader = include_str!("interpolate_trace.wgsl");
    let interpolate_shader = format!(
        "{}\n
        {}\n
        {}",
        constant_shader, qm31_shader, interpolate_impl_shader,
    );
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Gen Trace Shader"),
        source: wgpu::ShaderSource::Wgsl(gen_trace_shader.into()),
    });

    // Load interpolate shader
    let interpolate_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Interpolate Shader"),
        source: wgpu::ShaderSource::Wgsl(interpolate_shader.into()),
    });

    // Get the maximum buffer size supported by the device
    let max_buffer_size = device.limits().max_buffer_size;
    println!("Maximum buffer size supported: {} bytes", max_buffer_size);
    #[cfg(target_family = "wasm")]
    web_sys::console::log_1(
        &format!("Maximum buffer size supported: {} bytes", max_buffer_size).into(),
    );

    // Check if our buffer size exceeds the limit
    if max_buffer_size > usize::MAX as u64 {
        // If max_buffer_size is larger than what usize can represent on this platform
        if buffer_size == usize::MAX {
            // This is a special case where buffer_size has reached the maximum possible value
            panic!("Buffer size has reached the maximum value representable by usize");
        }
        // We know buffer_size is less than max_buffer_size since max_buffer_size > usize::MAX
        // and buffer_size <= usize::MAX
    } else {
        // Safe to convert max_buffer_size to usize since it's within range
        if buffer_size > max_buffer_size as usize {
            panic!(
                "Buffer size {} exceeds maximum allowed size {}",
                buffer_size, max_buffer_size
            );
        }
    }

    // Bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            // Binding 0: Input buffer
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 1: Gen Trace Output buffer
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 2: Interpolate Output buffer
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 3: Gen Interaction Trace Output buffer
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
        label: Some("Gen Trace Bind Group Layout"),
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: interpolate_output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: gen_interaction_trace_output_buffer.as_entire_binding(),
            },
        ],
        label: Some("Gen Trace Bind Group"),
    });

    // Pipeline layout
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
        label: Some("Gen Trace Pipeline Layout"),
    });

    // Compute pipeline
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Gen Trace Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("gen_trace_interpolate_columns"),
        cache: None,
        compilation_options: Default::default(),
    });

    // compute interaction trace
    let compute_interaction_trace_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Interaction Trace Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("compute_interaction_trace"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &HashMap::from([]),
                zero_initialize_workgroup_memory: true,
            },
        });

    // load interaction trace to original column
    let load_interaction_trace_to_original_column_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Load Interaction Trace to Original Column Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("interaction_trace_to_original_column"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &HashMap::from([]),
                zero_initialize_workgroup_memory: true,
            },
        });

    let interpolate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &interpolate_shader_module,
        entry_point: Some("interpolate"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create encoder
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Gen Trace Command Encoder"),
    });

    // Dispatch the compute shader
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Gen Trace Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(GEN_TRACE_WORKGROUP_SIZE, 1, 1);

        compute_pass.set_pipeline(&compute_interaction_trace_pipeline);
        compute_pass.dispatch_workgroups(1, 1, 1);

        compute_pass.set_pipeline(&load_interaction_trace_to_original_column_pipeline);
        compute_pass.dispatch_workgroups(1, 1, 1);

        compute_pass.set_pipeline(&interpolate_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(1, INTERPOLATE_WORKGROUP_SIZE, 1);
    }

    // Copy output to staging buffer for read access
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: buffer_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let interpolate_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Interpolate Staging Buffer"),
        size: std::mem::size_of::<InterpolateOutput>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let gen_interaction_trace_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Gen Interaction Trace Staging Buffer"),
        size: std::mem::size_of::<GenInteractionTraceOutput>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, staging_buffer.size());
    encoder.copy_buffer_to_buffer(
        &interpolate_output_buffer,
        0,
        &interpolate_staging_buffer,
        0,
        interpolate_staging_buffer.size(),
    );
    encoder.copy_buffer_to_buffer(
        &gen_interaction_trace_output_buffer,
        0,
        &gen_interaction_trace_staging_buffer,
        0,
        gen_interaction_trace_staging_buffer.size(),
    );

    WgpuInstance {
        instance,
        adapter,
        device,
        queue,
        staging_buffer,
        interpolate_staging_buffer,
        gen_interaction_trace_staging_buffer,
        encoder,
    }
}

pub async fn compute_integrated_module(
    log_n_rows: u32,
    lookup_elements: &PoseidonElements,
) -> (
    Vec<BaseColumn>,
    LookupData,
    Vec<CirclePoly<CpuBackend>>,
    Vec<CircleEvaluation<CpuBackend, BaseField>>,
) {
    let instance = init(log_n_rows, lookup_elements).await;

    #[cfg(not(target_family = "wasm"))]
    let gpu_start = Instant::now();
    #[cfg(target_family = "wasm")]
    let gpu_start = web_sys::window().unwrap().performance().unwrap().now();

    // Submit the commands
    instance.queue.submit(Some(instance.encoder.finish()));

    let staging_buffer_slice = instance.staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    staging_buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    instance
        .device
        .poll(wgpu::Maintain::wait())
        .panic_on_timeout();
    let result = async {
        receiver.recv_async().await.unwrap().unwrap();
        let data = staging_buffer_slice.get_mapped_range();
        let output = GenTraceOutputVec::from_bytes(&data, log_n_rows);
        drop(data);
        instance.staging_buffer.unmap();
        (output.trace, output.lookup_data, output.original_trace)
    };

    let interpolate_output_slice = instance.interpolate_staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    interpolate_output_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    instance
        .device
        .poll(wgpu::Maintain::wait())
        .panic_on_timeout();
    let interpolate_result = async {
        receiver.recv_async().await.unwrap().unwrap();
        let data = interpolate_output_slice.get_mapped_range();
        let output = InterpolateOutputVec::from_bytes(&data, log_n_rows);
        drop(data);
        instance.interpolate_staging_buffer.unmap();
        output
    };

    let (trace, lookup_data, original_trace) = result.await;
    let _interpolate_output = interpolate_result.await;

    #[cfg(not(target_family = "wasm"))]
    println!(
        "Gen Trace Interpolate Columns GPU time: {:?}",
        gpu_start.elapsed()
    );

    #[cfg(target_family = "wasm")]
    let gpu_end = web_sys::window().unwrap().performance().unwrap().now();
    #[cfg(target_family = "wasm")]
    web_sys::console::log_1(
        &format!(
            "Gen Trace Interpolate Columns GPU time: {:?}ms",
            gpu_end - gpu_start
        )
        .into(),
    );

    // let lookup_data = LookupData {
    //     initial_state: std::array::from_fn(|_| std::array::from_fn(|_| BaseColumn::zeros(1))),
    //     final_state: std::array::from_fn(|_| std::array::from_fn(|_| BaseColumn::zeros(1))),
    // };
    // (Vec::new(), lookup_data, _interpolate_output.results)
    (
        trace,
        lookup_data,
        _interpolate_output.results,
        original_trace,
    )
}
