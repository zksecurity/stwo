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
const N_ORIGINAL_ROWS: u32 = N_ROWS;
const N_COLUMNS_PER_REP: u32 = N_STATE * (1 + FULL_ROUNDS) + N_PARTIAL_ROUNDS;
const GEN_TRACE_WORKGROUP_SIZE: u32 = N_ROWS * N_LANES / GEN_TRACE_THREADS_PER_WORKGROUP;
const GEN_TRACE_THREADS_PER_WORKGROUP: u32 = 256;
const INTERPOLATE_WORKGROUP_SIZE: u32 = 8;
#[allow(dead_code)]
const INTERPOLATE_THREADS_PER_WORKGROUP: u32 = 256;
const MAX_ARRAY_LOG_SIZE: u32 = 25;
const MAX_ARRAY_SIZE: usize = 1 << MAX_ARRAY_LOG_SIZE;

pub const N_LINE_TWIDDLES_SIZE: u32 = N_EXTENDED_ROWS * N_LANES;
pub const N_LINE_TWIDDLES_FLAT_SIZE: u32 = N_LINE_TWIDDLES_SIZE * 2;
pub const N_CIRCLE_TWIDDLES_SIZE: u32 = N_LINE_TWIDDLES_SIZE * 2;
pub const N_ORIGINAL_TRACE_COLUMNS: u32 = 1 + N_COLUMNS + N_INTERACTION_COLUMNS;

use crate::core::backend::cpu::circle::circle_twiddles_from_line_twiddles;
use crate::core::backend::gpu::qm31::GpuM31;
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
use crate::core::poly::circle::{CanonicCoset, CirclePoly, PolyOps};
use crate::core::poly::utils::domain_line_twiddles_from_tree;
#[allow(unused_imports)]
use crate::examples::poseidon::LookupData;

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
struct GpuGenTraceInput {
    pub log_size: u32,
    pub twiddles: Twiddles,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuOriginalColumn {
    pub coeffs: [GpuM31; (N_LANES * N_ORIGINAL_ROWS) as usize],
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
    original_traces: [GpuOriginalColumn; N_ORIGINAL_TRACE_COLUMNS as usize],
    trace: [GpuBaseColumn; N_COLUMNS as usize],
    lookup_data: GpuLookupData,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Copy)]
#[repr(C)]
pub struct InterpolateOutput {
    results: [u32; MAX_ARRAY_SIZE],
}

#[allow(dead_code)]
impl BaseColumn {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= std::mem::size_of::<Self>());
        let slice = unsafe { &*(bytes.as_ptr() as *const GpuBaseColumn) };
        (*slice).into()
    }
}

#[allow(dead_code)]
impl LookupData {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let base_column_size = std::mem::size_of::<GpuBaseColumn>();
        let base_column_vec_size = base_column_size * N_STATE as usize;
        let state_size = base_column_vec_size * N_INSTANCES_PER_ROW as usize;
        let lookup_data_size = state_size * 2;
        assert!(bytes.len() >= lookup_data_size);
        let initial_state_slice: [[BaseColumn; N_STATE as usize]; N_INSTANCES_PER_ROW as usize] =
            bytes
                .chunks(base_column_vec_size)
                .take(N_INSTANCES_PER_ROW as usize)
                .map(|chunk| {
                    chunk
                        .chunks(base_column_size)
                        .take(N_STATE as usize)
                        .map(|chunk| BaseColumn::from_bytes(chunk))
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
        let final_state_slice: [[BaseColumn; N_STATE as usize]; N_INSTANCES_PER_ROW as usize] =
            bytes[state_size..]
                .chunks(base_column_vec_size)
                .take(N_INSTANCES_PER_ROW as usize)
                .map(|chunk| {
                    chunk
                        .chunks(base_column_size)
                        .take(N_STATE as usize)
                        .map(|chunk| BaseColumn::from_bytes(chunk))
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
        Self {
            initial_state: initial_state_slice,
            final_state: final_state_slice,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Ids {
    workgroup_id_x: u32,
    workgroup_id_y: u32,
    workgroup_id_z: u32,
    local_invocation_id_x: u32,
    local_invocation_id_y: u32,
    local_invocation_id_z: u32,
    global_invocation_id_x: u32,
    global_invocation_id_y: u32,
    global_invocation_id_z: u32,
    local_invocation_index: u32,
    num_workgroups_x: u32,
    num_workgroups_y: u32,
    num_workgroups_z: u32,
    workgroup_index: u32,
    global_invocation_index: u32,
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
impl ByteSerialize for GpuGenTraceInput {}
impl ByteSerialize for GpuLookupData {}

impl InterpolateOutput {
    fn from_bytes(bytes: &[u8]) -> Self {
        unsafe { *(bytes.as_ptr() as *const Self) }
    }
}

#[allow(dead_code)]
struct WgpuInstance {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    staging_buffer: wgpu::Buffer,
    interpolate_staging_buffer: wgpu::Buffer,
    encoder: wgpu::CommandEncoder,
}

fn create_gpu_input(log_size: u32) -> GpuGenTraceInput {
    let domain = CanonicCoset::new(log_size + 3).circle_domain();
    let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);
    let line_twiddles = domain_line_twiddles_from_tree(domain, &twiddles.twiddles);
    let mut twiddle_input = Twiddles {
        line_twiddles_layer_count: line_twiddles.len() as u32,
        line_twiddles_sizes: [0; N_LINE_TWIDDLES_SIZE as usize],
        line_twiddles_offsets: [0; N_LINE_TWIDDLES_SIZE as usize],
        line_twiddles_flat: [GpuM31 { data: 0 }; N_LINE_TWIDDLES_FLAT_SIZE as usize],
        circle_twiddles: [GpuM31 { data: 0 }; N_CIRCLE_TWIDDLES_SIZE as usize],
        circle_twiddles_size: 0,
        mod_inv: GpuM31 { data: 0 },
    };
    for (i, twiddle) in line_twiddles.iter().enumerate() {
        twiddle_input.line_twiddles_sizes[i] = twiddle.len() as u32;
        twiddle_input.line_twiddles_offsets[i] = if i == 0 {
            0
        } else {
            twiddle_input.line_twiddles_offsets[i - 1] + twiddle_input.line_twiddles_sizes[i - 1]
        };
        for (j, &twiddle) in twiddle.iter().enumerate() {
            twiddle_input.line_twiddles_flat[twiddle_input.line_twiddles_offsets[i] as usize + j] =
                twiddle.into();
        }
    }
    let circle_twiddles: Vec<_> = circle_twiddles_from_line_twiddles(line_twiddles[0])
        .map(|x| GpuM31::from(x))
        .collect();
    twiddle_input.circle_twiddles[..circle_twiddles.len()].copy_from_slice(&circle_twiddles);
    twiddle_input.circle_twiddles_size = circle_twiddles.len() as u32;

    let inv = BaseField::from_u32_unchecked(domain.size() as u32).inverse();
    twiddle_input.mod_inv = inv.into();

    GpuGenTraceInput {
        log_size,
        twiddles: twiddle_input,
    }
}

async fn init(log_n_rows: u32) -> WgpuInstance {
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

    let input_data = create_gpu_input(log_n_rows);

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

    // Load shader
    let gen_trace_interpolate_columns_constants_shader =
        include_str!("gen_trace_interpolate_columns_constants.wgsl");
    let gen_trace_impl_shader = include_str!("gen_trace.wgsl");
    let interpolate_impl_shader = include_str!("interpolate.wgsl");
    let qm31_impl_shader = include_str!("qm31.wgsl");
    let gen_trace_shader = format!(
        "{}\n
        {}",
        gen_trace_interpolate_columns_constants_shader, gen_trace_impl_shader,
    );
    let interpolate_shader = format!(
        "{}\n
        {}\n
        {}",
        gen_trace_interpolate_columns_constants_shader, qm31_impl_shader, interpolate_impl_shader,
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
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, staging_buffer.size());
    encoder.copy_buffer_to_buffer(
        &interpolate_output_buffer,
        0,
        &interpolate_staging_buffer,
        0,
        interpolate_staging_buffer.size(),
    );

    WgpuInstance {
        instance,
        adapter,
        device,
        queue,
        staging_buffer,
        interpolate_staging_buffer,
        encoder,
    }
}

pub async fn gen_trace_interpolate_columns(
    log_n_rows: u32,
) -> (Vec<BaseColumn>, LookupData, Vec<CirclePoly<CpuBackend>>) {
    let instance = init(log_n_rows).await;

    #[cfg(not(target_family = "wasm"))]
    let gpu_start = Instant::now();
    #[cfg(target_family = "wasm")]
    let gpu_start = web_sys::window().unwrap().performance().unwrap().now();

    // Submit the commands
    instance.queue.submit(Some(instance.encoder.finish()));

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
        let output = InterpolateOutput::from_bytes(&data);
        drop(data);
        instance.interpolate_staging_buffer.unmap();
        output
    };

    // let (trace, lookup_data) = result.await;
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

    let lookup_data = LookupData {
        initial_state: std::array::from_fn(|_| std::array::from_fn(|_| BaseColumn::zeros(1))),
        final_state: std::array::from_fn(|_| std::array::from_fn(|_| BaseColumn::zeros(1))),
    };

    let mut polys = Vec::new();
    for i in 0..N_COLUMNS {
        polys.push(CirclePoly::new(
            _interpolate_output.results[i as usize * (1 << log_n_rows) as usize
                ..(i as usize + 1) * (1 << log_n_rows) as usize]
                .iter()
                .map(|&x| M31(x))
                .collect(),
        ));
    }

    (Vec::new(), lookup_data, polys)
}
