#[cfg(not(target_family = "wasm"))]
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const N_ROWS: u32 = 32;
const N_STATE: u32 = 16;
const N_INSTANCES_PER_ROW: u32 = 1 << N_LOG_INSTANCES_PER_ROW;
const N_LOG_INSTANCES_PER_ROW: u32 = 3;
const N_COLUMNS: u32 = N_INSTANCES_PER_ROW * N_COLUMNS_PER_REP;
const N_HALF_FULL_ROUNDS: u32 = 4;
const FULL_ROUNDS: u32 = 2 * N_HALF_FULL_ROUNDS;
const N_PARTIAL_ROUNDS: u32 = 14;
const N_LANES: u32 = 16;
const N_COLUMNS_PER_REP: u32 = N_STATE * (1 + FULL_ROUNDS) + N_PARTIAL_ROUNDS;
// const LOG_N_LANES: u32 = 4;
const WORKGROUP_SIZE: u32 = 8;
#[allow(dead_code)]
const THREADS_PER_WORKGROUP: u32 = 256;
const MAX_ARRAY_LOG_SIZE: u32 = 20;
const MAX_ARRAY_SIZE: usize = 1 << MAX_ARRAY_LOG_SIZE;

use crate::core::backend::cpu::circle::circle_twiddles_from_line_twiddles;
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

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Complex {
    real: f32,
    imag: f32,
}

#[derive(Debug, Clone)]
#[repr(C)]
struct GenTraceInput<F> {
    pub initial_x: F,
    pub initial_y: F,
    pub log_size: u32,
    pub circle_twiddles: Vec<F>,
    pub circle_twiddles_size: u32,
    pub line_twiddles_flat: Vec<F>,
    pub line_twiddles_layer_count: u32,
    pub line_twiddles_sizes: Vec<u32>,
    pub line_twiddles_offsets: Vec<u32>,
    pub mod_inv: u32,
    pub current_layer: u32,
}

impl<F> GenTraceInput<F>
where
    F: Into<u32> + From<u32> + Copy,
{
    fn as_bytes(&self) -> &[u8] {
        let total_size = std::mem::size_of::<GenTraceInput<F>>();
        let mut bytes = Vec::with_capacity(total_size);

        // initial_x, initial_y
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.initial_x as *const F as *const u8,
                std::mem::size_of::<F>(),
            )
        });
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.initial_y as *const F as *const u8,
                std::mem::size_of::<F>(),
            )
        });

        // log_size
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.log_size as *const u32 as *const u8,
                std::mem::size_of::<u32>(),
            )
        });

        let mut padded_circle_twiddles = vec![F::from(0u32); MAX_ARRAY_SIZE];
        padded_circle_twiddles[..self.circle_twiddles.len()].copy_from_slice(&self.circle_twiddles);
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                padded_circle_twiddles.as_ptr() as *const u8,
                MAX_ARRAY_SIZE * std::mem::size_of::<F>(),
            )
        });

        // circle_twiddles_size
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.circle_twiddles_size as *const u32 as *const u8,
                std::mem::size_of::<u32>(),
            )
        });

        let mut padded_line_twiddles = vec![F::from(0u32); MAX_ARRAY_SIZE];
        padded_line_twiddles[..self.line_twiddles_flat.len()]
            .copy_from_slice(&self.line_twiddles_flat);
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                padded_line_twiddles.as_ptr() as *const u8,
                MAX_ARRAY_SIZE * std::mem::size_of::<F>(),
            )
        });

        // line_twiddles_layer_count
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.line_twiddles_layer_count as *const u32 as *const u8,
                std::mem::size_of::<u32>(),
            )
        });

        let mut padded_sizes = vec![0u32; MAX_ARRAY_SIZE];
        padded_sizes[..self.line_twiddles_sizes.len()].copy_from_slice(&self.line_twiddles_sizes);
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                padded_sizes.as_ptr() as *const u8,
                MAX_ARRAY_SIZE * std::mem::size_of::<u32>(),
            )
        });

        let mut padded_offsets = vec![0u32; MAX_ARRAY_SIZE];
        padded_offsets[..self.line_twiddles_offsets.len()]
            .copy_from_slice(&self.line_twiddles_offsets);
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                padded_offsets.as_ptr() as *const u8,
                MAX_ARRAY_SIZE * std::mem::size_of::<u32>(),
            )
        });

        // mod_inv
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.mod_inv as *const u32 as *const u8,
                std::mem::size_of::<u32>(),
            )
        });

        // current_layer
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.current_layer as *const u32 as *const u8,
                std::mem::size_of::<u32>(),
            )
        });

        Box::leak(bytes.into_boxed_slice())
    }

    fn zero() -> Self {
        Self {
            initial_x: F::from(0),
            initial_y: F::from(0),
            log_size: 0,
            circle_twiddles: vec![F::from(0); MAX_ARRAY_SIZE],
            circle_twiddles_size: 0,
            line_twiddles_flat: vec![F::from(0); MAX_ARRAY_SIZE],
            line_twiddles_layer_count: 0,
            line_twiddles_sizes: vec![0; MAX_ARRAY_SIZE],
            line_twiddles_offsets: vec![0; MAX_ARRAY_SIZE],
            mod_inv: 0,
            current_layer: 0,
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
struct GpuM31 {
    data: u32,
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
    trace: [GpuBaseColumn; N_COLUMNS as usize],
    lookup_data: GpuLookupData,
}

#[allow(dead_code)]
pub struct InterpolateOutput {
    results: [u32; MAX_ARRAY_SIZE],
}

#[derive(Clone, Debug)]
#[repr(C)]
struct GenTraceOutputVec {
    trace: Vec<BaseColumn>,
    lookup_data: LookupData,
}

#[allow(dead_code)]
struct InterpolateOutputVec {
    results: Vec<CirclePoly<CpuBackend>>,
}

impl InterpolateOutputVec {
    #[allow(dead_code)]
    pub fn from_bytes(bytes: &[u8], log_n_rows: u32) -> Self {
        assert!(bytes.len() >= std::mem::size_of::<[u32; MAX_ARRAY_SIZE]>());

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
    fn from_bytes(bytes: &[u8]) -> Self {
        let base_column_size = std::mem::size_of::<GpuBaseColumn>();
        let lookup_data_size = std::mem::size_of::<GpuLookupData>();
        assert!(bytes.len() >= base_column_size * N_COLUMNS as usize + lookup_data_size);
        let base_column_slice = bytes
            .chunks(base_column_size)
            .take(N_COLUMNS as usize)
            .map(|chunk| BaseColumn::from_bytes(chunk))
            .collect::<Vec<_>>();
        let lookup_data_start = base_column_size * N_COLUMNS as usize;
        let lookup_data =
            LookupData::from_bytes(&bytes[lookup_data_start..lookup_data_start + lookup_data_size]);
        Self {
            trace: base_column_slice,
            lookup_data,
        }
    }
}

#[allow(dead_code)]
impl BaseColumn {
    fn from_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= std::mem::size_of::<Self>());
        let slice = unsafe { &*(bytes.as_ptr() as *const GpuBaseColumn) };
        (*slice).into()
    }
}

#[allow(dead_code)]
impl LookupData {
    fn from_bytes(bytes: &[u8]) -> Self {
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

fn create_gpu_input(log_size: u32) -> GenTraceInput<BaseField> {
    let mut input = GenTraceInput::zero();
    input.log_size = log_size;

    let domain = CanonicCoset::new(log_size + 3).circle_domain();
    let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);

    // line twiddles
    let domain = CanonicCoset::new(log_size).circle_domain();
    let line_twiddles = domain_line_twiddles_from_tree(domain, &twiddles.itwiddles);
    input.line_twiddles_layer_count = line_twiddles.len() as u32;
    for (i, twiddle) in line_twiddles.iter().enumerate() {
        input.line_twiddles_sizes[i] = twiddle.len() as u32;
        input.line_twiddles_offsets[i] = if i == 0 {
            0
        } else {
            input.line_twiddles_offsets[i - 1] + input.line_twiddles_sizes[i - 1]
        };
        for (j, twiddle) in twiddle.iter().enumerate() {
            input.line_twiddles_flat[input.line_twiddles_offsets[i] as usize + j] = *twiddle;
        }
    }

    // circle twiddles
    let circle_twiddles: Vec<_> = circle_twiddles_from_line_twiddles(line_twiddles[0]).collect();
    input.circle_twiddles[..circle_twiddles.len()].copy_from_slice(&circle_twiddles);
    input.circle_twiddles_size = circle_twiddles.len() as u32;

    let inv = BaseField::from_u32_unchecked(domain.size() as u32).inverse();
    input.mod_inv = inv.into();

    input
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

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::SHADER_INT64,
                required_limits: wgpu::Limits::default(),
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
    let shader_source = include_str!("gen_trace_interpolate_columns.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Gen Trace Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Load interpolate shader
    let interpolate_shader_source = include_str!("interpolate.wgsl");
    let interpolate_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Interpolate Shader"),
        source: wgpu::ShaderSource::Wgsl(interpolate_shader_source.into()),
    });

    // Get the maximum buffer size supported by the device
    let max_buffer_size = device.limits().max_buffer_size;
    println!("Maximum buffer size supported: {} bytes", max_buffer_size);

    // Check if our buffer size exceeds the limit
    if buffer_size > max_buffer_size as usize {
        panic!(
            "Required buffer size {} exceeds device maximum of {}",
            buffer_size, max_buffer_size
        );
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
        compute_pass.dispatch_workgroups(WORKGROUP_SIZE, 1, 1);

        compute_pass.set_pipeline(&interpolate_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let first_workgroup_size = 32;
        compute_pass.dispatch_workgroups(1, first_workgroup_size, 1);
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

    // // Wait for the GPU to finish and map the staging buffer
    // let buffer_slice = instance.staging_buffer.slice(..);
    // let (sender, receiver) = flume::bounded(1);
    // buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    // instance
    //     .device
    //     .poll(wgpu::Maintain::wait())
    //     .panic_on_timeout();
    // let result = async {
    //     receiver.recv_async().await.unwrap().unwrap();
    //     let data = buffer_slice.get_mapped_range();

    //     let output = GenTraceOutputVec::from_bytes(&data);
    //     drop(data);
    //     instance.staging_buffer.unmap();

    //     let output_trace: Vec<BaseColumn> =
    //         output.trace.clone().into_iter().map(|c| c.into()).collect();
    //     (output_trace, output.lookup_data.into())
    // };

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
    (Vec::new(), lookup_data, _interpolate_output.results)
    // (trace, lookup_data, _interpolate_output.results)
}
