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
const THREADS_PER_WORKGROUP: u32 = 256;

use crate::core::backend::simd::column::BaseColumn;
#[allow(unused_imports)]
use crate::core::backend::simd::m31::PackedM31;
#[allow(unused_imports)]
use crate::core::backend::Column;
#[allow(unused_imports)]
use crate::core::fields::m31::M31;
#[allow(unused_imports)]
use crate::examples::poseidon::LookupData;

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Complex {
    real: f32,
    imag: f32,
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct GenTraceInput {
    log_n_rows: u32,
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
struct GpuStateData {
    data: [[GpuM31; N_LANES as usize]; N_STATE as usize],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct GpuBaseColumn {
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
struct GenTraceOutput {
    trace: [GpuBaseColumn; N_COLUMNS as usize],
    lookup_data: GpuLookupData,
}

#[derive(Clone, Debug)]
#[repr(C)]
struct GenTraceOutputVec {
    trace: Vec<BaseColumn>,
    lookup_data: LookupData,
}

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

impl BaseColumn {
    fn from_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= std::mem::size_of::<Self>());
        let slice = unsafe { &*(bytes.as_ptr() as *const GpuBaseColumn) };
        (*slice).into()
    }
}

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
struct ShaderResult {
    values: [Ids; THREADS_PER_WORKGROUP as usize * WORKGROUP_SIZE as usize],
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

impl ByteSerialize for GenTraceInput {}
impl ByteSerialize for BaseColumn {}
impl ByteSerialize for GpuStateData {}
impl ByteSerialize for GenTraceOutput {}
impl ByteSerialize for ShaderResult {}

#[allow(dead_code)]
struct WgpuInstance {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    staging_buffer: wgpu::Buffer,
    state_staging_buffer: wgpu::Buffer,
    shader_result_staging_buffer: wgpu::Buffer,
    encoder: wgpu::CommandEncoder,
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

    let input_data: GenTraceInput = GenTraceInput { log_n_rows };

    // Create buffers
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(&[input_data]),
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

    let state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("State Buffer"),
        size: (std::mem::size_of::<GpuStateData>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader_result = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shader Result Buffer"),
        size: (std::mem::size_of::<ShaderResult>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Load shader
    let shader_source = include_str!("gen_trace_interpolate_columns.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Gen Trace Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
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
            // Binding 1: Output buffer
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
            // Binding 2: Debug buffer
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
            // Binding 3: Workgroup result buffer
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
                resource: state_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: shader_result.as_entire_binding(),
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

        // Workgroup size defined in shader
        compute_pass.dispatch_workgroups(WORKGROUP_SIZE, 1, 1);
    }

    // Copy output to staging buffer for read access
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: buffer_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, staging_buffer.size());

    // create storage buffer for debug data
    let state_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("State Staging Buffer"),
        size: (std::mem::size_of::<GpuStateData>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // create storage buffer for workgroup result
    let shader_result_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shader Result Staging Buffer"),
        size: (std::mem::size_of::<ShaderResult>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(
        &state_buffer,
        0,
        &state_staging_buffer,
        0,
        state_staging_buffer.size(),
    );

    encoder.copy_buffer_to_buffer(
        &shader_result,
        0,
        &shader_result_staging_buffer,
        0,
        shader_result_staging_buffer.size(),
    );

    WgpuInstance {
        instance,
        adapter,
        device,
        queue,
        staging_buffer,
        state_staging_buffer,
        shader_result_staging_buffer,
        encoder,
    }
}

pub async fn gen_trace_interpolate_columns(log_n_rows: u32) -> (Vec<BaseColumn>, LookupData) {
    let instance = init(log_n_rows).await;

    #[cfg(not(target_family = "wasm"))]
    let gpu_start = Instant::now();
    #[cfg(target_family = "wasm")]
    let gpu_start = web_sys::window().unwrap().performance().unwrap().now();

    // Submit the commands
    instance.queue.submit(Some(instance.encoder.finish()));

    // let buffer_slice = state_staging_buffer.slice(..);
    // let (sender, receiver) = flume::bounded(1);
    // buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    // device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    // if let Ok(Ok(())) = receiver.recv_async().await {
    //     let data = buffer_slice.get_mapped_range();
    //     let _result = *GpuStateData::from_bytes(&data);
    //     drop(data);
    //     state_staging_buffer.unmap();

    //     println!("State data: {:?}", _result);
    // }

    let buffer_slice = instance.shader_result_staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    instance
        .device
        .poll(wgpu::Maintain::wait())
        .panic_on_timeout();

    if let Ok(Ok(())) = receiver.recv_async().await {
        let data = buffer_slice.get_mapped_range();
        let _result = *ShaderResult::from_bytes(&data);
        drop(data);
        instance.shader_result_staging_buffer.unmap();

        // _result.values.iter().enumerate().for_each(|(i, v)| {
        //     println!("Shader result[{}]: {:?}", i, v);
        // });
    }

    // Wait for the GPU to finish and map the staging buffer
    let buffer_slice = instance.staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    instance
        .device
        .poll(wgpu::Maintain::wait())
        .panic_on_timeout();
    let result = async {
        receiver.recv_async().await.unwrap().unwrap();
        let data = buffer_slice.get_mapped_range();

        let output = GenTraceOutputVec::from_bytes(&data);
        drop(data);
        instance.staging_buffer.unmap();

        let output_trace: Vec<BaseColumn> =
            output.trace.clone().into_iter().map(|c| c.into()).collect();
        (output_trace, output.lookup_data.into())
    };

    let (trace, lookup_data) = result.await;

    #[cfg(not(target_family = "wasm"))]
    println!("GPU time: {:?}", gpu_start.elapsed());

    #[cfg(target_family = "wasm")]
    let gpu_end = web_sys::window().unwrap().performance().unwrap().now();
    #[cfg(target_family = "wasm")]
    web_sys::console::log_1(&format!("GPU time: {:?}", gpu_end - gpu_start).into());

    (trace, lookup_data)
}
