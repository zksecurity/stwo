use std::collections::HashMap;

use wgpu::util::DeviceExt;

use super::qm31::GpuQM31;
use crate::core::backend::gpu::qm31::GpuM31;
use crate::core::backend::simd::column::BaseColumn;
use crate::examples::poseidon::{LookupData, PoseidonElements};

pub const N_ROWS: u32 = 32;
pub const N_STATE: u32 = 16;
pub const N_LOG_INSTANCES_PER_ROW: u32 = 3;
pub const N_INSTANCES_PER_ROW: u32 = 1 << N_LOG_INSTANCES_PER_ROW;
pub const N_LANES: u32 = 16;
pub const N_EXTENDED_ROWS: u32 = N_ROWS * 4;
pub const N_CONSTRAINTS: u32 = 1144;
pub const N_COLUMNS: u32 = 1264;
pub const N_INTERACTION_COLUMNS: u32 = N_INSTANCES_PER_ROW * 4;
pub const N_WORKGROUPS: u32 = N_EXTENDED_ROWS * N_LANES / THREADS_PER_WORKGROUP;
pub const THREADS_PER_WORKGROUP: u32 = 256;
pub const N_HALF_FULL_ROUNDS: u32 = 4;
pub const N_PARTIAL_ROUNDS: u32 = 14;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct GpuLookupData {
    initial_state: [[GpuBaseColumn; N_STATE as usize]; N_INSTANCES_PER_ROW as usize],
    final_state: [[GpuBaseColumn; N_STATE as usize]; N_INSTANCES_PER_ROW as usize],
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
pub struct GpuBaseColumn {
    data: [[GpuM31; N_LANES as usize]; N_ROWS as usize],
    length: u32,
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
pub struct GenInteractionTraceInput {
    log_size: u32,
    lookup_data: GpuLookupData,
    lookup_elements: GpuLookupElements,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuQM31Column {
    pub data: [GpuQM31; N_ROWS as usize],
    pub length: u32,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct GenInteractionTraceOutput {
    pub interaction_trace_qm31: [GpuQM31Column; N_INSTANCES_PER_ROW as usize],
    pub interaction_trace_buffers: [GpuQM31Column; 2],
    pub total_sum: GpuQM31,
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

impl ByteSerialize for GpuLookupData {}
impl ByteSerialize for GpuLookupElements {}
impl ByteSerialize for GpuBaseColumn {}
impl ByteSerialize for GpuQM31Column {}
impl ByteSerialize for GenInteractionTraceOutput {}

impl GenInteractionTraceInput {
    fn as_bytes(&self) -> &[u8] {
        let total_size = std::mem::size_of::<GenInteractionTraceInput>();
        let mut bytes = Vec::with_capacity(total_size);

        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.log_size as *const u32 as *const u8,
                std::mem::size_of::<u32>(),
            )
        });

        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.lookup_data as *const GpuLookupData as *const u8,
                std::mem::size_of::<GpuLookupData>(),
            )
        });

        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.lookup_elements as *const GpuLookupElements as *const u8,
                std::mem::size_of::<GpuLookupElements>(),
            )
        });

        Box::leak(bytes.into_boxed_slice())
    }
}

impl GenInteractionTraceOutput {
    fn from_bytes(bytes: &[u8]) -> Self {
        unsafe { *(bytes.as_ptr() as *const Self) }
    }
}

pub struct WgpuInstance {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub staging_buffer: wgpu::Buffer,
    pub encoder: wgpu::CommandEncoder,
}

async fn init(
    log_size: u32,
    lookup_data: LookupData,
    lookup_elements: &PoseidonElements,
) -> WgpuInstance {
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

    let input_data = create_interaction_trace_input(log_size, lookup_data, lookup_elements);

    // Create buffers
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: input_data.as_bytes(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let buffer_size = std::mem::size_of::<GenInteractionTraceOutput>();
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: buffer_size as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Load shader
    let qm31_shader = include_str!("qm31.wgsl");
    let fraction_shader = include_str!("fraction.wgsl");
    let utils_shader = include_str!("utils.wgsl");
    let trace_constants = include_str!("gen_trace_interpolate_columns_constants.wgsl");
    let interaction_trace_shader = include_str!("gen_interaction_trace.wgsl");
    let combined_shader = format!(
        "{}\n
        {}\n
        {}\n
        {}\n    
        {}",
        qm31_shader, fraction_shader, utils_shader, trace_constants, interaction_trace_shader
    );
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Interaction Trace Shader"),
        source: wgpu::ShaderSource::Wgsl(combined_shader.into()),
    });

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
        ],
        label: Some("Compute Interaction Trace Bind Group Layout"),
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
        ],
        label: Some("Compute Interaction Trace Bind Group"),
    });

    // Pipeline layout
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
        label: Some("Compute Interaction Trace Pipeline Layout"),
    });

    // Compute pipeline
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Interaction Trace Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("compute_interaction_trace"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &HashMap::from([]),
            zero_initialize_workgroup_memory: true,
        },
    });

    // Create encoder
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Compute Interaction Trace Command Encoder"),
    });

    // Dispatch the compute shader
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Interaction Trace Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(N_WORKGROUPS, 1, 1);
    }

    // Copy output to staging buffer for read access
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: buffer_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, staging_buffer.size());

    WgpuInstance {
        instance,
        adapter,
        device,
        queue,
        staging_buffer,
        encoder,
    }
}

impl From<&BaseColumn> for GpuBaseColumn {
    fn from(value: &BaseColumn) -> Self {
        let gpu_data = value
            .data
            .iter()
            .map(|packed| {
                let m31_array = packed.to_array(); // [M31; N_LANES]
                let mut gpu_m31_array = [GpuM31 { data: 0 }; N_LANES as usize];
                for (i, m31_val) in m31_array.iter().enumerate() {
                    gpu_m31_array[i] = GpuM31 { data: m31_val.0 };
                }
                gpu_m31_array
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        GpuBaseColumn {
            data: gpu_data,
            length: value.length as u32,
        }
    }
}
impl LookupData {
    fn to_gpu_base_columns(&self) -> GpuLookupData {
        let mut initial_gpu = [[GpuBaseColumn {
            data: [[GpuM31 { data: 0 }; N_LANES as usize]; N_ROWS as usize],
            length: N_LANES * N_ROWS,
        }; N_STATE as usize]; N_INSTANCES_PER_ROW as usize];
        for (row_idx, row) in self.initial_state.iter().enumerate() {
            for (col_idx, base_col) in row.iter().enumerate() {
                initial_gpu[row_idx][col_idx] = GpuBaseColumn::from(base_col);
            }
        }

        let mut final_gpu = [[GpuBaseColumn {
            data: [[GpuM31 { data: 0 }; N_LANES as usize]; N_ROWS as usize],
            length: N_LANES * N_ROWS,
        }; N_STATE as usize]; N_INSTANCES_PER_ROW as usize];
        for (row_idx, row) in self.final_state.iter().enumerate() {
            for (col_idx, base_col) in row.iter().enumerate() {
                final_gpu[row_idx][col_idx] = GpuBaseColumn::from(base_col);
            }
        }

        GpuLookupData {
            initial_state: initial_gpu,
            final_state: final_gpu,
        }
    }
}

fn create_interaction_trace_input(
    log_size: u32,
    lookup_data: LookupData,
    lookup_elements: &PoseidonElements,
) -> GenInteractionTraceInput {
    let gpu_lookup_elements = GpuLookupElements {
        z: lookup_elements.0.z.into(),
        alpha: lookup_elements.0.alpha.into(),
        alpha_powers: lookup_elements.0.alpha_powers.map(|p| p.into()),
    };

    GenInteractionTraceInput {
        log_size,
        lookup_data: lookup_data.to_gpu_base_columns(),
        lookup_elements: gpu_lookup_elements,
    }
}

pub async fn compute_interaction_trace_gpu<'a>(
    log_size: u32,
    lookup_data: LookupData,
    lookup_elements: &PoseidonElements,
) -> GenInteractionTraceOutput {
    let instance = init(log_size, lookup_data, lookup_elements).await;
    instance.queue.submit(Some(instance.encoder.finish()));
    let output_slice = instance.staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    output_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    instance
        .device
        .poll(wgpu::Maintain::wait())
        .panic_on_timeout();
    let result = async {
        receiver.recv_async().await.unwrap().unwrap();
        let data = output_slice.get_mapped_range();
        let output = GenInteractionTraceOutput::from_bytes(&data);
        drop(data);
        instance.staging_buffer.unmap();
        output
    };

    let output = result.await;
    output
}
