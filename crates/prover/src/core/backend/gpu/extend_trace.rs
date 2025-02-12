use std::collections::HashMap;

use itertools::Itertools;
use wgpu::util::DeviceExt;

use crate::core::backend::cpu::circle::circle_twiddles_from_line_twiddles;
use crate::core::backend::gpu::qm31::GpuM31;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Column, CpuBackend};
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::{CircleDomain, CirclePoly, PolyOps};
use crate::core::poly::utils::domain_line_twiddles_from_tree;

pub const N_ROWS: u32 = 64;
pub const N_STATE: u32 = 16;
pub const N_LOG_INSTANCES_PER_ROW: u32 = 3;
pub const N_INSTANCES_PER_ROW: u32 = 1 << N_LOG_INSTANCES_PER_ROW;
pub const N_LANES: u32 = 16;
pub const N_EXTENDED_ROWS: u32 = N_ROWS * 4;
pub const N_ORIGINAL_ROWS: u32 = N_ROWS;
pub const N_CONSTRAINTS: u32 = 1144;
pub const N_COLUMNS: u32 = 1264;
pub const N_INTERACTION_COLUMNS: u32 = N_INSTANCES_PER_ROW * 4;
pub const N_WORKGROUPS: u32 = N_EXTENDED_ROWS * N_LANES / THREADS_PER_WORKGROUP;
pub const THREADS_PER_WORKGROUP: u32 = 256;
pub const N_HALF_FULL_ROUNDS: u32 = 4;
pub const N_PARTIAL_ROUNDS: u32 = 14;

pub const N_LINE_TWIDDLES_SIZE: u32 = N_EXTENDED_ROWS * N_LANES;
pub const N_LINE_TWIDDLES_FLAT_SIZE: u32 = N_LINE_TWIDDLES_SIZE * 2;
pub const N_CIRCLE_TWIDDLES_SIZE: u32 = N_LINE_TWIDDLES_SIZE * 2;
pub const N_ORIGINAL_TRACE_COLUMNS: u32 = 1 + N_COLUMNS + N_INTERACTION_COLUMNS;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuExtendedColumn {
    pub data: [[GpuM31; N_LANES as usize]; N_EXTENDED_ROWS as usize],
    pub length: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Twiddles {
    pub circle_twiddles: [GpuM31; N_CIRCLE_TWIDDLES_SIZE as usize],
    pub circle_twiddles_size: u32,
    pub line_twiddles_flat: [GpuM31; N_LINE_TWIDDLES_FLAT_SIZE as usize],
    pub line_twiddles_layer_count: u32,
    pub line_twiddles_sizes: [u32; N_LINE_TWIDDLES_SIZE as usize],
    pub line_twiddles_offsets: [u32; N_LINE_TWIDDLES_SIZE as usize],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuOriginalColumn {
    pub coeffs: [GpuM31; (N_LANES * N_ORIGINAL_ROWS) as usize],
    pub length: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuExtended1DColumn {
    pub data: [GpuM31; (N_LANES * N_EXTENDED_ROWS) as usize],
    pub length: u32,
}

impl From<&&&CirclePoly<SimdBackend>> for GpuOriginalColumn {
    fn from(value: &&&CirclePoly<SimdBackend>) -> Self {
        let mut coeffs = [GpuM31 { data: 0 }; (N_LANES * N_ORIGINAL_ROWS) as usize];
        let coeffs_vec = value.coeffs.to_cpu();
        for (i, &coeff) in coeffs_vec.iter().enumerate() {
            coeffs[i] = coeff.into();
        }

        GpuOriginalColumn {
            coeffs,
            length: N_LANES * N_ORIGINAL_ROWS,
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

#[derive(Debug, Clone, Copy)]
pub struct ExtendTraceInput {
    pub original_trace: [GpuOriginalColumn; N_ORIGINAL_TRACE_COLUMNS as usize],
    pub twiddles: Twiddles,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct ExtendTraceOutput {
    pub extended_trace: [GpuExtended1DColumn; N_ORIGINAL_TRACE_COLUMNS as usize],
}

#[derive(Debug, Clone)]
pub struct ExtendTraceResults {
    pub output: ExtendTraceOutput,
}

impl ByteSerialize for GpuExtendedColumn {}
impl ByteSerialize for GpuExtended1DColumn {}
impl ByteSerialize for GpuOriginalColumn {}
impl ByteSerialize for ExtendTraceOutput {}

impl ExtendTraceInput {
    fn as_bytes(&self) -> &[u8] {
        let total_size = std::mem::size_of::<ExtendTraceInput>();
        println!("total_size: {}", total_size);
        let original_trace_size =
            N_ORIGINAL_TRACE_COLUMNS as usize * std::mem::size_of::<GpuOriginalColumn>();
        println!("original_trace_size: {}", original_trace_size);
        let twiddles_size = std::mem::size_of::<Twiddles>();
        println!("twiddles_size: {}", twiddles_size);
        let mut bytes = Vec::with_capacity(total_size);
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.original_trace as *const GpuOriginalColumn as *const u8,
                N_ORIGINAL_TRACE_COLUMNS as usize * std::mem::size_of::<GpuOriginalColumn>(),
            )
        });
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.twiddles as *const Twiddles as *const u8,
                std::mem::size_of::<Twiddles>(),
            )
        });
        Box::leak(bytes.into_boxed_slice())
    }
}

impl ExtendTraceOutput {
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
    original_trace: TreeVec<Vec<&&CirclePoly<SimdBackend>>>,
    eval_domain: CircleDomain,
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

    let input_data = create_extend_trace_gpu_input(original_trace, eval_domain);

    // Create buffers
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Extend Trace Input Buffer"),
        contents: input_data.as_bytes(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let buffer_size = std::mem::size_of::<ExtendTraceOutput>();
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Extend TraceOutput Buffer"),
        size: buffer_size as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Load shader
    let qm31_shader = include_str!("qm31.wgsl");
    let fraction_shader = include_str!("fraction.wgsl");
    let utils_shader = include_str!("utils.wgsl");
    let extend_trace_shader = include_str!("extend_trace.wgsl");
    let combined_shader = format!(
        "{}\n
        {}\n
        {}\n
        {}",
        qm31_shader, fraction_shader, utils_shader, extend_trace_shader
    );
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Extend Trace Shader"),
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
        label: Some("Extend Trace Bind Group Layout"),
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
        label: Some("Extend Trace Bind Group"),
    });

    // Pipeline layout
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
        label: Some("Extend Trace Pipeline Layout"),
    });

    // Compute pipeline
    let evaluate_line_twiddle_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Extend Trace Line Twiddle Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("evaluate_line_twiddle"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &HashMap::from([]),
                zero_initialize_workgroup_memory: true,
            },
        });

    let evaluate_circle_twiddle_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Extend Trace Circle Twiddle Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("evaluate_circle_twiddle"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &HashMap::from([]),
                zero_initialize_workgroup_memory: true,
            },
        });

    // Create encoder
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Extend Trace Command Encoder"),
    });

    // Dispatch the compute shader
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Extend Trace Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.set_pipeline(&evaluate_line_twiddle_pipeline);
        compute_pass.dispatch_workgroups(1, 256, 1);

        compute_pass.set_pipeline(&evaluate_circle_twiddle_pipeline);
        compute_pass.dispatch_workgroups(1, 256, 1);
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

fn create_extend_trace_gpu_input(
    original_trace: TreeVec<Vec<&&CirclePoly<SimdBackend>>>,
    eval_domain: CircleDomain,
) -> ExtendTraceInput {
    // original_trace is 2-d vec, want to flatten it to 1-d vec
    let original_trace_gpu: [GpuOriginalColumn; N_ORIGINAL_TRACE_COLUMNS as usize] = original_trace
        .iter()
        .flatten()
        .map(|eval| GpuOriginalColumn::from(eval))
        .collect_vec()
        .try_into()
        .expect("Wrong length");

    let twiddles = CpuBackend::precompute_twiddles(eval_domain.half_coset);
    println!("eval domain log size: {}", eval_domain.log_size());

    // line twiddles
    let line_twiddles = domain_line_twiddles_from_tree(eval_domain, &twiddles.twiddles);
    println!("line_twiddles length: {}", line_twiddles.len());
    println!("line_twiddles[0] length: {}", line_twiddles[0].len());

    let mut twiddle_input = Twiddles {
        line_twiddles_layer_count: line_twiddles.len() as u32,
        line_twiddles_sizes: [0; N_LINE_TWIDDLES_SIZE as usize],
        line_twiddles_offsets: [0; N_LINE_TWIDDLES_SIZE as usize],
        line_twiddles_flat: [GpuM31 { data: 0 }; N_LINE_TWIDDLES_FLAT_SIZE as usize],
        circle_twiddles: [GpuM31 { data: 0 }; N_CIRCLE_TWIDDLES_SIZE as usize],
        circle_twiddles_size: 0,
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

    // circle twiddles
    let circle_twiddles: Vec<GpuM31> = circle_twiddles_from_line_twiddles(line_twiddles[0])
        .map(|twiddle| twiddle.into())
        .collect();
    println!("circle_twiddles length: {}", circle_twiddles.len());
    twiddle_input.circle_twiddles[..circle_twiddles.len()].copy_from_slice(&circle_twiddles);
    twiddle_input.circle_twiddles_size = circle_twiddles.len() as u32;

    ExtendTraceInput {
        original_trace: original_trace_gpu,
        twiddles: twiddle_input,
    }
}

pub async fn extended_trace_gpu<'a>(
    original_trace: TreeVec<Vec<&&CirclePoly<SimdBackend>>>,
    eval_domain: CircleDomain,
) -> ExtendTraceResults {
    let instance = init(original_trace, eval_domain).await;
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
        let output = ExtendTraceOutput::from_bytes(&data);
        drop(data);
        instance.staging_buffer.unmap();
        output
    };

    let output = result.await;
    ExtendTraceResults { output }
}
