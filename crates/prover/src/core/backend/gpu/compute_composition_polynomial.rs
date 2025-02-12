use std::collections::HashMap;

use itertools::Itertools;
use wgpu::util::DeviceExt;

use super::qm31::GpuQM31;
use crate::constraint_framework::{
    INTERACTION_TRACE_IDX, ORIGINAL_TRACE_IDX, PREPROCESSED_TRACE_IDX,
};
use crate::core::backend::gpu::qm31::GpuM31;
use crate::core::backend::CpuBackend;
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::QM31;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::examples::poseidon::PoseidonElements;

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
pub struct GpuExtendedColumn {
    pub data: [[GpuM31; N_LANES as usize]; N_EXTENDED_ROWS as usize],
    pub length: u32,
}

impl From<&CircleEvaluation<CpuBackend, M31, BitReversedOrder>> for GpuExtendedColumn {
    fn from(value: &CircleEvaluation<CpuBackend, M31, BitReversedOrder>) -> Self {
        let mut data = [[GpuM31 { data: 0 }; N_LANES as usize]; N_EXTENDED_ROWS as usize];
        for (i, chunk) in value.values.chunks(N_LANES as usize).enumerate() {
            let mut row = [GpuM31 { data: 0 }; N_LANES as usize];
            for (j, &val) in chunk.iter().enumerate() {
                row[j] = val.into();
            }
            data[i] = row;
        }
        GpuExtendedColumn {
            data,
            length: N_EXTENDED_ROWS,
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
pub struct ComputeCompositionPolynomialInput {
    extended_preprocessed_trace: GpuExtendedColumn,
    extended_trace: [GpuExtendedColumn; N_COLUMNS as usize],
    extended_interaction_trace: [GpuExtendedColumn; N_INTERACTION_COLUMNS as usize],
    denom_inv: [GpuM31; 4],
    random_coeff_powers: [GpuQM31; N_CONSTRAINTS as usize],
    lookup_elements: GpuLookupElements,
    trace_domain_log_size: u32,
    eval_domain_log_size: u32,
    total_sum: GpuQM31,
}

#[derive(Debug, Clone, Copy)]
pub struct ComputeCompositionPolynomialOutput {
    pub poly: [[GpuQM31; N_LANES as usize]; N_EXTENDED_ROWS as usize],
}

#[derive(Debug, Clone)]
pub struct ComputationResults {
    pub output: ComputeCompositionPolynomialOutput,
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

impl ByteSerialize for GpuExtendedColumn {}
impl ByteSerialize for ComputeCompositionPolynomialOutput {}

impl ComputeCompositionPolynomialInput {
    fn as_bytes(&self) -> &[u8] {
        let total_size = std::mem::size_of::<ComputeCompositionPolynomialInput>();
        let mut bytes = Vec::with_capacity(total_size);
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.extended_preprocessed_trace as *const GpuExtendedColumn as *const u8,
                std::mem::size_of::<GpuExtendedColumn>(),
            )
        });
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.extended_trace as *const GpuExtendedColumn as *const u8,
                N_COLUMNS as usize * std::mem::size_of::<GpuExtendedColumn>(),
            )
        });

        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.extended_interaction_trace as *const GpuExtendedColumn as *const u8,
                N_INTERACTION_COLUMNS as usize * std::mem::size_of::<GpuExtendedColumn>(),
            )
        });

        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.denom_inv as *const GpuM31 as *const u8,
                4 * std::mem::size_of::<GpuM31>(),
            )
        });

        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.random_coeff_powers as *const GpuQM31 as *const u8,
                N_CONSTRAINTS as usize * std::mem::size_of::<GpuQM31>(),
            )
        });

        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.lookup_elements as *const GpuLookupElements as *const u8,
                std::mem::size_of::<GpuLookupElements>(),
            )
        });

        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.trace_domain_log_size as *const u32 as *const u8,
                std::mem::size_of::<u32>(),
            )
        });

        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.eval_domain_log_size as *const u32 as *const u8,
                std::mem::size_of::<u32>(),
            )
        });

        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &self.total_sum as *const GpuQM31 as *const u8,
                std::mem::size_of::<GpuQM31>(),
            )
        });

        Box::leak(bytes.into_boxed_slice())
    }
}

impl ComputeCompositionPolynomialOutput {
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
    trace: TreeVec<Vec<&CircleEvaluation<CpuBackend, M31, BitReversedOrder>>>,
    denom_inv: Vec<M31>,
    random_coeff_powers: Vec<QM31>,
    lookup_elements: PoseidonElements,
    trace_domain_log_size: u32,
    eval_domain_log_size: u32,
    total_sum: QM31,
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

    let input_data = create_gpu_input(
        trace,
        denom_inv,
        random_coeff_powers,
        lookup_elements,
        trace_domain_log_size,
        eval_domain_log_size,
        total_sum,
    );

    // Create buffers
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: input_data.as_bytes(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let buffer_size = std::mem::size_of::<ComputeCompositionPolynomialOutput>();
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
    let composition_shader = include_str!("compute_composition_polynomial.wgsl");
    let combined_shader = format!(
        "{}\n
        {}\n
        {}\n
        {}",
        qm31_shader, fraction_shader, utils_shader, composition_shader
    );
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Composition Polynomial Shader"),
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
        label: Some("Compute Composition Polynomial Bind Group Layout"),
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
        label: Some("Compute Composition Polynomial Bind Group"),
    });

    // Pipeline layout
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
        label: Some("Compute Composition Polynomial Pipeline Layout"),
    });

    // Compute pipeline
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Composition Polynomial Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("compute_composition_polynomial"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &HashMap::from([]),
            zero_initialize_workgroup_memory: true,
        },
    });

    // Create encoder
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Compute Composition Polynomial Command Encoder"),
    });

    // Dispatch the compute shader
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Composition Polynomial Compute Pass"),
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

fn create_gpu_input(
    trace: TreeVec<Vec<&CircleEvaluation<CpuBackend, M31, BitReversedOrder>>>,
    denom_inv: Vec<M31>,
    random_coeff_powers: Vec<QM31>,
    lookup_elements: PoseidonElements,
    trace_domain_log_size: u32,
    eval_domain_log_size: u32,
    total_sum: QM31,
) -> ComputeCompositionPolynomialInput {
    let extended_preprocessed_trace_gpu =
        GpuExtendedColumn::from(trace.0[PREPROCESSED_TRACE_IDX][0]);

    let extended_trace_gpu: [GpuExtendedColumn; N_COLUMNS as usize] = trace.0[ORIGINAL_TRACE_IDX]
        .iter()
        .map(|eval| GpuExtendedColumn::from(*eval))
        .collect_vec()
        .try_into()
        .expect("Wrong length");

    let extended_interaction_trace_gpu: [GpuExtendedColumn; N_INTERACTION_COLUMNS as usize] = trace
        .0[INTERACTION_TRACE_IDX]
        .iter()
        .map(|eval| GpuExtendedColumn::from(*eval))
        .collect_vec()
        .try_into()
        .expect("Wrong length");

    let denom_inv_gpu: [GpuM31; 4] = denom_inv
        .into_iter()
        .map(GpuM31::from)
        .collect::<Vec<_>>()
        .try_into()
        .expect("Wrong length");

    let random_coeff_powers_gpu: [GpuQM31; N_CONSTRAINTS as usize] = random_coeff_powers
        .into_iter()
        .map(GpuQM31::from)
        .collect::<Vec<_>>()
        .try_into()
        .expect("Wrong length");

    let lookup_elements_gpu = GpuLookupElements::from(lookup_elements);

    ComputeCompositionPolynomialInput {
        extended_preprocessed_trace: extended_preprocessed_trace_gpu,
        extended_trace: extended_trace_gpu,
        extended_interaction_trace: extended_interaction_trace_gpu,
        denom_inv: denom_inv_gpu,
        random_coeff_powers: random_coeff_powers_gpu,
        lookup_elements: lookup_elements_gpu,
        trace_domain_log_size,
        eval_domain_log_size,
        total_sum: total_sum.into(),
    }
}

pub async fn compute_composition_polynomial_gpu<'a>(
    trace: TreeVec<Vec<&CircleEvaluation<CpuBackend, M31, BitReversedOrder>>>,
    denom_inv: Vec<M31>,
    random_coeff_powers: Vec<QM31>,
    lookup_elements: PoseidonElements,
    trace_domain_log_size: u32,
    eval_domain_log_size: u32,
    total_sum: QM31,
) -> ComputationResults {
    let instance = init(
        trace,
        denom_inv,
        random_coeff_powers,
        lookup_elements,
        trace_domain_log_size,
        eval_domain_log_size,
        total_sum,
    )
    .await;
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
        let output = ComputeCompositionPolynomialOutput::from_bytes(&data);
        drop(data);
        instance.staging_buffer.unmap();
        output
    };

    let output = result.await;
    ComputationResults { output }
}
