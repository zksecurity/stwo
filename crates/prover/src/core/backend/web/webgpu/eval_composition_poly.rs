use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::sync::Arc;

use itertools::Itertools;

use super::constants::*;
use super::gpu_types::*;
use super::qm31::GpuQM31;
use super::ByteSerialize;
use crate::constraint_framework::WebDomainEvaluator;
use crate::core::backend::cpu::circle::circle_twiddles_from_line_twiddles;
use crate::core::backend::web::webgpu::qm31::GpuM31;
use crate::core::backend::web::WebBackend;
use crate::core::backend::CpuBackend;
use crate::core::fields::m31::{BaseField, M31};
use crate::core::fields::qm31::QM31;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::{CircleDomain, CirclePoly, PolyOps};
use crate::core::poly::utils::domain_line_twiddles_from_tree;
use crate::examples::poseidon::PoseidonElements;
// pub struct
pub struct WgpuInstance {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub input_buffer: wgpu::Buffer,
    pub output_buffer: wgpu::Buffer,
    pub staging_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub evaluate_line_twiddle_pipeline: wgpu::ComputePipeline,
    pub evaluate_circle_twiddle_pipeline: wgpu::ComputePipeline,
    pub composition_polynomial_compute_pipeline: wgpu::ComputePipeline,
}

pub struct EvalCompositionPolynomialArgs<'a> {
    pub original_trace: &'a TreeVec<Vec<&'a CirclePoly<WebBackend>>>,
    pub eval_domain: CircleDomain,
    pub denom_inv: Vec<M31>,
    pub random_coeff_powers: Vec<QM31>,
    pub lookup_elements: &'a PoseidonElements,
    pub trace_domain_log_size: u32,
    pub eval_domain_log_size: u32,
    pub log_size: u32,
    pub total_sum: QM31,
}

pub async fn compute_composition_polynomial_wgpu(
    input: Arc<ComputeCompositionPolynomialInput>,
    instance: &WgpuInstance,
) -> Arc<ComputeCompositionPolynomialOutput> {
    let encoder = init_encoder(&instance);
    instance
        .queue
        .write_buffer(&instance.input_buffer, 0, input.as_bytes());
    instance.queue.submit(Some(encoder.finish()));
    let output_slice = instance.staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    output_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    instance
        .device
        .poll(wgpu::Maintain::wait())
        .panic_on_timeout();

    let _ = receiver.recv_async().await.unwrap();
    let data = output_slice.get_mapped_range();
    let output = ComputeCompositionPolynomialOutput::from_bytes(&data);
    drop(data);
    instance.staging_buffer.unmap();
    Arc::new(output)
}

pub fn create_composition_polynomial_gpu_input(
    eval: &mut WebDomainEvaluator<'_>,
    lookup_elements: &PoseidonElements,
) -> ComputeCompositionPolynomialInput {
    web_sys::console::log_1(&format!("create_composition_polynomial_gpu_input").into());

    let original_trace_gpu: [GpuOriginalColumn; N_ORIGINAL_TRACE_COLUMNS as usize] =
        std::array::from_fn(|i| {
            let itm = eval
                .trace_poly
                .iter()
                .flatten()
                .nth(i)
                .expect("trace_poly too short");
            GpuOriginalColumn::from(itm)
        });

    web_sys::console::log_1(&format!("original trace done").into());

    // flatten twiddles
    let twiddles = CpuBackend::precompute_twiddles(eval.eval_domain.half_coset);
    let line_twiddles = domain_line_twiddles_from_tree(eval.eval_domain, &twiddles.twiddles);
    let mut twiddle_input = Twiddles {
        line_twiddles_layer_count: line_twiddles.len() as u32,
        line_twiddles_sizes: [0; N_LINE_TWIDDLES_SIZE as usize],
        line_twiddles_offsets: [0; N_LINE_TWIDDLES_SIZE as usize],
        line_twiddles_flat: [GpuM31 { 0: 0 }; N_LINE_TWIDDLES_FLAT_SIZE as usize],
        circle_twiddles: [GpuM31 { 0: 0 }; N_CIRCLE_TWIDDLES_SIZE as usize],
        circle_twiddles_size: 0,
    };

    web_sys::console::log_1(&format!("twiddles done").into());

    let mut offset = 0;
    for (i, twiddle_layer) in line_twiddles.iter().enumerate() {
        let size = twiddle_layer.len();
        twiddle_input.line_twiddles_sizes[i] = size as u32;
        twiddle_input.line_twiddles_offsets[i] = offset as u32;

        // Flatten into flat buffer
        for (j, &twiddle) in twiddle_layer.iter().enumerate() {
            twiddle_input.line_twiddles_flat[offset + j] = GpuM31::from(twiddle);
        }
        offset += size;
    }

    web_sys::console::log_1(&format!("line twiddles done").into());

    // circle twiddles
    let circle = circle_twiddles_from_line_twiddles(line_twiddles[0]);
    let circle_len = circle.try_len().unwrap();
    debug_assert!(circle_len <= N_CIRCLE_TWIDDLES_SIZE as usize);
    for (i, tw) in circle.enumerate() {
        twiddle_input.circle_twiddles[i] = GpuM31::from(tw);
    }
    twiddle_input.circle_twiddles_size = circle_len as u32;

    let denom_inv_gpu: [GpuM31; 4] = {
        let mut arr = [GpuM31 { 0: 0 }; 4];
        for i in 0..4 {
            arr[i] = GpuM31::from(eval.denom_inv[i]);
        }
        arr
    };

    let mut random_coeff_powers_gpu: [GpuQM31; N_CONSTRAINTS as usize] =
        [GpuQM31 { 0: [0, 0, 0, 0] }; N_CONSTRAINTS as usize];

    for i in 0..N_CONSTRAINTS as usize {
        random_coeff_powers_gpu[i] = GpuQM31::from(eval.random_coeff_powers[i]);
    }

    let lookup_elements_gpu = GpuLookupElements::from(lookup_elements);

    web_sys::console::log_1(&format!("input done").into());

    // print size of ComputeCompositionPolynomialInput
    web_sys::console::log_1(
        &format!(
            "size of ComputeCompositionPolynomialInput: {:?}",
            std::mem::size_of::<ComputeCompositionPolynomialInput>()
        )
        .into(),
    );

    ComputeCompositionPolynomialInput {
        original_trace: original_trace_gpu,
        twiddles: twiddle_input,
        denom_inv: denom_inv_gpu,
        random_coeff_powers: random_coeff_powers_gpu,
        lookup_elements: lookup_elements_gpu,
        trace_domain_log_size: eval.trace_domain_log_size,
        eval_domain_log_size: eval.eval_domain.log_size(),
        cumsum_shift: (eval.claimed_sum / BaseField::from_u32_unchecked(1 << eval.log_size)).into(),
    }
}

pub async fn init_wgpu_instance() -> WgpuInstance {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();
    // let mut limit = wgpu::Limits::default();
    let adapter_limits = adapter.limits();

    web_sys::console::log_1(
        &format!("max_buffer_size: {:?}", adapter_limits.max_buffer_size).into(),
    );

    // let _limits = wgpu::Limits {
    //     // bump storage‐binding to 512 MiB
    //     max_storage_buffer_binding_size: 512 * 1024 * 1024,
    //     // bump overall buffer size to 4 GiB (adapter reports it supports this)
    //     max_buffer_size: adapter_limits.max_buffer_size / 2, // or hard‑code 4 GiB if you’ve
    //     // checked
    //     ..adapter_limits
    // };

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::default(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .unwrap();

    web_sys::console::log_1(&format!("device: {:?}", device).into());

    // Load shader
    let constants_shader = include_str!("constants.wgsl")
        .replace("${N_ROWS}", &N_ROWS.to_string())
        .replace("${N_CONSTRAINTS]", &N_CONSTRAINTS.to_string());

    let qm31_shader = include_str!("qm31.wgsl");
    let fraction_shader = include_str!("fraction.wgsl");
    let utils_shader = include_str!("utils.wgsl");
    let extend_trace_shader = include_str!("extend_trace.wgsl");
    let composition_shader = include_str!("eval_composition_poly.wgsl");

    // Load extend trace shader
    let extend_trace_combined_shader = format!(
        "{}\n
        {}\n
        {}\n
        {}\n
        {}",
        constants_shader, qm31_shader, fraction_shader, utils_shader, extend_trace_shader
    );
    let extend_trace_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Extend Trace Shader"),
        source: wgpu::ShaderSource::Wgsl(extend_trace_combined_shader.into()),
    });

    web_sys::console::log_1(&format!("extend trace shader done").into());

    // Load composition polynomial shader
    let composition_polynomial_combined_shader = format!(
        "{}\n
        {}\n
        {}\n
        {}\n
        {}",
        constants_shader, qm31_shader, fraction_shader, utils_shader, composition_shader
    );

    let composition_polynomial_shader_module =
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Composition Polynomial Shader"),
            source: wgpu::ShaderSource::Wgsl(composition_polynomial_combined_shader.into()),
        });

    web_sys::console::log_1(&format!("shader done").into());

    // Create buffers
    let input_buffer_size = std::mem::size_of::<ComputeCompositionPolynomialInput>();
    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Input Buffer"),
        size: input_buffer_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let buffer_size = std::mem::size_of::<ComputeCompositionPolynomialOutput>();
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: buffer_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Copy output to staging buffer for read access
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: buffer_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let extend_trace_buffer_size = std::mem::size_of::<ExtendTraceOutput>();
    let extend_trace_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Extend Trace Buffer"),
        size: extend_trace_buffer_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    web_sys::console::log_1(&format!("buffer done").into());

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
            // Binding 2: Extend trace buffer
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
            wgpu::BindGroupEntry {
                binding: 2,
                resource: extend_trace_buffer.as_entire_binding(),
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
    let evaluate_line_twiddle_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Extend Trace Line Twiddle Pipeline"),
            layout: Some(&pipeline_layout),
            module: &extend_trace_shader_module,
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
            module: &extend_trace_shader_module,
            entry_point: Some("evaluate_circle_twiddle"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &HashMap::from([]),
                zero_initialize_workgroup_memory: true,
            },
        });

    let composition_polynomial_compute_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Composition Polynomial Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &composition_polynomial_shader_module,
            entry_point: Some("compute_composition_polynomial"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &HashMap::from([]),
                zero_initialize_workgroup_memory: true,
            },
        });

    WgpuInstance {
        instance,
        adapter,
        device,
        queue,
        input_buffer,
        output_buffer,
        staging_buffer,
        bind_group,
        evaluate_line_twiddle_pipeline,
        evaluate_circle_twiddle_pipeline,
        composition_polynomial_compute_pipeline,
    }
}

pub fn init_encoder(instance: &WgpuInstance) -> wgpu::CommandEncoder {
    let mut encoder = instance
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Composition Polynomial Command Encoder"),
        });
    // Dispatch the compute shader
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Composition Polynomial Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_bind_group(0, &instance.bind_group, &[]);

        compute_pass.set_pipeline(&instance.evaluate_line_twiddle_pipeline);
        compute_pass.dispatch_workgroups(1, N_EXTEND_TRACE_WORKGROUPS, 1);

        compute_pass.set_pipeline(&instance.evaluate_circle_twiddle_pipeline);
        compute_pass.dispatch_workgroups(1, N_EXTEND_TRACE_WORKGROUPS, 1);

        compute_pass.set_pipeline(&instance.composition_polynomial_compute_pipeline);
        compute_pass.dispatch_workgroups(N_WORKGROUPS, 1, 1);
    }

    // Copy output to staging buffer for read access
    encoder.copy_buffer_to_buffer(
        &instance.output_buffer,
        0,
        &instance.staging_buffer,
        0,
        instance.staging_buffer.size(),
    );

    encoder
}
