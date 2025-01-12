use wgpu::util::DeviceExt;

use super::qm31::GpuQM31;
use crate::core::fields::qm31::QM31;
use crate::core::lookups::utils::Fraction as CpuFraction;

#[derive(Copy, Clone, Debug)]
pub struct GpuFraction {
    numerator: GpuQM31,
    denominator: GpuQM31,
}

impl From<CpuFraction<GpuQM31, GpuQM31>> for GpuFraction {
    fn from(f: CpuFraction<GpuQM31, GpuQM31>) -> Self {
        GpuFraction {
            numerator: f.numerator.into(),
            denominator: f.denominator.into(),
        }
    }
}

impl From<CpuFraction<QM31, QM31>> for GpuFraction {
    fn from(f: CpuFraction<QM31, QM31>) -> Self {
        GpuFraction {
            numerator: f.numerator.into(),
            denominator: f.denominator.into(),
        }
    }
}

// GPU computation structures
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ComputeInputs {
    first: GpuFraction,
    second: GpuFraction,
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

impl ByteSerialize for GpuFraction {}
impl ByteSerialize for ComputeInputs {}

pub struct GpuFractionInstance {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub input_buffer: wgpu::Buffer,
    pub output_buffer: wgpu::Buffer,
    pub staging_buffer: wgpu::Buffer,
}

impl GpuFractionInstance {
    pub async fn new<T: ByteSerialize>(input_data: &T, output_size: usize) -> Self {
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
                    label: Some("Field Operations Device"),
                    required_features: wgpu::Features::SHADER_INT64,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .unwrap();

        // Create input buffer
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Field Input Buffer"),
            contents: input_data.as_bytes(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Create output buffer
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Field Output Buffer"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffer for reading results
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Field Staging Buffer"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            input_buffer,
            output_buffer,
            staging_buffer,
        }
    }

    pub fn create_pipeline(
        &self,
        shader_source: &str,
        entry_point: &str,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroup) {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Field Operations Shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
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
                    label: Some("Field Operations Bind Group Layout"),
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Field Operations Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Field Operations Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.output_buffer.as_entire_binding(),
                },
            ],
            label: Some("Field Operations Bind Group"),
        });

        (pipeline, bind_group)
    }

    pub async fn run_computation<T: ByteSerialize + Copy>(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroup_count: (u32, u32, u32),
    ) -> T {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Field Operations Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Field Operations Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(
                workgroup_count.0,
                workgroup_count.1,
                workgroup_count.2,
            );
        }

        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.staging_buffer,
            0,
            self.staging_buffer.size(),
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.staging_buffer.slice(..);
        let (tx, rx) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::wait());

        rx.recv_async().await.unwrap().unwrap();
        let data = buffer_slice.get_mapped_range();
        let result = *T::from_bytes(&data);
        drop(data);
        self.staging_buffer.unmap();

        result
    }
}

#[derive(Debug)]
pub enum GpuFractionOperation {
    Add,
    Zero,
}

impl GpuFractionOperation {
    pub fn shader_source(&self) -> String {
        let base_source = include_str!("fraction.wgsl");
        let qm31_source = include_str!("qm31.wgsl");

        let inputs = r#"
            struct Inputs {
                first: Fraction,
                second: Fraction,
            }

            @group(0) @binding(0) var<storage, read> inputs: Inputs;
        "#;

        let output = r#"
            @group(0) @binding(1) var<storage, read_write> output: Fraction;
        "#;

        let operation = match self {
            GpuFractionOperation::Add => {
                r#"
                @compute @workgroup_size(1)
                fn main() {{
                    let result = fraction_add(inputs.first, inputs.second);
                    output = result;
                }}
                "#
            }
            GpuFractionOperation::Zero => {
                r#"
                @compute @workgroup_size(1)
                fn main() {{
                    let result = fraction_zero();
                    output = result;
                }}
                "#
            }
        };

        format!(
            "{}\n{}\n{}\n{}\n{}\n",
            qm31_source, base_source, inputs, output, operation
        )
    }
}

impl From<GpuFraction> for CpuFraction<QM31, QM31> {
    fn from(f: GpuFraction) -> Self {
        CpuFraction::new(f.numerator.into(), f.denominator.into())
    }
}

pub async fn compute_fraction_operation(
    a: CpuFraction<QM31, QM31>,
    b: CpuFraction<QM31, QM31>,
    operation: GpuFractionOperation,
) -> CpuFraction<QM31, QM31> {
    let inputs = ComputeInputs {
        first: a.into(),
        second: b.into(),
    };

    let instance = GpuFractionInstance::new(&inputs, std::mem::size_of::<GpuFraction>()).await;

    let shader_source = operation.shader_source();
    let (pipeline, bind_group) = instance.create_pipeline(&shader_source, "main");

    let gpu_result = instance
        .run_computation::<GpuFraction>(&pipeline, &bind_group, (1, 1, 1))
        .await;

    gpu_result.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::fields::qm31::QM31;

    #[test]
    fn test_fraction_add() {
        // CPU implementation
        let cpu_a = CpuFraction::new(
            QM31::from_u32_unchecked(1u32, 0u32, 0u32, 0u32),
            QM31::from_u32_unchecked(3u32, 0u32, 0u32, 0u32),
        );
        let cpu_b = CpuFraction::new(
            QM31::from_u32_unchecked(2u32, 0u32, 0u32, 0u32),
            QM31::from_u32_unchecked(6u32, 0u32, 0u32, 0u32),
        );
        let cpu_result = cpu_a + cpu_b;

        // GPU implementation
        let gpu_result = pollster::block_on(compute_fraction_operation(
            cpu_a,
            cpu_b,
            GpuFractionOperation::Add,
        ));

        assert_eq!(cpu_result.numerator, gpu_result.numerator);
        assert_eq!(cpu_result.denominator, gpu_result.denominator);
    }
}
