use wgpu::util::DeviceExt;

use crate::core::fields::cm31::CM31;
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::QM31;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuM31 {
    pub data: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuCM31 {
    pub a: GpuM31,
    pub b: GpuM31,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuQM31 {
    pub a: GpuCM31,
    pub b: GpuCM31,
}

impl From<M31> for GpuM31 {
    fn from(value: M31) -> Self {
        GpuM31 { data: value.into() }
    }
}

impl From<CM31> for GpuCM31 {
    fn from(value: CM31) -> Self {
        GpuCM31 {
            a: value.0.into(),
            b: value.1.into(),
        }
    }
}

impl From<QM31> for GpuQM31 {
    fn from(value: QM31) -> Self {
        GpuQM31 {
            a: value.0.into(),
            b: value.1.into(),
        }
    }
}

impl From<GpuQM31> for QM31 {
    fn from(value: GpuQM31) -> Self {
        QM31::from_m31_array([
            value.a.a.data.into(),
            value.a.b.data.into(),
            value.b.a.data.into(),
            value.b.b.data.into(),
        ])
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

impl ByteSerialize for GpuM31 {}
impl ByteSerialize for GpuCM31 {}
impl ByteSerialize for GpuQM31 {}

pub struct GpuFieldInstance {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub input_buffer: wgpu::Buffer,
    pub output_buffer: wgpu::Buffer,
    pub staging_buffer: wgpu::Buffer,
}

impl GpuFieldInstance {
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
pub enum GpuFieldOperation {
    Add,
    Multiply,
    Subtract,
    Divide,
    Negate,
}

impl GpuFieldOperation {
    fn shader_source(&self) -> String {
        let base_shader = include_str!("qm31.wgsl");

        let inputs = r#"
            struct Inputs {
                first: QM31,
                second: QM31,
            }

            @group(0) @binding(0) var<storage, read> inputs: Inputs;
        "#;

        let output = r#"
            @group(0) @binding(1) var<storage, read_write> output: QM31;
        "#;

        let operation = match self {
            GpuFieldOperation::Add => {
                r#"
                @compute @workgroup_size(1)
                fn main() {
                    output.a = cm31_add(inputs.first.a, inputs.second.a);
                    output.b = cm31_add(inputs.first.b, inputs.second.b);
                }
            "#
            }
            GpuFieldOperation::Multiply => {
                r#"
                @compute @workgroup_size(1)
                fn main() {
                    output = qm31_mul(inputs.first, inputs.second);
                }
            "#
            }
            GpuFieldOperation::Subtract => {
                r#"
                @compute @workgroup_size(1)
                fn main() {
                    output.a = cm31_sub(inputs.first.a, inputs.second.a);
                    output.b = cm31_sub(inputs.first.b, inputs.second.b);
                }
            "#
            }
            GpuFieldOperation::Divide => {
                r#"
                @compute @workgroup_size(1)
                fn main() {
                    let inv_b = qm31_inverse(inputs.second);
                    output = qm31_mul(inputs.first, inv_b);
                }
            "#
            }
            GpuFieldOperation::Negate => {
                r#"
                @compute @workgroup_size(1)
                fn main() {
                    output.a = cm31_neg(inputs.first.a);
                    output.b = cm31_neg(inputs.first.b);
                }
            "#
            }
        };

        format!("{}\n{}\n{}\n{}\n", base_shader, inputs, output, operation)
    }
}

#[derive(Debug)]
pub struct GpuFieldInputs {
    pub first: GpuQM31,
    pub second: GpuQM31,
}

impl ByteSerialize for GpuFieldInputs {}

pub async fn compute_field_operation(a: QM31, b: QM31, operation: GpuFieldOperation) -> QM31 {
    let inputs = GpuFieldInputs {
        first: GpuQM31::from(a),
        second: GpuQM31::from(b),
    };

    let instance = GpuFieldInstance::new(&inputs, std::mem::size_of::<GpuQM31>()).await;

    let shader_source = operation.shader_source();
    let (pipeline, bind_group) = instance.create_pipeline(&shader_source, "main");

    let result = instance
        .run_computation::<GpuQM31>(&pipeline, &bind_group, (1, 1, 1))
        .await;

    QM31(
        CM31(result.a.a.data.into(), result.a.b.data.into()),
        CM31(result.b.a.data.into(), result.b.b.data.into()),
    )
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use super::*;
    use crate::core::fields::m31::{M31, P};
    use crate::core::fields::qm31::QM31;
    use crate::{cm31, qm31};

    #[test]
    fn test_gpu_field_values() {
        let qm0 = qm31!(1, 2, 3, 4);
        let qm1 = qm31!(4, 5, 6, 7);

        // Test round-trip conversion CPU -> GPU -> CPU
        let gpu_qm0 = GpuQM31::from(qm0);
        let gpu_qm1 = GpuQM31::from(qm1);

        let cpu_qm0 = QM31(
            CM31(gpu_qm0.a.a.data.into(), gpu_qm0.a.b.data.into()),
            CM31(gpu_qm0.b.a.data.into(), gpu_qm0.b.b.data.into()),
        );

        let cpu_qm1 = QM31(
            CM31(gpu_qm1.a.a.data.into(), gpu_qm1.a.b.data.into()),
            CM31(gpu_qm1.b.a.data.into(), gpu_qm1.b.b.data.into()),
        );

        assert_eq!(
            qm0, cpu_qm0,
            "Round-trip conversion should preserve values for qm0"
        );
        assert_eq!(
            qm1, cpu_qm1,
            "Round-trip conversion should preserve values for qm1"
        );
    }

    #[test]
    fn test_gpu_m31_field_arithmetic() {
        // Test M31 field operations
        let m = M31::from(19u32);
        let one = M31::from(1u32);
        let zero = M31::zero();

        // Create QM31 values for GPU computation
        let m_qm = QM31(CM31(m, zero), CM31::zero());
        let one_qm = QM31(CM31(one, zero), CM31::zero());
        let zero_qm = QM31(CM31(zero, zero), CM31::zero());

        // Test multiplication
        let cpu_mul = m * one;
        let gpu_mul = pollster::block_on(compute_field_operation(
            m_qm,
            one_qm,
            GpuFieldOperation::Multiply,
        ));
        assert_eq!(gpu_mul.0 .0, cpu_mul, "M31 multiplication failed");

        // Test addition
        let cpu_add = m + one;
        let gpu_add = pollster::block_on(compute_field_operation(
            m_qm,
            one_qm,
            GpuFieldOperation::Add,
        ));
        assert_eq!(gpu_add.0 .0, cpu_add, "M31 addition failed");

        // Test subtraction
        let cpu_sub = m - one;
        let gpu_sub = pollster::block_on(compute_field_operation(
            m_qm,
            one_qm,
            GpuFieldOperation::Subtract,
        ));
        assert_eq!(gpu_sub.0 .0, cpu_sub, "M31 subtraction failed");

        // Test negation
        let cpu_neg = -m;
        let gpu_neg = pollster::block_on(compute_field_operation(
            m_qm,
            zero_qm,
            GpuFieldOperation::Negate,
        ));
        assert_eq!(gpu_neg.0 .0, cpu_neg, "M31 negation failed");

        // Test division and inverse
        let cpu_div = one / m;
        let gpu_div = pollster::block_on(compute_field_operation(
            one_qm,
            m_qm,
            GpuFieldOperation::Divide,
        ));
        assert_eq!(gpu_div.0 .0, cpu_div, "M31 division failed");

        // Test with large numbers (near P)
        let large = M31::from(P - 1);
        let large_qm = QM31(CM31(large, zero), CM31::zero());

        // Test large number multiplication
        let cpu_large_mul = large * m;
        let gpu_large_mul = pollster::block_on(compute_field_operation(
            large_qm,
            m_qm,
            GpuFieldOperation::Multiply,
        ));
        assert_eq!(
            gpu_large_mul.0 .0, cpu_large_mul,
            "M31 large number multiplication failed"
        );

        // Test large number division
        let cpu_large_div = one / large;
        let gpu_large_div = pollster::block_on(compute_field_operation(
            one_qm,
            large_qm,
            GpuFieldOperation::Divide,
        ));
        assert_eq!(
            gpu_large_div.0 .0, cpu_large_div,
            "M31 large number division failed"
        );
    }

    #[test]
    fn test_gpu_cm31_field_arithmetic() {
        let cm0 = cm31!(1, 2);
        let cm1 = cm31!(4, 5);
        let m = M31::from(8u32);
        let cm = CM31::from(m);
        let zero = CM31::zero();

        // Test multiplication
        let cpu_mul = cm0 * cm1;
        let gpu_mul = pollster::block_on(compute_field_operation(
            QM31(cm0, zero),
            QM31(cm1, zero),
            GpuFieldOperation::Multiply,
        ));
        assert_eq!(gpu_mul.0, cpu_mul, "CM31 multiplication failed");

        // Test addition
        let cpu_add = cm0 + cm1;
        let gpu_add = pollster::block_on(compute_field_operation(
            QM31(cm0, zero),
            QM31(cm1, zero),
            GpuFieldOperation::Add,
        ));
        assert_eq!(gpu_add.0, cpu_add, "CM31 addition failed");

        // Test subtraction
        let cpu_sub = cm0 - cm1;
        let gpu_sub = pollster::block_on(compute_field_operation(
            QM31(cm0, zero),
            QM31(cm1, zero),
            GpuFieldOperation::Subtract,
        ));
        assert_eq!(gpu_sub.0, cpu_sub, "CM31 subtraction failed");

        // Test negation
        let cpu_neg = -cm0;
        let gpu_neg = pollster::block_on(compute_field_operation(
            QM31(cm0, zero),
            QM31(zero, zero),
            GpuFieldOperation::Negate,
        ));
        assert_eq!(gpu_neg.0, cpu_neg, "CM31 negation failed");

        // Test division
        let cpu_div = cm0 / cm1;
        let gpu_div = pollster::block_on(compute_field_operation(
            QM31(cm0, zero),
            QM31(cm1, zero),
            GpuFieldOperation::Divide,
        ));
        assert_eq!(gpu_div.0, cpu_div, "CM31 division failed");

        // Test scalar operations
        let cpu_scalar_mul = cm1 * m;
        let gpu_scalar_mul = pollster::block_on(compute_field_operation(
            QM31(cm1, zero),
            QM31(cm, zero),
            GpuFieldOperation::Multiply,
        ));
        assert_eq!(
            gpu_scalar_mul.0, cpu_scalar_mul,
            "CM31 scalar multiplication failed"
        );

        let cpu_scalar_add = cm1 + m;
        let gpu_scalar_add = pollster::block_on(compute_field_operation(
            QM31(cm1, zero),
            QM31(cm, zero),
            GpuFieldOperation::Add,
        ));
        assert_eq!(
            gpu_scalar_add.0, cpu_scalar_add,
            "CM31 scalar addition failed"
        );

        let cpu_scalar_sub = cm1 - m;
        let gpu_scalar_sub = pollster::block_on(compute_field_operation(
            QM31(cm1, zero),
            QM31(cm, zero),
            GpuFieldOperation::Subtract,
        ));
        assert_eq!(
            gpu_scalar_sub.0, cpu_scalar_sub,
            "CM31 scalar subtraction failed"
        );

        let cpu_scalar_div = cm1 / m;
        let gpu_scalar_div = pollster::block_on(compute_field_operation(
            QM31(cm1, zero),
            QM31(cm, zero),
            GpuFieldOperation::Divide,
        ));
        assert_eq!(
            gpu_scalar_div.0, cpu_scalar_div,
            "CM31 scalar division failed"
        );

        // Test with large numbers (near P)
        let large = cm31!(P - 1, P - 2);
        let large_qm = QM31(large, zero);

        // Test large number multiplication
        let cpu_large_mul = large * cm1;
        let gpu_large_mul = pollster::block_on(compute_field_operation(
            large_qm,
            QM31(cm1, zero),
            GpuFieldOperation::Multiply,
        ));
        assert_eq!(
            gpu_large_mul.0, cpu_large_mul,
            "CM31 large number multiplication failed"
        );

        // Test large number division
        let cpu_large_div = large / cm1;
        let gpu_large_div = pollster::block_on(compute_field_operation(
            large_qm,
            QM31(cm1, zero),
            GpuFieldOperation::Divide,
        ));
        assert_eq!(
            gpu_large_div.0, cpu_large_div,
            "CM31 large number division failed"
        );
    }

    #[test]
    fn test_gpu_qm31_field_arithmetic() {
        let qm0 = qm31!(1, 2, 3, 4);
        let qm1 = qm31!(4, 5, 6, 7);
        let m = M31::from(8u32);
        let qm = QM31::from(m);
        let zero = QM31::zero();

        // Test multiplication
        let cpu_mul = qm0 * qm1;
        let gpu_mul = pollster::block_on(compute_field_operation(
            qm0,
            qm1,
            GpuFieldOperation::Multiply,
        ));
        assert_eq!(gpu_mul, cpu_mul, "QM31 multiplication failed");

        // Test addition
        let cpu_add = qm0 + qm1;
        let gpu_add = pollster::block_on(compute_field_operation(qm0, qm1, GpuFieldOperation::Add));
        assert_eq!(gpu_add, cpu_add, "QM31 addition failed");

        // Test subtraction
        let cpu_sub = qm0 - qm1;
        let gpu_sub = pollster::block_on(compute_field_operation(
            qm0,
            qm1,
            GpuFieldOperation::Subtract,
        ));
        assert_eq!(gpu_sub, cpu_sub, "QM31 subtraction failed");

        // Test negation
        let cpu_neg = -qm0;
        let gpu_neg = pollster::block_on(compute_field_operation(
            qm0,
            zero,
            GpuFieldOperation::Negate,
        ));
        assert_eq!(gpu_neg, cpu_neg, "QM31 negation failed");

        // Test division
        let cpu_div = qm0 / qm1;
        let gpu_div =
            pollster::block_on(compute_field_operation(qm0, qm1, GpuFieldOperation::Divide));
        assert_eq!(gpu_div, cpu_div, "QM31 division failed");

        // Test scalar operations
        let cpu_scalar_mul = qm1 * m;
        let gpu_scalar_mul = pollster::block_on(compute_field_operation(
            qm1,
            qm,
            GpuFieldOperation::Multiply,
        ));
        assert_eq!(
            cpu_scalar_mul, gpu_scalar_mul,
            "QM31 scalar multiplication failed"
        );

        let cpu_scalar_add = qm1 + m;
        let gpu_scalar_add =
            pollster::block_on(compute_field_operation(qm1, qm, GpuFieldOperation::Add));
        assert_eq!(
            cpu_scalar_add, gpu_scalar_add,
            "QM31 scalar addition failed"
        );

        let cpu_scalar_sub = qm1 - m;
        let gpu_scalar_sub = pollster::block_on(compute_field_operation(
            qm1,
            qm,
            GpuFieldOperation::Subtract,
        ));
        assert_eq!(
            cpu_scalar_sub, gpu_scalar_sub,
            "QM31 scalar subtraction failed"
        );

        let cpu_scalar_div = qm1 / m;
        let gpu_scalar_div =
            pollster::block_on(compute_field_operation(qm1, qm, GpuFieldOperation::Divide));
        assert_eq!(
            cpu_scalar_div, gpu_scalar_div,
            "QM31 scalar division failed"
        );

        // Test with large numbers (near P)
        let large = qm31!(P - 1, P - 2, P - 3, P - 4);

        // Test large number multiplication
        let cpu_large_mul = large * qm1;
        let gpu_large_mul = pollster::block_on(compute_field_operation(
            large,
            qm1,
            GpuFieldOperation::Multiply,
        ));
        assert_eq!(
            gpu_large_mul, cpu_large_mul,
            "QM31 large number multiplication failed"
        );

        // Test large number division
        let cpu_large_div = large / qm1;
        let gpu_large_div = pollster::block_on(compute_field_operation(
            large,
            qm1,
            GpuFieldOperation::Divide,
        ));
        assert_eq!(
            gpu_large_div, cpu_large_div,
            "QM31 large number division failed"
        );
    }
}
