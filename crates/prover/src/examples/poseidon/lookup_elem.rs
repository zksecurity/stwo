use super::PoseidonElements;
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::QM31;

/// lookup parameters for a given domain-log-size
struct PoseidonLookupConfig {
    z: [M31; 4],
    alpha: [M31; 4],
}

impl PoseidonLookupConfig {
    const LOG9: Self = Self {
        z: [
            M31::from_u32_unchecked(1620680704),
            M31::from_u32_unchecked(1901317872),
            M31::from_u32_unchecked(913853993),
            M31::from_u32_unchecked(1286799353),
        ],
        alpha: [
            M31::from_u32_unchecked(2011422255),
            M31::from_u32_unchecked(1962282213),
            M31::from_u32_unchecked(69078916),
            M31::from_u32_unchecked(407074834),
        ],
    };
    const LOG10: Self = Self {
        z: [
            M31::from_u32_unchecked(1465862614),
            M31::from_u32_unchecked(1583357442),
            M31::from_u32_unchecked(1715957657),
            M31::from_u32_unchecked(977402081),
        ],
        alpha: [
            M31::from_u32_unchecked(1058568156),
            M31::from_u32_unchecked(1376697150),
            M31::from_u32_unchecked(1770783003),
            M31::from_u32_unchecked(1982948122),
        ],
    };
    const LOG11: Self = Self {
        z: [
            M31::from_u32_unchecked(589075703),
            M31::from_u32_unchecked(149359250),
            M31::from_u32_unchecked(1907284710),
            M31::from_u32_unchecked(729671227),
        ],
        alpha: [
            M31::from_u32_unchecked(318198925),
            M31::from_u32_unchecked(1203679427),
            M31::from_u32_unchecked(870875217),
            M31::from_u32_unchecked(1185640677),
        ],
    };
    const LOG12: Self = Self {
        z: [
            M31::from_u32_unchecked(1628655791),
            M31::from_u32_unchecked(1055381932),
            M31::from_u32_unchecked(980792236),
            M31::from_u32_unchecked(1563574579),
        ],
        alpha: [
            M31::from_u32_unchecked(758947366),
            M31::from_u32_unchecked(782855802),
            M31::from_u32_unchecked(792359994),
            M31::from_u32_unchecked(1161959256),
        ],
    };
    const LOG13: Self = Self {
        z: [
            M31::from_u32_unchecked(668979421),
            M31::from_u32_unchecked(2097978502),
            M31::from_u32_unchecked(428317414),
            M31::from_u32_unchecked(1503540921),
        ],
        alpha: [
            M31::from_u32_unchecked(962480916),
            M31::from_u32_unchecked(462545530),
            M31::from_u32_unchecked(118859601),
            M31::from_u32_unchecked(1868751663),
        ],
    };
    const LOG14: Self = Self {
        z: [
            M31::from_u32_unchecked(1185288908),
            M31::from_u32_unchecked(1548569092),
            M31::from_u32_unchecked(792634712),
            M31::from_u32_unchecked(779398798),
        ],
        alpha: [
            M31::from_u32_unchecked(138774446),
            M31::from_u32_unchecked(799972521),
            M31::from_u32_unchecked(2070047733),
            M31::from_u32_unchecked(2053058841),
        ],
    };

    fn for_log_size(log_size: u32) -> Self {
        match log_size {
            9 => Self::LOG9,
            10 => Self::LOG10,
            11 => Self::LOG11,
            12 => Self::LOG12,
            13 => Self::LOG13,
            14 => Self::LOG14,
            _ => panic!("unsupported eval_domain_log_size: {}", log_size),
        }
    }
}

impl PoseidonElements {
    pub fn with_lookup(log_size: u32) -> Self {
        let cfg = PoseidonLookupConfig::for_log_size(log_size);
        let mut elems = PoseidonElements::dummy();

        elems.0.z = QM31::from_m31_array(cfg.z);
        elems.0.alpha = QM31::from_m31_array(cfg.alpha);

        let mut cur = QM31::from(1);
        elems.0.alpha_powers = std::array::from_fn(|_| {
            let res = cur;
            cur *= elems.0.alpha;
            res
        });

        elems
    }
}
