mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
}

pub mod ffi;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Id {
    Affine,
    Binary,
    Cast,
    Conv,
    DeltaNet,
    Fill,
    FlashAttnQ8, // New: Q8_0 quantized flash attention
    Indexing,
    Quantized,
    QuzoFused,   // QuZO fused perturbation+matmul kernels
    QuzoPerturb, // QuZO in-place perturbation kernels
    Reduce,
    Sort,
    Ternary,
    Unary,
}

pub const ALL_IDS: [Id; 15] = [
    Id::Affine,
    Id::Binary,
    Id::Cast,
    Id::Conv,
    Id::DeltaNet,
    Id::Fill,
    Id::FlashAttnQ8,
    Id::Indexing,
    Id::Quantized,
    Id::QuzoFused,
    Id::QuzoPerturb,
    Id::Reduce,
    Id::Sort,
    Id::Ternary,
    Id::Unary,
];

pub struct Module {
    index: usize,
    ptx: &'static str,
}

impl Module {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn ptx(&self) -> &'static str {
        self.ptx
    }
}

const fn module_index(id: Id) -> usize {
    let mut i = 0;
    while i < ALL_IDS.len() {
        if ALL_IDS[i] as u32 == id as u32 {
            return i;
        }
        i += 1;
    }
    panic!("id not found")
}

macro_rules! mdl {
    ($cst:ident, $id:ident) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            ptx: ptx::$cst,
        };
    };
}

mdl!(AFFINE, Affine);
mdl!(BINARY, Binary);
mdl!(CAST, Cast);
mdl!(CONV, Conv);
mdl!(DELTANET, DeltaNet);
mdl!(FILL, Fill);
mdl!(FLASH_ATTN_Q8, FlashAttnQ8);
mdl!(INDEXING, Indexing);
mdl!(QUANTIZED, Quantized);
mdl!(QUZO_FUSED, QuzoFused);
mdl!(QUZO_PERTURB, QuzoPerturb);
mdl!(REDUCE, Reduce);
mdl!(SORT, Sort);
mdl!(TERNARY, Ternary);
mdl!(UNARY, Unary);
