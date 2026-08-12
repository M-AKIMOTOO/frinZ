#![allow(non_snake_case)]

pub mod analysis;
pub mod api;
pub mod args;
pub mod bandpass;
pub mod bispectrum;
pub mod contamination;
pub mod contamination_subtract;
pub mod fft;
pub mod fitting;
pub mod folding;
pub mod frmap;
pub mod header;
pub mod inband;
pub mod input_support;
pub mod maser;
pub mod search;
pub mod spike34m;
pub mod stfft;

pub mod earth_rotation_imaging;
pub mod multisideband;
pub mod norm_acf;
pub mod npy_output;
pub mod output;
pub mod phsref;
pub mod plot;
pub mod png_compress;
pub mod processing;
pub mod raw_visibility;
pub mod read;
pub mod rfi;
pub mod uptimeplot;
pub mod utils;
pub mod uv;
pub mod wwz;

pub use api::{
    apply_delay_rate_correction, delay_search, frequency_spectrum, fringe_search, read_bandpass,
    read_cor, read_cor_with_options, search_cor_bytes, FrequencySpectrum, FringeSearchOutput,
    LibraryOptions, PhaseCorrection, SearchMode,
};
pub use header::CorHeader;
pub use read::{
    read_cor_bytes, read_cor_file, read_cor_file_with_options, CorData, CorReadOptions,
};
