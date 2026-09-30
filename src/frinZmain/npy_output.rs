//! Compressed self-describing NumPy NPZ sidecars for analysis and plot data.
use std::fs::File;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use zip::{write::SimpleFileOptions, CompressionMethod, ZipWriter};

use crate::output::insert_product_before_processing_suffixes;

const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";
const FORMAT_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy)]
pub struct NpyMeta<'a> {
    pub flag: &'a str,
    pub fft_point: u32,
    pub pp: u32,
    pub axis0_name: &'a str,
    pub axis0_unit: &'a str,
    pub axis1_name: &'a str,
    pub axis1_unit: &'a str,
}

impl<'a> NpyMeta<'a> {
    pub fn new(flag: &'a str, fft_point: u32, pp: u32) -> Self {
        Self {
            flag,
            fft_point,
            pp,
            axis0_name: "index",
            axis0_unit: "",
            axis1_name: "",
            axis1_unit: "",
        }
    }
    #[allow(dead_code)]
    pub fn axes(mut self, n0: &'a str, u0: &'a str, n1: &'a str, u1: &'a str) -> Self {
        self.axis0_name = n0;
        self.axis0_unit = u0;
        self.axis1_name = n1;
        self.axis1_unit = u1;
        self
    }
}

pub fn npz_sidecar_path(output_path: &Path, flag: &str) -> PathBuf {
    let parent = output_path.parent().unwrap_or_else(|| Path::new(""));
    let stem = output_path
        .file_stem()
        .and_then(|v| v.to_str())
        .unwrap_or("analysis");
    let flag = flag.trim_start_matches('-').replace('-', "_");
    let output_stem = insert_product_before_processing_suffixes(stem, &flag);
    parent.join(format!("{output_stem}.npz"))
}

#[allow(dead_code)]
pub fn write_complex_1d(
    path: &Path,
    meta: NpyMeta<'_>,
    values: &[num_complex::Complex<f32>],
    axis0: &[f64],
) -> io::Result<()> {
    write_complex_nd(
        path,
        meta,
        &[values.len()],
        values.iter().copied(),
        axis0,
        &[],
    )
}
#[allow(dead_code)]
pub fn write_real_1d(
    path: &Path,
    meta: NpyMeta<'_>,
    values: &[f32],
    axis0: &[f64],
) -> io::Result<()> {
    write_complex_nd(
        path,
        meta,
        &[values.len()],
        values.iter().map(|&re| num_complex::Complex::new(re, 0.0)),
        axis0,
        &[],
    )
}
#[allow(dead_code)]
pub fn write_complex_2d(
    path: &Path,
    meta: NpyMeta<'_>,
    shape: (usize, usize),
    values: impl IntoIterator<Item = num_complex::Complex<f32>>,
    axis0: &[f64],
    axis1: &[f64],
) -> io::Result<()> {
    write_complex_nd(path, meta, &[shape.0, shape.1], values, axis0, axis1)
}
#[allow(dead_code)]
pub fn write_real_2d(
    path: &Path,
    meta: NpyMeta<'_>,
    shape: (usize, usize),
    values: impl IntoIterator<Item = f32>,
    axis0: &[f64],
    axis1: &[f64],
) -> io::Result<()> {
    write_complex_nd(
        path,
        meta,
        &[shape.0, shape.1],
        values
            .into_iter()
            .map(|re| num_complex::Complex::new(re, 0.0)),
        axis0,
        axis1,
    )
}

/// Stream compressed entries to a temporary archive rather than retaining arrays.
/// Metadata methods defer I/O errors until write; array methods report them immediately.
#[allow(dead_code)]
pub struct NamedNpz {
    writer: Option<ZipWriter<BufWriter<File>>>,
    error: Option<io::Error>,
}
#[allow(dead_code)]
impl NamedNpz {
    pub fn new(meta: NpyMeta<'_>) -> Self {
        let mut output = match tempfile::tempfile() {
            Ok(file) => Self {
                writer: Some(ZipWriter::new(BufWriter::with_capacity(64 * 1024, file))),
                error: None,
            },
            Err(error) => Self {
                writer: None,
                error: Some(error),
            },
        };
        output.add_u8_1d("flag", meta.flag.as_bytes());
        for (name, value) in [
            ("fft_point", meta.fft_point),
            ("pp", meta.pp),
            ("format_version", FORMAT_VERSION),
        ] {
            let _ = output.add_array(name, "<u4", &[1], 4, value.to_le_bytes());
        }
        output
    }
    fn add_array(
        &mut self,
        name: &str,
        descr: &str,
        shape: &[usize],
        item_size: usize,
        bytes: impl IntoIterator<Item = u8>,
    ) -> io::Result<()> {
        if let Some(error) = &self.error {
            return Err(io::Error::new(error.kind(), error.to_string()));
        }
        let result = (|| {
            let byte_count = shape.iter().try_fold(item_size, |count, &dim| {
                count.checked_mul(dim).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "NPY shape overflow")
                })
            })?;
            let header = make_npy(descr, shape, &[]);
            let entry_size = byte_count.checked_add(header.len()).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "NPY entry size overflow")
            })?;
            let writer = self.writer.as_mut().expect("NPZ writer initialized");
            writer.start_file(format!("{name}.npy"), npz_options(entry_size))?;
            writer.write_all(&header)?;
            let mut buffer = [0u8; 64 * 1024];
            let mut used = 0;
            let mut count = 0usize;
            for byte in bytes {
                if count == byte_count {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "NPY data length exceeds shape",
                    ));
                }
                buffer[used] = byte;
                used += 1;
                count += 1;
                if used == buffer.len() {
                    writer.write_all(&buffer)?;
                    used = 0;
                }
            }
            if count != byte_count {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("NPY data length {count} bytes does not match {byte_count}"),
                ));
            }
            writer.write_all(&buffer[..used])
        })();
        if let Err(error) = &result {
            self.error = Some(io::Error::new(error.kind(), error.to_string()));
        }
        result
    }
    pub fn add_f64_1d(&mut self, name: &str, values: &[f64]) {
        let _ = self.add_array(
            name,
            "<f8",
            &[values.len()],
            8,
            values.iter().flat_map(|value| value.to_le_bytes()),
        );
    }
    pub fn add_f32_1d(&mut self, name: &str, values: &[f32]) {
        let _ = self.add_array(
            name,
            "<f4",
            &[values.len()],
            4,
            values.iter().flat_map(|value| value.to_le_bytes()),
        );
    }
    pub fn add_u8_1d(&mut self, name: &str, values: &[u8]) {
        let _ = self.add_array(name, "|u1", &[values.len()], 1, values.iter().copied());
    }
    pub fn add_u8_2d(
        &mut self,
        name: &str,
        shape: (usize, usize),
        values: impl IntoIterator<Item = u8>,
    ) -> io::Result<()> {
        self.add_array(name, "|u1", &[shape.0, shape.1], 1, values)
    }
    pub fn add_complex64_1d(&mut self, name: &str, values: &[num_complex::Complex<f32>]) {
        let _ = self.add_complex64_inner(name, &[values.len()], values.iter().copied());
    }
    pub fn add_f32_2d(
        &mut self,
        name: &str,
        shape: (usize, usize),
        values: impl IntoIterator<Item = f32>,
    ) -> io::Result<()> {
        self.add_array(
            name,
            "<f4",
            &[shape.0, shape.1],
            4,
            values.into_iter().flat_map(|value| value.to_le_bytes()),
        )
    }
    pub fn add_complex64_2d(
        &mut self,
        name: &str,
        shape: (usize, usize),
        values: impl IntoIterator<Item = num_complex::Complex<f32>>,
    ) -> io::Result<()> {
        self.add_complex64_inner(name, &[shape.0, shape.1], values)
    }
    fn add_complex64_inner(
        &mut self,
        name: &str,
        shape: &[usize],
        values: impl IntoIterator<Item = num_complex::Complex<f32>>,
    ) -> io::Result<()> {
        self.add_array(
            name,
            "<c8",
            shape,
            8,
            values.into_iter().flat_map(|value| {
                let mut bytes = [0u8; 8];
                bytes[..4].copy_from_slice(&value.re.to_le_bytes());
                bytes[4..].copy_from_slice(&value.im.to_le_bytes());
                bytes
            }),
        )
    }
    pub fn write(mut self, path: &Path) -> io::Result<()> {
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        let mut source = self
            .writer
            .take()
            .expect("NPZ writer initialized")
            .finish()?;
        source.flush()?;
        source.seek(SeekFrom::Start(0))?;
        let mut destination = BufWriter::with_capacity(64 * 1024, File::create(path)?);
        io::copy(source.get_mut(), &mut destination)?;
        destination.flush()
    }
}
fn npz_options(entry_size: usize) -> SimpleFileOptions {
    SimpleFileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        // Reserve ZIP64 before compression can push a near-limit entry past 4 GiB.
        .compression_level(Some(9))
        .large_file(entry_size >= u32::MAX as usize / 2)
}

pub fn write_named_real_1d_npz(
    path: &Path,
    meta: NpyMeta<'_>,
    series: &[(&str, &[f64])],
) -> io::Result<()> {
    let mut output = NamedNpz::new(meta);
    for (name, values) in series {
        output.add_f64_1d(name, values);
    }
    output.add_array(
        "series_count",
        "<u4",
        &[1],
        4,
        (series.len() as u32).to_le_bytes(),
    )?;
    output.write(path)
}

fn write_complex_nd(
    path: &Path,
    meta: NpyMeta<'_>,
    shape: &[usize],
    values: impl IntoIterator<Item = num_complex::Complex<f32>>,
    axis0: &[f64],
    axis1: &[f64],
) -> io::Result<()> {
    if shape.is_empty() || shape.len() > 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "NPZ sidecars support rank 1 or 2",
        ));
    }
    if (!axis0.is_empty() && axis0.len() != shape[0])
        || (shape.len() == 2 && !axis1.is_empty() && axis1.len() != shape[1])
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "axis length mismatch",
        ));
    }
    let mut output = NamedNpz::new(meta);
    output.add_complex64_inner("data", shape, values)?;
    output.add_f64_1d("axis0", axis0);
    output.add_f64_1d("axis1", axis1);
    output.add_u8_1d("axis0_name", meta.axis0_name.as_bytes());
    output.add_u8_1d("axis0_unit", meta.axis0_unit.as_bytes());
    output.add_u8_1d("axis1_name", meta.axis1_name.as_bytes());
    output.add_u8_1d("axis1_unit", meta.axis1_unit.as_bytes());
    let shape_values = [shape[0] as u32, shape.get(1).copied().unwrap_or(0) as u32];
    output.add_array(
        "shape",
        "<u4",
        &[2],
        4,
        shape_values.iter().flat_map(|value| value.to_le_bytes()),
    )?;
    output.write(path)
}

fn make_npy(descr: &str, shape: &[usize], payload: &[u8]) -> Vec<u8> {
    let shape_descr = tuple_descr(shape);
    let mut header = format!(
        "{{\x27descr\x27: \x27{descr}\x27, \x27fortran_order\x27: False, \x27shape\x27: {shape_descr}, }}"
    ).into_bytes();
    let padding = (64 - ((12 + header.len() + 1) % 64)) % 64;
    header.extend(std::iter::repeat_n(b' ', padding));
    header.push(b'\n');
    let mut output = Vec::with_capacity(12 + header.len() + payload.len());
    output.extend_from_slice(NPY_MAGIC);
    output.extend_from_slice(&[2, 0]);
    output.extend_from_slice(&(header.len() as u32).to_le_bytes());
    output.extend_from_slice(&header);
    output.extend_from_slice(payload);
    output
}

fn tuple_descr(shape: &[usize]) -> String {
    match shape {
        [a] => format!("({a},)"),
        [a, b] => format!("({a},{b})"),
        _ => "()".into(),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sidecar_name() {
        assert_eq!(
            npz_sidecar_path(Path::new("/tmp/x_bptable.bin"), "bptable"),
            PathBuf::from("/tmp/x_bptable.npz")
        );
        assert_eq!(
            npz_sidecar_path(
                Path::new("/tmp/x_delay_rate_search_bp_rfi_contamisubt.png"),
                "plot_delay_rate",
            ),
            PathBuf::from("/tmp/x_delay_rate_search_plot_delay_rate_bp_rfi_contamisubt.npz")
        );
    }
    #[test]
    fn writes_magic() {
        let p = std::env::temp_dir().join(format!("frinz_npy_test_{}.npz", std::process::id()));
        write_real_1d(
            &p,
            NpyMeta::new("test", 16, 2).axes("frequency", "Hz", "", ""),
            &[1.0, 2.0],
            &[10.0, 20.0],
        )
        .unwrap();
        let b = std::fs::read(&p).unwrap();
        assert_eq!(&b[..4], b"PK\x03\x04");
        let _ = std::fs::remove_file(p);
    }
}
