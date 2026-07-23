//! Compressed self-describing NumPy NPZ sidecars for analysis and plot data.
use flate2::write::DeflateEncoder;
use flate2::Compression;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

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
    if stem.ends_with(&flag) {
        parent.join(format!("{stem}.npz"))
    } else {
        parent.join(format!("{stem}_{flag}.npz"))
    }
}

#[allow(dead_code)]
pub fn write_complex_1d(
    path: &Path,
    meta: NpyMeta<'_>,
    values: &[num_complex::Complex<f32>],
    axis0: &[f64],
) -> io::Result<()> {
    write_npy(
        path,
        meta,
        &[values.len()],
        values.iter().map(|v| v.re),
        values.iter().map(|v| v.im),
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
    write_npy(
        path,
        meta,
        &[values.len()],
        values.iter().copied(),
        std::iter::repeat(0.0).take(values.len()),
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
    let values: Vec<_> = values.into_iter().collect();
    validate_len(values.len(), shape)?;
    write_npy(
        path,
        meta,
        &[shape.0, shape.1],
        values.iter().map(|v| v.re),
        values.iter().map(|v| v.im),
        axis0,
        axis1,
    )
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
    let values: Vec<_> = values.into_iter().collect();
    validate_len(values.len(), shape)?;
    write_npy(
        path,
        meta,
        &[shape.0, shape.1],
        values.iter().copied(),
        std::iter::repeat(0.0).take(values.len()),
        axis0,
        axis1,
    )
}

#[allow(dead_code)]
pub struct NamedNpz {
    entries: Vec<(String, Vec<u8>)>,
}

#[allow(dead_code)]
impl NamedNpz {
    pub fn new(meta: NpyMeta<'_>) -> Self {
        let u32_payload = |values: &[u32]| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>()
        };
        let text_entry = |name: &str, value: &str| {
            (
                name.to_string(),
                make_npy("|u1", &[value.len()], value.as_bytes()),
            )
        };
        Self {
            entries: vec![
                text_entry("flag.npy", meta.flag),
                (
                    "fft_point.npy".to_string(),
                    make_npy("<u4", &[1], &u32_payload(&[meta.fft_point])),
                ),
                (
                    "pp.npy".to_string(),
                    make_npy("<u4", &[1], &u32_payload(&[meta.pp])),
                ),
                (
                    "format_version.npy".to_string(),
                    make_npy("<u4", &[1], &u32_payload(&[FORMAT_VERSION])),
                ),
            ],
        }
    }

    pub fn add_f64_1d(&mut self, name: &str, values: &[f64]) {
        let payload = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>();
        self.entries.push((
            format!("{name}.npy"),
            make_npy("<f8", &[values.len()], &payload),
        ));
    }

    pub fn add_f32_1d(&mut self, name: &str, values: &[f32]) {
        let payload = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>();
        self.entries.push((
            format!("{name}.npy"),
            make_npy("<f4", &[values.len()], &payload),
        ));
    }

    pub fn add_u8_1d(&mut self, name: &str, values: &[u8]) {
        self.entries.push((
            format!("{name}.npy"),
            make_npy("|u1", &[values.len()], values),
        ));
    }

    pub fn add_complex64_1d(&mut self, name: &str, values: &[num_complex::Complex<f32>]) {
        let mut payload = Vec::with_capacity(values.len() * 8);
        for value in values {
            payload.extend_from_slice(&value.re.to_le_bytes());
            payload.extend_from_slice(&value.im.to_le_bytes());
        }
        self.entries.push((
            format!("{name}.npy"),
            make_npy("<c8", &[values.len()], &payload),
        ));
    }

    pub fn add_f32_2d(
        &mut self,
        name: &str,
        shape: (usize, usize),
        values: impl IntoIterator<Item = f32>,
    ) -> io::Result<()> {
        let values: Vec<f32> = values.into_iter().collect();
        validate_len(values.len(), shape)?;
        let payload = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>();
        self.entries.push((
            format!("{name}.npy"),
            make_npy("<f4", &[shape.0, shape.1], &payload),
        ));
        Ok(())
    }

    pub fn add_complex64_2d(
        &mut self,
        name: &str,
        shape: (usize, usize),
        values: impl IntoIterator<Item = num_complex::Complex<f32>>,
    ) -> io::Result<()> {
        let values: Vec<num_complex::Complex<f32>> = values.into_iter().collect();
        validate_len(values.len(), shape)?;
        let mut payload = Vec::with_capacity(values.len() * 8);
        for value in values {
            payload.extend_from_slice(&value.re.to_le_bytes());
            payload.extend_from_slice(&value.im.to_le_bytes());
        }
        self.entries.push((
            format!("{name}.npy"),
            make_npy("<c8", &[shape.0, shape.1], &payload),
        ));
        Ok(())
    }

    pub fn write(self, path: &Path) -> io::Result<()> {
        write_npz(path, &self.entries)
    }
}

pub fn write_named_real_1d_npz(
    path: &Path,
    meta: NpyMeta<'_>,
    series: &[(&str, &[f64])],
) -> io::Result<()> {
    let u32_payload = |values: &[u32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>()
    };
    let text_entry = |name: &str, value: &str| {
        (
            name.to_string(),
            make_npy("|u1", &[value.len()], value.as_bytes()),
        )
    };
    let mut entries: Vec<(String, Vec<u8>)> = Vec::with_capacity(series.len() + 5);
    for (name, values) in series {
        let payload = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>();
        entries.push((
            format!("{name}.npy"),
            make_npy("<f8", &[values.len()], &payload),
        ));
    }
    entries.push(text_entry("flag.npy", meta.flag));
    entries.push((
        "fft_point.npy".to_string(),
        make_npy("<u4", &[1], &u32_payload(&[meta.fft_point])),
    ));
    entries.push((
        "pp.npy".to_string(),
        make_npy("<u4", &[1], &u32_payload(&[meta.pp])),
    ));
    entries.push((
        "format_version.npy".to_string(),
        make_npy("<u4", &[1], &u32_payload(&[FORMAT_VERSION])),
    ));
    entries.push((
        "series_count.npy".to_string(),
        make_npy("<u4", &[1], &u32_payload(&[series.len() as u32])),
    ));
    write_npz(path, &entries)
}

fn validate_len(len: usize, shape: (usize, usize)) -> io::Result<()> {
    if len != shape.0.saturating_mul(shape.1) {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("npy data length {len} does not match shape {shape:?}"),
        ))
    } else {
        Ok(())
    }
}

fn write_npy(
    path: &Path,
    meta: NpyMeta<'_>,
    shape: &[usize],
    real: impl IntoIterator<Item = f32>,
    imag: impl IntoIterator<Item = f32>,
    axis0: &[f64],
    axis1: &[f64],
) -> io::Result<()> {
    if shape.is_empty() || shape.len() > 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "NPZ sidecars support rank 1 or 2",
        ));
    }
    let element_count = shape.iter().product::<usize>();
    let real: Vec<f32> = real.into_iter().collect();
    let imag: Vec<f32> = imag.into_iter().collect();
    if real.len() != element_count || imag.len() != element_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "complex data length mismatch",
        ));
    }
    if !axis0.is_empty() && axis0.len() != shape[0] {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "axis0 length mismatch",
        ));
    }
    if shape.len() == 2 && !axis1.is_empty() && axis1.len() != shape[1] {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "axis1 length mismatch",
        ));
    }

    let mut complex_payload = Vec::with_capacity(element_count * 8);
    for (re, im) in real.iter().zip(imag.iter()) {
        complex_payload.extend_from_slice(&re.to_le_bytes());
        complex_payload.extend_from_slice(&im.to_le_bytes());
    }
    let f64_payload = |values: &[f64]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>()
    };
    let u32_payload = |values: &[u32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>()
    };
    let text_entry = |name: &str, value: &str| {
        (
            name.to_string(),
            make_npy("|u1", &[value.len()], value.as_bytes()),
        )
    };
    let shape_values = [shape[0] as u32, shape.get(1).copied().unwrap_or(0) as u32];
    let entries = vec![
        (
            "data.npy".to_string(),
            make_npy("<c8", shape, &complex_payload),
        ),
        (
            "axis0.npy".to_string(),
            make_npy("<f8", &[axis0.len()], &f64_payload(axis0)),
        ),
        (
            "axis1.npy".to_string(),
            make_npy("<f8", &[axis1.len()], &f64_payload(axis1)),
        ),
        text_entry("flag.npy", meta.flag),
        text_entry("axis0_name.npy", meta.axis0_name),
        text_entry("axis0_unit.npy", meta.axis0_unit),
        text_entry("axis1_name.npy", meta.axis1_name),
        text_entry("axis1_unit.npy", meta.axis1_unit),
        (
            "fft_point.npy".to_string(),
            make_npy("<u4", &[1], &u32_payload(&[meta.fft_point])),
        ),
        (
            "pp.npy".to_string(),
            make_npy("<u4", &[1], &u32_payload(&[meta.pp])),
        ),
        (
            "format_version.npy".to_string(),
            make_npy("<u4", &[1], &u32_payload(&[FORMAT_VERSION])),
        ),
        (
            "shape.npy".to_string(),
            make_npy("<u4", &[2], &u32_payload(&shape_values)),
        ),
    ];
    write_npz(path, &entries)
}

fn make_npy(descr: &str, shape: &[usize], payload: &[u8]) -> Vec<u8> {
    let shape_descr = tuple_descr(shape);
    let mut header = format!(
        "{{\x27descr\x27: \x27{descr}\x27, \x27fortran_order\x27: False, \x27shape\x27: {shape_descr}, }}"
    ).into_bytes();
    let padding = (64 - ((12 + header.len() + 1) % 64)) % 64;
    header.extend(std::iter::repeat(b' ').take(padding));
    header.push(b'\n');
    let mut output = Vec::with_capacity(12 + header.len() + payload.len());
    output.extend_from_slice(NPY_MAGIC);
    output.extend_from_slice(&[2, 0]);
    output.extend_from_slice(&(header.len() as u32).to_le_bytes());
    output.extend_from_slice(&header);
    output.extend_from_slice(payload);
    output
}

fn write_npz(path: &Path, entries: &[(String, Vec<u8>)]) -> io::Result<()> {
    struct CentralEntry {
        name: String,
        crc: u32,
        compressed: u32,
        uncompressed: u32,
        offset: u32,
    }
    let mut out = BufWriter::new(File::create(path)?);
    let mut central = Vec::with_capacity(entries.len());
    let mut offset = 0_u32;
    for (name, npy) in entries {
        let name_bytes = name.as_bytes();
        let uncompressed = u32::try_from(npy.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "NPZ entry exceeds ZIP32 size limit",
            )
        })?;
        let crc = crc32fast::hash(npy);
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(npy)?;
        let compressed_data = encoder.finish()?;
        let compressed = u32::try_from(compressed_data.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "compressed NPZ entry exceeds ZIP32 size limit",
            )
        })?;
        out.write_all(&0x04034b50_u32.to_le_bytes())?;
        out.write_all(&20_u16.to_le_bytes())?;
        out.write_all(&0_u16.to_le_bytes())?;
        out.write_all(&8_u16.to_le_bytes())?;
        out.write_all(&0_u16.to_le_bytes())?;
        out.write_all(&0_u16.to_le_bytes())?;
        out.write_all(&crc.to_le_bytes())?;
        out.write_all(&compressed.to_le_bytes())?;
        out.write_all(&uncompressed.to_le_bytes())?;
        out.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
        out.write_all(&0_u16.to_le_bytes())?;
        out.write_all(name_bytes)?;
        out.write_all(&compressed_data)?;
        central.push(CentralEntry {
            name: name.clone(),
            crc,
            compressed,
            uncompressed,
            offset,
        });
        offset += 30 + name_bytes.len() as u32 + compressed;
    }
    let central_offset = offset;
    for entry in &central {
        let name = entry.name.as_bytes();
        out.write_all(&0x02014b50_u32.to_le_bytes())?;
        out.write_all(&20_u16.to_le_bytes())?;
        out.write_all(&20_u16.to_le_bytes())?;
        out.write_all(&0_u16.to_le_bytes())?;
        out.write_all(&8_u16.to_le_bytes())?;
        out.write_all(&0_u16.to_le_bytes())?;
        out.write_all(&0_u16.to_le_bytes())?;
        out.write_all(&entry.crc.to_le_bytes())?;
        out.write_all(&entry.compressed.to_le_bytes())?;
        out.write_all(&entry.uncompressed.to_le_bytes())?;
        out.write_all(&(name.len() as u16).to_le_bytes())?;
        out.write_all(&0_u16.to_le_bytes())?;
        out.write_all(&0_u16.to_le_bytes())?;
        out.write_all(&0_u16.to_le_bytes())?;
        out.write_all(&0_u16.to_le_bytes())?;
        out.write_all(&0_u32.to_le_bytes())?;
        out.write_all(&entry.offset.to_le_bytes())?;
        out.write_all(name)?;
        offset += 46 + name.len() as u32;
    }
    let central_size = offset - central_offset;
    let entry_count = u16::try_from(central.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "too many NPZ entries"))?;
    out.write_all(&0x06054b50_u32.to_le_bytes())?;
    out.write_all(&0_u16.to_le_bytes())?;
    out.write_all(&0_u16.to_le_bytes())?;
    out.write_all(&entry_count.to_le_bytes())?;
    out.write_all(&entry_count.to_le_bytes())?;
    out.write_all(&central_size.to_le_bytes())?;
    out.write_all(&central_offset.to_le_bytes())?;
    out.write_all(&0_u16.to_le_bytes())?;
    out.flush()
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
