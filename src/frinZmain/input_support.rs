use memmap2::Mmap;
use std::error::Error;
use std::fs::{self, File};
use std::io::{copy, Read, Write};
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;
use zstd::stream::read::Decoder;

fn is_zstd_input(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("zst"))
        .unwrap_or(false)
}

pub fn read_input_prefix(path: &Path, len: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut buf = vec![0u8; len];
    if is_zstd_input(path) {
        let file = File::open(path)?;
        let mut decoder = Decoder::new(file)?;
        decoder.read_exact(&mut buf)?;
    } else {
        let mut file = File::open(path)?;
        file.read_exact(&mut buf)?;
    }
    Ok(buf)
}

pub fn read_input_bytes(path: &Path) -> Result<Vec<u8>, Box<dyn Error>> {
    if is_zstd_input(path) {
        let file = File::open(path)?;
        let mut decoder = Decoder::new(file)?;
        let mut buffer = Vec::new();
        decoder.read_to_end(&mut buffer)?;
        Ok(buffer)
    } else {
        Ok(fs::read(path)?)
    }
}

pub fn open_input_mmap(path: &Path) -> Result<(Mmap, Option<NamedTempFile>), Box<dyn Error>> {
    if !is_zstd_input(path) {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        return Ok((mmap, None));
    }

    let mut decoder = Decoder::new(File::open(path)?)?;
    let mut temp = tempfile::Builder::new()
        .prefix("frinz_input_")
        .suffix(".cor")
        .tempfile_in(std::env::temp_dir())?;
    copy(&mut decoder, &mut temp)?;
    temp.as_file_mut().flush()?;

    let map_file = temp.reopen()?;
    let mmap = unsafe { Mmap::map(&map_file)? };
    Ok((mmap, Some(temp)))
}

pub fn output_stem_from_path(path: &Path) -> Result<String, Box<dyn Error>> {
    let stem = if is_zstd_input(path) {
        PathBuf::from(path.file_stem().ok_or("Invalid input filename")?)
            .file_stem()
            .ok_or("Invalid compressed input filename")?
            .to_string_lossy()
            .to_string()
    } else {
        path.file_stem()
            .ok_or("Invalid input filename")?
            .to_string_lossy()
            .to_string()
    };
    Ok(stem)
}
