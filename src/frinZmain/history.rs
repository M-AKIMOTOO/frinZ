use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const MAGIC: &[u8; 4] = b"FZH3";
const PREVIOUS_MAGIC: &[u8; 4] = b"FZH2";
const LEGACY_MAGIC: &[u8; 4] = b"FZH1";
const MAX_HISTORY_ENTRY_BYTES: usize = 16 * 1024 * 1024;
const MAX_HISTORY_ARCHIVE_BYTES: usize = 256 * 1024 * 1024;

struct HistoryEntry {
    /// YYYYMMDDhhmmss as an integer. Zero marks entries predating timestamps.
    timestamp: u64,
    arguments: Vec<String>,
}

#[cfg(unix)]
struct HistoryLock {
    _file: fs::File,
}

#[cfg(unix)]
impl HistoryLock {
    fn acquire(history_path: &Path) -> io::Result<Self> {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::OpenOptionsExt;

        let lock_path = history_path.with_file_name(".frinZhistory.lock");
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(lock_path)?;
        // LOCK_EX serializes rewrite operations across frinZ processes.
        let result = unsafe { flock(file.as_raw_fd(), 2) };
        if result == -1 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { _file: file })
    }
}

#[cfg(unix)]
extern "C" {
    fn flock(fd: std::os::raw::c_int, operation: std::os::raw::c_int) -> std::os::raw::c_int;
}

#[cfg(not(unix))]
struct HistoryLock;

#[cfg(not(unix))]
impl HistoryLock {
    fn acquire(_history_path: &Path) -> io::Result<Self> {
        Ok(Self)
    }
}

fn history_path() -> io::Result<PathBuf> {
    std::env::var_os("HOME")
        .map(|home| PathBuf::from(home).join(".frinZhistory"))
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "HOME is not set"))
}

fn encode_arguments(arguments: &[String]) -> io::Result<Vec<u8>> {
    let mut raw = Vec::new();
    let count = u32::try_from(arguments.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "too many arguments"))?;
    raw.extend_from_slice(&count.to_le_bytes());
    for argument in arguments {
        let bytes = argument.as_bytes();
        let length = u32::try_from(bytes.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "one argument is too long"))?;
        raw.extend_from_slice(&length.to_le_bytes());
        raw.extend_from_slice(bytes);
    }
    if raw.len() > MAX_HISTORY_ENTRY_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "command line is too large to store in history",
        ));
    }
    Ok(raw)
}

fn checksum(bytes: &[u8]) -> u32 {
    let mut value = 2_166_136_261u32;
    for byte in bytes {
        value ^= u32::from(*byte);
        value = value.wrapping_mul(16_777_619);
    }
    value
}

/// Appends one argument vector and rewrites the compact history archive.
pub fn append_history(arguments: &[String]) -> io::Result<()> {
    let timestamp = chrono::Local::now()
        .format("%Y%m%d%H%M%S")
        .to_string()
        .parse::<u64>()
        .map_err(|_| invalid_history("could not encode local timestamp"))?;
    let path = history_path()?;
    let _lock = HistoryLock::acquire(&path)?;
    let mut entries = load_history()?;
    entries.push(HistoryEntry {
        timestamp,
        arguments: arguments.to_vec(),
    });
    rewrite_history(&path, &entries)
}

fn rewrite_history(path: &Path, entries: &[HistoryEntry]) -> io::Result<()> {
    let raw = encode_history_entries(entries)?;
    let archive = pack_history_archive(&raw)?;
    write_history_atomically(path, &archive)
}

fn compact_legacy_history() -> io::Result<()> {
    let path = history_path()?;
    let data = match fs::read(&path) {
        Ok(data) => data,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    if !data.starts_with(LEGACY_MAGIC) && !data.starts_with(PREVIOUS_MAGIC) {
        return Ok(());
    }
    let _lock = HistoryLock::acquire(&path)?;
    let data = fs::read(&path)?;
    if !data.starts_with(LEGACY_MAGIC) && !data.starts_with(PREVIOUS_MAGIC) {
        return Ok(());
    }
    let entries = decode_history_data(&data)?;
    rewrite_history(&path, &entries)
}

fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

fn decode_arguments(raw: &[u8]) -> io::Result<Vec<String>> {
    if raw.len() < 4 {
        return Err(invalid_history("short argument list"));
    }
    let count = read_u32(raw, 0) as usize;
    if count > raw.len() / 4 {
        return Err(invalid_history("invalid argument count"));
    }
    let mut offset = 4;
    let mut arguments = Vec::with_capacity(count);
    for _ in 0..count {
        if raw.len().saturating_sub(offset) < 4 {
            return Err(invalid_history("short argument length"));
        }
        let length = read_u32(raw, offset) as usize;
        offset += 4;
        let end = offset
            .checked_add(length)
            .filter(|end| *end <= raw.len())
            .ok_or_else(|| invalid_history("short argument data"))?;
        let argument = String::from_utf8(raw[offset..end].to_vec())
            .map_err(|_| invalid_history("argument is not UTF-8"))?;
        arguments.push(argument);
        offset = end;
    }
    if offset != raw.len() {
        return Err(invalid_history("trailing argument data"));
    }
    Ok(arguments)
}

fn encode_history_entries(entries: &[HistoryEntry]) -> io::Result<Vec<u8>> {
    let count = u32::try_from(entries.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "history is too long"))?;
    let mut raw = Vec::new();
    raw.extend_from_slice(&count.to_le_bytes());
    for entry in entries {
        let encoded = encode_arguments(&entry.arguments)?;
        let length = u32::try_from(encoded.len()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "history entry is too long")
        })?;
        raw.extend_from_slice(&entry.timestamp.to_le_bytes());
        raw.extend_from_slice(&length.to_le_bytes());
        raw.extend_from_slice(&encoded);
        if raw.len() > MAX_HISTORY_ARCHIVE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "history archive exceeds 256 MiB",
            ));
        }
    }
    Ok(raw)
}

fn decode_history_entries_v2(raw: &[u8]) -> io::Result<Vec<HistoryEntry>> {
    if raw.len() < 4 {
        return Err(invalid_history("short history archive"));
    }
    let count = read_u32(raw, 0) as usize;
    if count > raw.len() / 4 {
        return Err(invalid_history("invalid history entry count"));
    }
    let mut offset = 4;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        if raw.len().saturating_sub(offset) < 4 {
            return Err(invalid_history("short history entry length"));
        }
        let length = read_u32(raw, offset) as usize;
        offset += 4;
        if length > MAX_HISTORY_ENTRY_BYTES {
            return Err(invalid_history("history entry is too large"));
        }
        let end = offset
            .checked_add(length)
            .filter(|end| *end <= raw.len())
            .ok_or_else(|| invalid_history("short history entry"))?;
        entries.push(HistoryEntry {
            timestamp: 0,
            arguments: decode_arguments(&raw[offset..end])?,
        });
        offset = end;
    }
    if offset != raw.len() {
        return Err(invalid_history("trailing history archive data"));
    }
    Ok(entries)
}

fn decode_history_entries_v3(raw: &[u8]) -> io::Result<Vec<HistoryEntry>> {
    if raw.len() < 4 {
        return Err(invalid_history("short history archive"));
    }
    let count = read_u32(raw, 0) as usize;
    if count > raw.len() / 12 {
        return Err(invalid_history("invalid history entry count"));
    }
    let mut offset = 4;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        if raw.len().saturating_sub(offset) < 12 {
            return Err(invalid_history("short timestamped history entry"));
        }
        let timestamp = u64::from_le_bytes(raw[offset..offset + 8].try_into().unwrap());
        let length = read_u32(raw, offset + 8) as usize;
        offset += 12;
        if length > MAX_HISTORY_ENTRY_BYTES {
            return Err(invalid_history("history entry is too large"));
        }
        let end = offset
            .checked_add(length)
            .filter(|end| *end <= raw.len())
            .ok_or_else(|| invalid_history("short history entry"))?;
        entries.push(HistoryEntry {
            timestamp,
            arguments: decode_arguments(&raw[offset..end])?,
        });
        offset = end;
    }
    if offset != raw.len() {
        return Err(invalid_history("trailing history archive data"));
    }
    Ok(entries)
}

fn decode_dictionary_arguments(
    payload: &[u8],
    previous_entries: &[Vec<String>],
) -> io::Result<Vec<String>> {
    if payload.len() < 4 {
        return Err(invalid_history("short dictionary record"));
    }
    let count = read_u32(payload, 0) as usize;
    if count > payload.len() / 5 {
        return Err(invalid_history("invalid dictionary argument count"));
    }
    let mut offset = 4;
    let mut arguments = Vec::with_capacity(count);
    for _ in 0..count {
        let tag = *payload
            .get(offset)
            .ok_or_else(|| invalid_history("short dictionary token"))?;
        offset += 1;
        match tag {
            0 => {
                if payload.len().saturating_sub(offset) < 4 {
                    return Err(invalid_history("short dictionary literal length"));
                }
                let length = read_u32(payload, offset) as usize;
                offset += 4;
                let end = offset
                    .checked_add(length)
                    .filter(|end| *end <= payload.len())
                    .ok_or_else(|| invalid_history("short dictionary literal"))?;
                let argument = String::from_utf8(payload[offset..end].to_vec())
                    .map_err(|_| invalid_history("argument is not UTF-8"))?;
                arguments.push(argument);
                offset = end;
            }
            1 => {
                if payload.len().saturating_sub(offset) < 8 {
                    return Err(invalid_history("short dictionary reference"));
                }
                let entry_index = read_u32(payload, offset) as usize;
                let argument_index = read_u32(payload, offset + 4) as usize;
                offset += 8;
                let argument = previous_entries
                    .get(entry_index)
                    .and_then(|entry| entry.get(argument_index))
                    .ok_or_else(|| invalid_history("invalid dictionary reference"))?;
                arguments.push(argument.clone());
            }
            _ => return Err(invalid_history("unknown dictionary token")),
        }
    }
    if offset != payload.len() {
        return Err(invalid_history("trailing dictionary data"));
    }
    Ok(arguments)
}

fn invalid_history(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn pack_history_archive(raw: &[u8]) -> io::Result<Vec<u8>> {
    if raw.len() > MAX_HISTORY_ARCHIVE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "history archive exceeds 256 MiB",
        ));
    }
    let lzss = lzss_compress(raw);
    let huffman = huffman_compress(&lzss)?;
    let (mode, intermediate_length, payload) = [
        (0u8, 0usize, raw),
        (1u8, 0usize, lzss.as_slice()),
        (2u8, lzss.len(), huffman.as_slice()),
    ]
    .into_iter()
    .min_by_key(|(_, _, data)| data.len())
    .unwrap();

    let mut archive = Vec::with_capacity(33 + payload.len());
    archive.extend_from_slice(MAGIC);
    archive.extend_from_slice(&(raw.len() as u64).to_le_bytes());
    archive.extend_from_slice(&(intermediate_length as u64).to_le_bytes());
    archive.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    archive.push(mode);
    archive.extend_from_slice(&checksum(raw).to_le_bytes());
    archive.extend_from_slice(payload);
    Ok(archive)
}

fn unpack_archive_raw(data: &[u8], expected_magic: &[u8; 4]) -> io::Result<Vec<u8>> {
    const HEADER_LENGTH: usize = 33;
    if data.len() < HEADER_LENGTH || &data[..expected_magic.len()] != expected_magic {
        return Err(invalid_history(
            "short or invalid compressed history archive",
        ));
    }
    let raw_length = usize::try_from(u64::from_le_bytes(data[4..12].try_into().unwrap()))
        .map_err(|_| invalid_history("history archive is too large"))?;
    let intermediate_length = usize::try_from(u64::from_le_bytes(data[12..20].try_into().unwrap()))
        .map_err(|_| invalid_history("compressed history archive is too large"))?;
    let payload_length = usize::try_from(u64::from_le_bytes(data[20..28].try_into().unwrap()))
        .map_err(|_| invalid_history("compressed history archive is too large"))?;
    let mode = data[28];
    let expected_checksum = read_u32(data, 29);
    if raw_length > MAX_HISTORY_ARCHIVE_BYTES {
        return Err(invalid_history("history archive exceeds 256 MiB"));
    }
    let end = HEADER_LENGTH
        .checked_add(payload_length)
        .filter(|end| *end == data.len())
        .ok_or_else(|| invalid_history("compressed history archive has an invalid length"))?;
    let payload = &data[HEADER_LENGTH..end];
    let raw = match mode {
        0 if intermediate_length == 0 && payload.len() == raw_length => payload.to_vec(),
        0 => return Err(invalid_history("invalid raw history archive length")),
        1 if intermediate_length == 0 => lzss_decompress(payload, raw_length)?,
        1 => return Err(invalid_history("invalid LZSS history archive header")),
        2 if intermediate_length <= MAX_HISTORY_ARCHIVE_BYTES + MAX_HISTORY_ARCHIVE_BYTES / 4 => {
            let lzss = huffman_decompress(payload, intermediate_length)?;
            lzss_decompress(&lzss, raw_length)?
        }
        2 => return Err(invalid_history("compressed history archive is too large")),
        _ => return Err(invalid_history("unknown history compression mode")),
    };
    if checksum(&raw) != expected_checksum {
        return Err(invalid_history("history archive checksum mismatch"));
    }
    Ok(raw)
}

fn unpack_history_archive_v2(data: &[u8]) -> io::Result<Vec<HistoryEntry>> {
    decode_history_entries_v2(&unpack_archive_raw(data, PREVIOUS_MAGIC)?)
}

fn unpack_history_archive_v3(data: &[u8]) -> io::Result<Vec<HistoryEntry>> {
    decode_history_entries_v3(&unpack_archive_raw(data, MAGIC)?)
}

fn write_history_atomically(path: &Path, data: &[u8]) -> io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temporary_path = parent.join(format!(
        ".frinZhistory.{}.{}.tmp",
        std::process::id(),
        stamp
    ));

    let result = (|| {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temporary_path)?;
        file.write_all(data)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary_path, path)?;
        if let Ok(directory) = fs::File::open(parent) {
            let _ = directory.sync_all();
        }
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result
}

fn load_history() -> io::Result<Vec<HistoryEntry>> {
    let path = history_path()?;
    let data = match fs::read(path) {
        Ok(data) => data,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };
    if data.is_empty() {
        return Ok(Vec::new());
    }
    decode_history_data(&data)
}

fn decode_history_data(data: &[u8]) -> io::Result<Vec<HistoryEntry>> {
    if data.starts_with(MAGIC) {
        return unpack_history_archive_v3(data);
    }
    if data.starts_with(PREVIOUS_MAGIC) {
        return unpack_history_archive_v2(data);
    }
    decode_legacy_history(&data)
}

fn decode_legacy_history(data: &[u8]) -> io::Result<Vec<HistoryEntry>> {
    if !data.starts_with(LEGACY_MAGIC) {
        return Err(invalid_history("unsupported .frinZhistory format"));
    }
    let mut entries = Vec::new();
    let mut previous_arguments: Vec<Vec<String>> = Vec::new();
    let mut offset = 0;
    while offset < data.len() {
        if data.len() - offset < LEGACY_MAGIC.len() + 13 {
            break;
        }
        if &data[offset..offset + LEGACY_MAGIC.len()] != LEGACY_MAGIC {
            return Err(invalid_history("unsupported .frinZhistory format"));
        }
        offset += LEGACY_MAGIC.len();
        let raw_length = read_u32(&data, offset) as usize;
        let payload_length = read_u32(&data, offset + 4) as usize;
        let mode = data[offset + 8];
        let expected_checksum = read_u32(&data, offset + 9);
        offset += 13;
        if raw_length > MAX_HISTORY_ENTRY_BYTES {
            return Err(invalid_history("history entry is too large"));
        }
        let end = offset
            .checked_add(payload_length)
            .filter(|end| *end <= data.len());
        let Some(end) = end else {
            break;
        };
        let payload = &data[offset..end];
        let (raw, arguments) = match mode {
            0 if payload.len() == raw_length => {
                let raw = payload.to_vec();
                let arguments = decode_arguments(&raw)?;
                (raw, arguments)
            }
            0 => return Err(invalid_history("invalid uncompressed record length")),
            1 => {
                let raw = lzss_decompress(payload, raw_length)?;
                let arguments = decode_arguments(&raw)?;
                (raw, arguments)
            }
            2 => {
                let arguments = decode_dictionary_arguments(payload, &previous_arguments)?;
                let raw = encode_arguments(&arguments)?;
                if raw.len() != raw_length {
                    return Err(invalid_history("invalid dictionary record length"));
                }
                (raw, arguments)
            }
            _ => return Err(invalid_history("unknown history compression mode")),
        };
        if checksum(&raw) != expected_checksum {
            return Err(invalid_history("history record checksum mismatch"));
        }
        previous_arguments.push(arguments.clone());
        entries.push(HistoryEntry {
            timestamp: 0,
            arguments,
        });
        offset = end;
    }
    Ok(entries)
}

fn read_history() -> io::Result<Vec<HistoryEntry>> {
    load_history()
}

/// Prints stored commands. A limit of zero prints every entry.
pub fn print_history(limit: usize) -> io::Result<()> {
    compact_legacy_history()?;
    let entries = read_history()?;
    if entries.is_empty() {
        println!("frinZ の履歴はありません。");
        return Ok(());
    }

    let start = if limit == 0 {
        0
    } else {
        entries.len().saturating_sub(limit)
    };
    for (index, entry) in entries.iter().enumerate().skip(start) {
        let mut command = String::from("frinZ");
        for argument in &entry.arguments {
            command.push(' ');
            command.push_str(&shell_quote(argument));
        }
        println!("{:>5}  {:014}  {command}", index + 1, entry.timestamp);
    }
    Ok(())
}

fn shell_quote(argument: &str) -> String {
    if !argument.is_empty()
        && argument
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_./:@%+=,-".contains(&byte))
    {
        return argument.to_owned();
    }
    format!("'{}'", argument.replace('\'', "'\\''"))
}

fn three_byte_key(input: &[u8], offset: usize) -> Option<u32> {
    (offset + 2 < input.len()).then(|| {
        (u32::from(input[offset]) << 16)
            | (u32::from(input[offset + 1]) << 8)
            | u32::from(input[offset + 2])
    })
}

fn longest_match(
    input: &[u8],
    position: usize,
    positions: &HashMap<u32, Vec<usize>>,
) -> (usize, usize) {
    let Some(key) = three_byte_key(input, position) else {
        return (0, 0);
    };
    let Some(candidates) = positions.get(&key) else {
        return (0, 0);
    };

    let mut best = (0usize, 0usize);
    let max_length = (input.len() - position).min(u8::MAX as usize);
    for candidate in candidates.iter().rev().take(64) {
        let candidate = *candidate;
        let distance = position - candidate;
        if distance > u16::MAX as usize {
            break;
        }
        let mut length = 0;
        while length < max_length && input[candidate + length] == input[position + length] {
            length += 1;
        }
        if length > best.1 {
            best = (distance, length);
        }
    }
    best
}

fn lzss_compress(input: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len());
    let mut positions: HashMap<u32, Vec<usize>> = HashMap::new();
    let mut position = 0;

    while position < input.len() {
        let flag_offset = output.len();
        output.push(0);
        let mut flags = 0u8;
        for bit in 0..8 {
            if position >= input.len() {
                break;
            }
            let (distance, length) = longest_match(input, position, &positions);
            let advance = if length >= 3 {
                flags |= 1 << bit;
                output.extend_from_slice(&(distance as u16).to_le_bytes());
                output.push(length as u8);
                length
            } else {
                output.push(input[position]);
                1
            };

            let next_position = position + advance;
            for indexed_position in position..next_position {
                if let Some(key) = three_byte_key(input, indexed_position) {
                    let candidates = positions.entry(key).or_default();
                    candidates.push(indexed_position);
                    if candidates.len() > 256 {
                        candidates.drain(..128);
                    }
                }
            }
            position = next_position;
        }
        output[flag_offset] = flags;
    }
    output
}

fn lzss_decompress(input: &[u8], expected_length: usize) -> io::Result<Vec<u8>> {
    let mut output = Vec::with_capacity(expected_length);
    let mut offset = 0;
    while offset < input.len() && output.len() < expected_length {
        let flags = input[offset];
        offset += 1;
        for bit in 0..8 {
            if output.len() == expected_length {
                break;
            }
            if flags & (1 << bit) != 0 {
                if input.len().saturating_sub(offset) < 3 {
                    return Err(invalid_history("short compressed match"));
                }
                let distance = u16::from_le_bytes([input[offset], input[offset + 1]]) as usize;
                let length = input[offset + 2] as usize;
                offset += 3;
                if distance == 0 || distance > output.len() || length < 3 {
                    return Err(invalid_history("invalid compressed match"));
                }
                if output
                    .len()
                    .checked_add(length)
                    .filter(|length| *length <= expected_length)
                    .is_none()
                {
                    return Err(invalid_history(
                        "compressed record exceeds its declared size",
                    ));
                }
                for _ in 0..length {
                    let byte = output[output.len() - distance];
                    output.push(byte);
                }
            } else {
                if offset == input.len() {
                    return Err(invalid_history("short compressed literal"));
                }
                output.push(input[offset]);
                offset += 1;
            }
        }
    }
    if output.len() != expected_length || offset != input.len() {
        return Err(invalid_history("compressed record has an invalid length"));
    }
    Ok(output)
}

struct HuffmanNode {
    symbol: Option<u8>,
    left: Option<usize>,
    right: Option<usize>,
}

enum HuffmanTree {
    Leaf(u8),
    Branch(Box<HuffmanTree>, Box<HuffmanTree>),
}

struct BitWriter {
    bytes: Vec<u8>,
    bit_length: usize,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_length: 0,
        }
    }

    fn write_bit(&mut self, value: bool) {
        if self.bit_length % 8 == 0 {
            self.bytes.push(0);
        }
        if value {
            let bit = 7 - self.bit_length % 8;
            let last = self.bytes.len() - 1;
            self.bytes[last] |= 1 << bit;
        }
        self.bit_length += 1;
    }

    fn write_byte(&mut self, value: u8) {
        for bit in (0..8).rev() {
            self.write_bit(value & (1 << bit) != 0);
        }
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    bit_offset: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            bit_offset: 0,
        }
    }

    fn read_bit(&mut self) -> io::Result<bool> {
        let byte = *self
            .bytes
            .get(self.bit_offset / 8)
            .ok_or_else(|| invalid_history("short Huffman bitstream"))?;
        let bit = 7 - self.bit_offset % 8;
        self.bit_offset += 1;
        Ok(byte & (1 << bit) != 0)
    }

    fn read_byte(&mut self) -> io::Result<u8> {
        let mut value = 0u8;
        for _ in 0..8 {
            value = (value << 1) | u8::from(self.read_bit()?);
        }
        Ok(value)
    }

    fn finish(&self) -> io::Result<()> {
        if self.bit_offset.saturating_add(7) / 8 != self.bytes.len() {
            return Err(invalid_history("Huffman payload has trailing bytes"));
        }
        for bit_offset in self.bit_offset..self.bytes.len() * 8 {
            let byte = self.bytes[bit_offset / 8];
            let bit = 7 - bit_offset % 8;
            if byte & (1 << bit) != 0 {
                return Err(invalid_history("invalid Huffman padding bits"));
            }
        }
        Ok(())
    }
}

fn collect_huffman_codes(
    nodes: &[HuffmanNode],
    node_index: usize,
    path: &mut Vec<bool>,
    codes: &mut [Vec<bool>],
) {
    let node = &nodes[node_index];
    if let Some(symbol) = node.symbol {
        codes[symbol as usize] = path.clone();
        return;
    }
    path.push(false);
    collect_huffman_codes(nodes, node.left.unwrap(), path, codes);
    path.pop();
    path.push(true);
    collect_huffman_codes(nodes, node.right.unwrap(), path, codes);
    path.pop();
}

fn write_huffman_tree(nodes: &[HuffmanNode], node_index: usize, writer: &mut BitWriter) {
    let node = &nodes[node_index];
    if let Some(symbol) = node.symbol {
        writer.write_bit(true);
        writer.write_byte(symbol);
    } else {
        writer.write_bit(false);
        write_huffman_tree(nodes, node.left.unwrap(), writer);
        write_huffman_tree(nodes, node.right.unwrap(), writer);
    }
}

fn huffman_compress(input: &[u8]) -> io::Result<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let mut frequencies = [0u64; 256];
    for byte in input {
        frequencies[*byte as usize] += 1;
    }

    let mut nodes = Vec::new();
    let mut queue = BinaryHeap::new();
    for (symbol, frequency) in frequencies.into_iter().enumerate() {
        if frequency == 0 {
            continue;
        }
        let node_index = nodes.len();
        nodes.push(HuffmanNode {
            symbol: Some(symbol as u8),
            left: None,
            right: None,
        });
        queue.push(Reverse((frequency, node_index)));
    }

    while queue.len() > 1 {
        let Reverse((left_frequency, left)) = queue.pop().unwrap();
        let Reverse((right_frequency, right)) = queue.pop().unwrap();
        let node_index = nodes.len();
        nodes.push(HuffmanNode {
            symbol: None,
            left: Some(left),
            right: Some(right),
        });
        queue.push(Reverse((left_frequency + right_frequency, node_index)));
    }
    let Reverse((_, root)) = queue.pop().unwrap();

    let mut codes: [Vec<bool>; 256] = std::array::from_fn(|_| Vec::new());
    collect_huffman_codes(&nodes, root, &mut Vec::new(), &mut codes);
    let mut writer = BitWriter::new();
    write_huffman_tree(&nodes, root, &mut writer);
    if nodes[root].symbol.is_none() {
        for byte in input {
            for &bit in &codes[*byte as usize] {
                writer.write_bit(bit);
            }
        }
    }
    Ok(writer.bytes)
}

fn read_huffman_tree(
    reader: &mut BitReader<'_>,
    depth: usize,
    node_count: &mut usize,
) -> io::Result<HuffmanTree> {
    *node_count += 1;
    if depth > 256 || *node_count > 511 {
        return Err(invalid_history("Huffman tree is too deep"));
    }
    if reader.read_bit()? {
        Ok(HuffmanTree::Leaf(reader.read_byte()?))
    } else {
        let left = read_huffman_tree(reader, depth + 1, node_count)?;
        let right = read_huffman_tree(reader, depth + 1, node_count)?;
        Ok(HuffmanTree::Branch(Box::new(left), Box::new(right)))
    }
}

fn decode_huffman_symbol(tree: &HuffmanTree, reader: &mut BitReader<'_>) -> io::Result<u8> {
    let mut node = tree;
    loop {
        match node {
            HuffmanTree::Leaf(symbol) => return Ok(*symbol),
            HuffmanTree::Branch(left, right) => {
                node = if reader.read_bit()? { right } else { left };
            }
        }
    }
}

fn huffman_decompress(input: &[u8], expected_length: usize) -> io::Result<Vec<u8>> {
    let mut reader = BitReader::new(input);
    let mut node_count = 0;
    let tree = read_huffman_tree(&mut reader, 0, &mut node_count)?;
    let mut output = Vec::with_capacity(expected_length);
    match &tree {
        HuffmanTree::Leaf(symbol) => output.resize(expected_length, *symbol),
        HuffmanTree::Branch(_, _) => {
            for _ in 0..expected_length {
                output.push(decode_huffman_symbol(&tree, &mut reader)?);
            }
        }
    }
    reader.finish()?;
    Ok(output)
}
