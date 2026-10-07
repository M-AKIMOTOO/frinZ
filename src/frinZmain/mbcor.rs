//! Frequency-aware fringe analysis for yi-corr YIMBCOR version 1.
//!
//! Keep the RF gap out of the noise weights. A rate is defined at reference_hz
//! and scales with each channel's actual RF; no target IF-phase fit is performed.
use crate::{
    args::Args,
    fft::cached_fft_plan,
    header::{parse_header, CorHeader},
    input_support::{open_input_data, output_stem_from_path, read_input_prefix},
    npy_output::{NamedNpz, NpyMeta},
};
use byteorder::{LittleEndian, ReadBytesExt};
use num_complex::Complex;
use plotters::prelude::*;
use rayon::prelude::*;
use std::{
    error::Error,
    fs,
    io::{self, BufWriter, Cursor, Write},
    path::Path,
};

type C = Complex<f32>;
const TAU: f64 = std::f64::consts::TAU;
const MAGIC: &[u8; 8] = b"YIMBCOR\0";
type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub fn is_mbcor(path: &Path) -> Result<bool> {
    Ok(read_input_prefix(path, 8)?.as_slice() == MAGIC)
}

struct Layout {
    headers: [CorHeader; 2],
    offsets: [usize; 2],
    stride: usize,
    reference_hz: f64,
    bandwidth_hz: f64,
    rows: usize,
}

fn invalid(message: &str) -> Box<dyn Error> {
    io::Error::new(io::ErrorKind::InvalidData, message).into()
}

impl Layout {
    fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 544 || &bytes[..8] != MAGIC {
            return Err(invalid("invalid/truncated MBCOR header"));
        }
        let mut cursor = Cursor::new(&bytes[8..32]);
        if cursor.read_u32::<LittleEndian>()? != 1 || cursor.read_u32::<LittleEndian>()? != 2 {
            return Err(invalid("MBCOR supports version 1 with two bands"));
        }
        let bandwidth_hz = cursor.read_f64::<LittleEndian>()?;
        let reference_hz = cursor.read_f64::<LittleEndian>()?;
        if !bandwidth_hz.is_finite()
            || bandwidth_hz <= 0.0
            || !reference_hz.is_finite()
            || reference_hz <= 0.0
        {
            return Err(invalid("invalid MBCOR bandwidth/reference RF"));
        }
        let headers = [
            parse_header(&mut Cursor::new(&bytes[32..288]))?,
            parse_header(&mut Cursor::new(&bytes[288..544]))?,
        ];
        let a = &headers[0];
        let b = &headers[1];
        if a.number_of_sector != b.number_of_sector
            || a.station1_name != b.station1_name
            || a.station2_name != b.station2_name
            || a.source_name != b.source_name
            || a.station1_position != b.station1_position
            || a.station2_position != b.station2_position
            || a.source_position_ra != b.source_position_ra
            || a.source_position_dec != b.source_position_dec
        {
            return Err(invalid(
                "MBCOR bands have different sources, baselines, or sector counts",
            ));
        }
        let size_a = 128 + a.fft_point as usize * 4;
        let size_b = 128 + b.fft_point as usize * 4;
        let stride = size_a
            .checked_add(size_b)
            .ok_or_else(|| invalid("MBCOR size overflow"))?;
        let rows = a.number_of_sector as usize;
        let expected = rows
            .checked_mul(stride)
            .and_then(|n| n.checked_add(544))
            .ok_or_else(|| invalid("MBCOR size overflow"))?;
        if bytes.len() != expected {
            return Err(invalid(
                "MBCOR payload length does not match both embedded headers",
            ));
        }
        let expected_bw = (a.sampling_speed as f64 + b.sampling_speed as f64) / 2.0;
        if (bandwidth_hz - expected_bw).abs() > 1e-6 * expected_bw {
            return Err(invalid(
                "MBCOR occupied bandwidth disagrees with embedded headers",
            ));
        }
        if b.observing_frequency < a.observing_frequency + a.sampling_speed as f64 / 2.0 {
            return Err(invalid("MBCOR bands must be ordered and non-overlapping"));
        }
        Ok(Self {
            headers,
            offsets: [544, 544 + size_a],
            stride,
            reference_hz,
            bandwidth_hz,
            rows,
        })
    }
    fn sector<'a>(&self, bytes: &'a [u8], band: usize, row: usize) -> &'a [u8] {
        let start = self.offsets[band] + row * self.stride;
        &bytes[start..start + 128 + self.headers[band].fft_point as usize * 4]
    }
    fn time(&self, bytes: &[u8], row: usize) -> Result<(f64, f64)> {
        let a = self.sector(bytes, 0, row);
        let b = self.sector(bytes, 1, row);
        if a[..16] != b[..16] || a[112..116] != b[112..116] {
            return Err(invalid("MBCOR band timestamps/integration differ"));
        }
        let mut cursor = Cursor::new(&a[..16]);
        let start_s = cursor.read_u32::<LittleEndian>()?;
        let start_ns = cursor.read_u32::<LittleEndian>()?;
        let end_s = cursor.read_u32::<LittleEndian>()?;
        let end_ns = cursor.read_u32::<LittleEndian>()?;
        if start_ns >= 1_000_000_000 || end_ns >= 1_000_000_000 {
            return Err(invalid("invalid MBCOR nanosecond timestamp"));
        }
        let duration = f32::from_le_bytes(a[112..116].try_into()?) as f64;
        let elapsed = end_s as f64 - start_s as f64 + (end_ns as f64 - start_ns as f64) * 1e-9;
        if !duration.is_finite() || duration <= 0.0 || (elapsed - duration).abs() > 2e-6 {
            return Err(invalid("invalid MBCOR integration duration"));
        }
        Ok((start_s as f64 + start_ns as f64 * 1e-9, duration))
    }
}

struct Window {
    frequency: Vec<f64>,
    band: Vec<usize>,
    values: Vec<Vec<C>>, // occupied channel, time
    reference: f64,
    df: f64,
    sampling_hz: f64,
    bins: Vec<usize>,
    delay_fft: usize,
    start: f64,
    dt: f64,
    rows: usize,
    count: usize,
}

fn ranges(args: &Args) -> Result<Vec<(f64, f64)>> {
    args.rfi
        .iter()
        .map(|s| {
            let (lo, hi) = s
                .split_once(',')
                .ok_or("MBCOR --rfi uses MIN,MAX in MHz relative to the low-band edge")?;
            let lo: f64 = lo.parse()?;
            let hi: f64 = hi.parse()?;
            if !lo.is_finite() || !hi.is_finite() || lo < 0.0 || hi <= lo {
                return Err(invalid("invalid MBCOR RFI range"));
            }
            Ok((lo * 1e6, hi * 1e6))
        })
        .collect()
}

impl Window {
    fn read(bytes: &[u8], layout: &Layout, first: usize, rows: usize, args: &Args) -> Result<Self> {
        let (start, dt) = layout.time(bytes, first)?;
        let (file_start, file_dt) = layout.time(bytes, 0)?;
        if (dt - file_dt).abs() > 2e-6 {
            return Err(invalid(
                "MBCOR sectors must use the same integration duration as the file start",
            ));
        }
        let h0 = &layout.headers[0];
        let df = h0.sampling_speed as f64 / h0.fft_point as f64;
        let min_frequency = h0.observing_frequency;
        let rfi = ranges(args)?;
        for row in first..first + rows {
            let (time, duration) = layout.time(bytes, row)?;
            if (duration - dt).abs() > 2e-6
                || (time - start - (row - first) as f64 * dt).abs() > 2e-6
            {
                return Err(invalid(
                    "MBCOR fringe FFT requires contiguous, uniformly integrated sectors",
                ));
            }
        }
        let mut frequency = Vec::new();
        let mut band = Vec::new();
        let mut values = Vec::new();
        let mut bins = Vec::new();
        let mut count = 0;
        for b in 0..2 {
            let h = &layout.headers[b];
            let step = h.sampling_speed as f64 / h.fft_point as f64;
            if (step - df).abs() > df * 1e-8 {
                return Err(invalid(
                    "MBCOR fringe FFT requires a common channel spacing",
                ));
            }
            for k in 0..h.fft_point as usize / 2 {
                let rf = h.observing_frequency + k as f64 * step;
                let relative = rf - min_frequency;
                if rfi.iter().any(|&(lo, hi)| relative >= lo && relative < hi)
                    || (args.frange.len() == 2
                        && (relative < args.frange[0] as f64 * 1e6
                            || relative > args.frange[1] as f64 * 1e6))
                {
                    continue;
                }
                let index = relative / df;
                if (index - index.round()).abs() > 1e-6 {
                    return Err(invalid("MBCOR RF bands are not on the same channel grid"));
                }
                if !(0.0..=16_777_215.0).contains(&index) {
                    return Err(invalid("MBCOR RF span/channel grid is too large"));
                }
                let mut channel = Vec::with_capacity(rows);
                for row in first..first + rows {
                    let sector = layout.sector(bytes, b, row);
                    let offset = 128 + k * 8;
                    let re = f32::from_le_bytes(sector[offset..offset + 4].try_into()?);
                    let im = f32::from_le_bytes(sector[offset + 4..offset + 8].try_into()?);
                    if !re.is_finite() || !im.is_finite() {
                        return Err(invalid("non-finite MBCOR visibility"));
                    }
                    let z = C::new(re, im);
                    if z != C::new(0.0, 0.0) {
                        count += 1;
                    }
                    let t = (start - file_start) + (row - first) as f64 * dt + dt / 2.0;
                    let delay_s = args.delay_correct as f64 / h0.sampling_speed as f64;
                    let phase = -TAU
                        * ((rf - layout.reference_hz) * delay_s
                            + rf / layout.reference_hz * args.rate_correct as f64 * t);
                    channel.push(z * C::from_polar(1.0, phase.rem_euclid(TAU) as f32));
                }
                if channel.iter().all(|z| *z == C::new(0.0, 0.0)) {
                    continue;
                }
                frequency.push(rf);
                band.push(b);
                values.push(channel);
                bins.push(index.round() as usize);
            }
        }
        if values.is_empty() || count == 0 {
            return Err(invalid("no usable MBCOR channels after masking"));
        }
        let delay_fft = (bins.iter().copied().max().unwrap() + 1).next_power_of_two();
        Ok(Self {
            frequency,
            band,
            values,
            reference: layout.reference_hz,
            df,
            sampling_hz: h0.sampling_speed as f64,
            bins,
            delay_fft,
            start,
            dt,
            rows,
            count,
        })
    }
    fn evaluate(&self, delay: f64, rate: f64) -> Complex<f64> {
        let mut sum = Complex::<f64>::new(0.0, 0.0);
        let center = (self.rows - 1) as f64 / 2.0;
        for (channel, &rf) in self.values.iter().zip(&self.frequency) {
            let p0 = -TAU * (rf - self.reference) * delay;
            let step = -TAU * rf / self.reference * rate * self.dt;
            let mut phasor = Complex::from_polar(1.0, p0 - step * center);
            let increment = Complex::from_polar(1.0, step);
            for z in channel {
                sum += Complex::new(z.re as f64, z.im as f64) * phasor;
                phasor *= increment;
            }
        }
        sum / self.count as f64
    }
    fn noise_proxy(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;
        for values in &self.values {
            for pair in values.windows(2) {
                if pair[0] != C::new(0.0, 0.0) && pair[1] != C::new(0.0, 0.0) {
                    sum += (pair[1].re as f64 - pair[0].re as f64).powi(2)
                        + (pair[1].im as f64 - pair[0].im as f64).powi(2);
                    count += 1;
                }
            }
        }
        if count == 0 {
            return f64::NAN;
        }
        // E|z[n+1]-z[n]|^2 = 4 sigma_quadrature^2 for independent samples.
        (sum / (4.0 * count as f64 * self.count as f64)).sqrt()
    }
}

fn centered_sample(values: &[C], index: f64, center: f64) -> C {
    let floor = index.floor() as i64;
    let x = (index - floor as f64) as f32;
    let weights = [
        -x * (x - 1.0) * (x - 2.0) / 6.0,
        (x + 1.0) * (x - 1.0) * (x - 2.0) / 2.0,
        -(x + 1.0) * x * (x - 2.0) / 2.0,
        (x + 1.0) * x * (x - 1.0) / 6.0,
    ];
    let mut out = C::new(0.0, 0.0);
    for (j, w) in weights.into_iter().enumerate() {
        let k = floor + j as i64 - 1;
        let phase = TAU * k as f64 / values.len() as f64 * center;
        out += values[k.rem_euclid(values.len() as i64) as usize] * C::from_polar(w, phase as f32);
    }
    out
}

struct Fringe {
    delay: f64,
    rate: f64,
    value: Complex<f64>,
    delays: Vec<f64>,
    delay_amplitude: Vec<f64>,
    rates: Vec<f64>,
    rate_amplitude: Vec<f64>,
}

fn in_range(value: f64, bounds: &[f32]) -> bool {
    bounds.len() != 2 || (value >= bounds[0] as f64 && value <= bounds[1] as f64)
}

fn search(window: &Window, args: &Args, pool: &rayon::ThreadPool) -> Result<Fringe> {
    let time_fft = window
        .rows
        .checked_mul(args.rate_padding.max(1) as usize)
        .and_then(usize::checked_next_power_of_two)
        .ok_or_else(|| invalid("MBCOR rate FFT size overflow"))?;
    let estimated_bytes = time_fft
        .checked_mul(window.values.len())
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| {
            window
                .delay_fft
                .checked_mul(pool.current_num_threads())
                .and_then(|k| k.checked_mul(32))
                .and_then(|k| n.checked_add(k))
        })
        .ok_or_else(|| invalid("MBCOR FFT memory estimate overflow"))?;
    if let Ok(memory) = sys_info::mem_info() {
        if estimated_bytes as u128 > memory.avail as u128 * 1024 * 4 / 5 {
            return Err(invalid(
                "MBCOR FFT exceeds available memory; use a shorter --length or fewer --cpu threads",
            ));
        }
    }
    let time_plan = cached_fft_plan(time_fft, false);
    let time_scratch = time_plan.get_inplace_scratch_len();
    let transforms: Vec<Vec<C>> = pool.install(|| {
        window
            .values
            .par_iter()
            .map(|channel| {
                let mut data = vec![C::new(0.0, 0.0); time_fft];
                data[..window.rows].copy_from_slice(channel);
                time_plan
                    .process_with_scratch(&mut data, &mut vec![C::new(0.0, 0.0); time_scratch]);
                data
            })
            .collect()
    });
    let max_rf = window.frequency.iter().copied().fold(0.0, f64::max);
    let rate_step = 1.0 / (time_fft as f64 * window.dt);
    let rate_limit = (time_fft as f64 / 2.0 * window.reference / max_rf).floor() as i64;
    let rates: Vec<f64> = (-rate_limit..=rate_limit)
        .map(|k| k as f64 * rate_step)
        .filter(|&r| in_range(r, &args.rrange))
        .collect();
    if rates.is_empty() {
        return Err(invalid("empty MBCOR rate search range"));
    }
    let delays: Vec<f64> = (0..window.delay_fft)
        .map(|j| {
            (j as i64 - window.delay_fft as i64 / 2) as f64 / (window.delay_fft as f64 * window.df)
        })
        .collect();
    let permitted: Vec<usize> = delays
        .iter()
        .enumerate()
        .filter(|&(_, &d)| in_range(d * window.sampling_hz, &args.drange))
        .map(|(j, _)| j)
        .collect();
    if permitted.is_empty() {
        return Err(invalid("empty MBCOR delay search range"));
    }
    let delay_plan = cached_fft_plan(window.delay_fft, false);
    let scratch_len = delay_plan.get_inplace_scratch_len();
    let spectrum_at_rate = |rate: f64, spectrum: &mut [C]| {
        spectrum.fill(C::new(0.0, 0.0));
        for ((transform, &rf), &bin) in transforms.iter().zip(&window.frequency).zip(&window.bins) {
            spectrum[bin] = centered_sample(
                transform,
                rate / rate_step * rf / window.reference,
                (window.rows - 1) as f64 / 2.0,
            );
        }
    };
    let peaks: Vec<(usize, f64)> = pool.install(|| {
        rates
            .par_iter()
            .map_init(
                || {
                    (
                        vec![C::new(0.0, 0.0); window.delay_fft],
                        vec![C::new(0.0, 0.0); scratch_len],
                    )
                },
                |(spectrum, scratch), &rate| {
                    spectrum_at_rate(rate, spectrum);
                    delay_plan.process_with_scratch(spectrum, scratch);
                    let mut best = (permitted[0], 0.0);
                    for &j in &permitted {
                        let amp = spectrum[(j + window.delay_fft / 2) % window.delay_fft].norm()
                            as f64
                            / window.count as f64;
                        if amp > best.1 {
                            best = (j, amp);
                        }
                    }
                    best
                },
            )
            .collect()
    });
    let rate_amplitude: Vec<f64> = peaks.iter().map(|p| p.1).collect();
    let rate_index = peaks
        .iter()
        .enumerate()
        .max_by(|a, b| a.1 .1.total_cmp(&b.1 .1))
        .unwrap()
        .0;
    let mut delay = delays[peaks[rate_index].0];
    let mut rate = rates[rate_index];
    let mut value = window.evaluate(delay, rate);
    // Re-evaluate and refine with exact RF/time phasors, without FFT interpolation.
    let mut d_step = 1.0 / (window.delay_fft as f64 * window.df);
    let mut r_step = rate_step;
    for _ in 0..args.iter.max(1) {
        let mut best = (delay, rate, value);
        for di in -2..=2 {
            for ri in -2..=2 {
                let d = delay + di as f64 * d_step / 2.0;
                let r = rate + ri as f64 * r_step / 2.0;
                if d.abs() >= 0.5 / window.df
                    || r.abs() > rate_limit as f64 * rate_step
                    || !in_range(d * window.sampling_hz, &args.drange)
                    || !in_range(r, &args.rrange)
                {
                    continue;
                }
                let v = window.evaluate(d, r);
                if v.norm_sqr() > best.2.norm_sqr() {
                    best = (d, r, v);
                }
            }
        }
        (delay, rate, value) = best;
        d_step /= 2.0;
        r_step /= 2.0;
    }
    let mut spectrum = vec![C::new(0.0, 0.0); window.delay_fft];
    spectrum_at_rate(rate, &mut spectrum);
    delay_plan.process(&mut spectrum);
    let delay_amplitude = (0..window.delay_fft)
        .map(|j| {
            spectrum[(j + window.delay_fft / 2) % window.delay_fft].norm() as f64
                / window.count as f64
        })
        .collect();
    Ok(Fringe {
        delay,
        rate,
        value,
        delays,
        delay_amplitude,
        rates,
        rate_amplitude,
    })
}

fn line_plot(
    path: &Path,
    title: &str,
    x_label: &str,
    y_label: &str,
    series: &[(String, Vec<(f64, f64)>)],
) -> Result<()> {
    let points: Vec<_> = series
        .iter()
        .flat_map(|(_, p)| p.iter())
        .filter(|p| p.0.is_finite() && p.1.is_finite())
        .collect();
    if points.is_empty() {
        return Ok(());
    }
    let mut xmin = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let mut xmax = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let mut ymin = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let mut ymax = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    if xmin == xmax {
        xmin -= 0.5;
        xmax += 0.5;
    }
    let pad = ((ymax - ymin) * 0.08).max(ymax.abs() * 0.01).max(1e-12);
    ymin -= pad;
    ymax += pad;
    let root = BitMapBackend::new(path, (1500, 650)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 28))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(85)
        .build_cartesian_2d(xmin..xmax, ymin..ymax)?;
    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .x_labels(9)
        .y_labels(7)
        .label_style(("sans-serif", 18))
        .axis_desc_style(("sans-serif", 20))
        .light_line_style(TRANSPARENT)
        .y_label_formatter(&|value| {
            if ymax.abs().max(ymin.abs()) < 0.01 {
                format!("{value:.2e}")
            } else {
                format!("{value:.2}")
            }
        })
        .draw()?;
    for (i, (label, points)) in series.iter().enumerate() {
        let color = [BLUE, RED, GREEN, BLACK][i % 4];
        chart
            .draw_series(LineSeries::new(points.iter().copied(), color))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 25, y)], color));
    }
    chart
        .configure_series_labels()
        .label_font(("sans-serif", 18))
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    Ok(())
}

pub fn run_mbcor(args: &Args) -> Result<()> {
    let path = args.input.as_ref().ok_or("MBCOR requires --input")?;
    let data = open_input_data(path)?;
    let bytes = data.as_slice();
    let layout = Layout::parse(bytes)?;
    let (file_start, dt) = layout.time(bytes, 0)?;
    println!(
        "MBCOR v1: {} -- {}, source {}",
        layout.headers[0].station1_name,
        layout.headers[0].station2_name,
        layout.headers[0].source_name
    );
    for (b, h) in layout.headers.iter().enumerate() {
        println!(
            "  Band {}: {:.3}--{:.3} MHz, {} channels",
            b + 1,
            h.observing_frequency / 1e6,
            (h.observing_frequency + h.sampling_speed as f64 / 2.0) / 1e6,
            h.fft_point / 2
        );
    }
    println!(
        "  Occupied bandwidth: {:.3} MHz; reference RF: {:.3} MHz",
        layout.bandwidth_hz / 1e6,
        layout.reference_hz / 1e6
    );
    println!(
        "  {} sectors, {:.9} s/sector, start timestamp {:.9}",
        layout.rows, dt, file_start
    );
    if args.header {
        return Ok(());
    }
    let skip = args.skip as f64 / dt;
    if args.skip < 0 || (skip - skip.round()).abs() > 1e-5 {
        return Err(invalid(
            "MBCOR --skip must be an integer number of integration sectors",
        ));
    }
    let first = skip.round() as usize;
    if first >= layout.rows {
        return Err(invalid("MBCOR --skip exceeds file duration"));
    }
    let rows = if args.length == 0 {
        layout.rows - first
    } else {
        args.length as usize
    };
    if rows == 0 || args.length < 0 || args.loop_ < 1 {
        return Err(invalid("invalid MBCOR length/loop"));
    }
    if first
        .checked_add(
            rows.checked_mul(args.loop_ as usize)
                .ok_or("MBCOR window overflow")?,
        )
        .is_none_or(|n| n > layout.rows)
    {
        return Err(invalid(
            "MBCOR --length times --loop exceeds available sectors",
        ));
    }
    if !args
        .search
        .iter()
        .all(|s| matches!(s.as_str(), "peak" | "deep" | "deep2" | "coherent"))
    {
        return Err(invalid("MBCOR supports --search peak/deep/deep2/coherent; phase rate/acel fitting is not implemented"));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(if args.cpu == 0 {
            std::thread::available_parallelism()?.get()
        } else {
            args.cpu as usize
        })
        .build()?;
    let directory = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("frinZ")
        .join("mbcor");
    fs::create_dir_all(&directory)?;
    let stem = output_stem_from_path(path)?;
    let result_path = directory.join(format!("{stem}_joint.tsv"));
    let mut results = BufWriter::new(fs::File::create(&result_path)?);
    writeln!(
        results,
        "# frinZ {}: true RF coordinates; common rate at reference_hz; no IF phase fitting",
        env!("CARGO_PKG_VERSION")
    )?;
    writeln!(results,"# sigma_diff is a channel/time first-difference quadrature noise proxy; includes phase/source variations and assumes independent channel/time noise.")?;
    writeln!(results,"# It is not the legacy delay-plane SNR estimator. Calibration-source statistics are not an independent faint-target sensitivity test.")?;
    writeln!(results,"window\tstart_timestamp\tduration_s\treference_hz\toccupied_bandwidth_hz\tusable_bandwidth_hz\tdelay_ns\tdelay_low_band_samples\trate_hz\tdelay_rate_s_per_s\tamplitude\tphase_deg\tsigma_diff\tsnr_diff\tband1_amplitude\tband1_phase_deg\tband2_amplitude\tband2_phase_deg")?;
    let mut loop_phase = Vec::new();
    for loop_index in 0..args.loop_ as usize {
        let window = Window::read(bytes, &layout, first + loop_index * rows, rows, args)?;
        println!(
            "Joint fringe window {}/{}: {} sectors ({:.3} s), {} occupied channels",
            loop_index + 1,
            args.loop_,
            rows,
            rows as f64 * dt,
            window.values.len()
        );
        let fringe = search(&window, args, &pool)?;
        let sigma = window.noise_proxy();
        let mut means = Vec::new();
        let mut band_sum = [Complex::<f64>::new(0.0, 0.0); 2];
        let mut band_count = [0; 2];
        for ((channel, &rf), &b) in window
            .values
            .iter()
            .zip(&window.frequency)
            .zip(&window.band)
        {
            let mut sum = Complex::<f64>::new(0.0, 0.0);
            let mut count = 0;
            for (n, &z) in channel.iter().enumerate() {
                let t = (n as f64 - (rows - 1) as f64 / 2.0) * dt;
                let phase = -TAU
                    * ((rf - window.reference) * fringe.delay
                        + rf / window.reference * fringe.rate * t);
                sum += Complex::new(z.re as f64, z.im as f64) * Complex::from_polar(1.0, phase);
                if z != C::new(0.0, 0.0) {
                    count += 1;
                }
            }
            means.push(if count > 0 {
                sum / count as f64
            } else {
                Complex::new(0.0, 0.0)
            });
            band_sum[b] += sum;
            band_count[b] += count;
        }
        let band_mean = [0, 1].map(|b| {
            if band_count[b] > 0 {
                band_sum[b] / band_count[b] as f64
            } else {
                Complex::new(0.0, 0.0)
            }
        });
        let usable = window.values.len() as f64 * window.df;
        writeln!(results,"{}\t{:.9}\t{:.9}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.9}\t{:.9}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.9}\t{:.9e}\t{:.6}\t{:.9e}\t{:.9}\t{:.9e}\t{:.9}",
            loop_index,window.start,rows as f64*dt,window.reference,layout.bandwidth_hz,usable,fringe.delay*1e9,
            fringe.delay*window.sampling_hz,fringe.rate,fringe.rate/window.reference,fringe.value.norm(),fringe.value.arg().to_degrees(),sigma,
            fringe.value.norm()/sigma,band_mean[0].norm(),band_mean[0].arg().to_degrees(),band_mean[1].norm(),band_mean[1].arg().to_degrees())?;
        println!("  Delay {:+.6} ns, rate {:+.9} Hz at {:.3} MHz, amplitude {:.6e}, phase {:+.4} deg, SNR(diff) {:.2}",
            fringe.delay*1e9,fringe.rate,window.reference/1e6,fringe.value.norm(),fringe.value.arg().to_degrees(),fringe.value.norm()/sigma);
        println!(
            "  Band phases {:+.4}/{:+.4} deg; RF gap excluded from {:.3} MHz usable bandwidth",
            band_mean[0].arg().to_degrees(),
            band_mean[1].arg().to_degrees(),
            usable / 1e6
        );
        let prefix = format!("{stem}_w{loop_index:04}");
        let spectrum_path = directory.join(format!("{prefix}_spectrum.tsv"));
        let mut spectra = BufWriter::new(fs::File::create(&spectrum_path)?);
        writeln!(
            spectra,
            "band\tfrequency_hz\treal\timag\tamplitude\tphase_deg"
        )?;
        for (j, &rf) in window.frequency.iter().enumerate() {
            writeln!(
                spectra,
                "{}\t{:.9}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.9}",
                window.band[j] + 1,
                rf,
                means[j].re,
                means[j].im,
                means[j].norm(),
                means[j].arg().to_degrees()
            )?;
        }
        spectra.flush()?;
        let mut time_mean = [Vec::new(), Vec::new(), Vec::new()];
        let times: Vec<f64> = (0..rows).map(|n| (n as f64 + 0.5) * dt).collect();
        let mut time_output = BufWriter::new(fs::File::create(
            directory.join(format!("{prefix}_time.tsv")),
        )?);
        writeln!(
            time_output,
            "timestamp\tband1_real\tband1_imag\tband2_real\tband2_imag\tjoint_real\tjoint_imag"
        )?;
        for (n, &t) in times.iter().enumerate() {
            let mut sum = [Complex::<f64>::new(0.0, 0.0); 2];
            let mut counts = [0; 2];
            for ((channel, &rf), &b) in window
                .values
                .iter()
                .zip(&window.frequency)
                .zip(&window.band)
            {
                let z = channel[n];
                let phase = -TAU
                    * ((rf - window.reference) * fringe.delay
                        + rf / window.reference * fringe.rate * (t - rows as f64 * dt / 2.0));
                sum[b] += Complex::new(z.re as f64, z.im as f64) * Complex::from_polar(1.0, phase);
                if z != C::new(0.0, 0.0) {
                    counts[b] += 1;
                }
            }
            let joint = (sum[0] + sum[1]) / (counts[0] + counts[1]).max(1) as f64;
            let means = [0, 1].map(|b| sum[b] / counts[b].max(1) as f64);
            time_mean[0].push(means[0]);
            time_mean[1].push(means[1]);
            time_mean[2].push(joint);
            writeln!(
                time_output,
                "{:.9}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.9e}",
                window.start + t,
                means[0].re,
                means[0].im,
                means[1].re,
                means[1].im,
                joint.re,
                joint.im
            )?;
        }
        time_output.flush()?;
        let labels = ["Band 1", "Band 2", "Joint"];
        line_plot(
            &directory.join(format!("{prefix}_phase.png")),
            &format!(
                "{}: {} -- {} joint fringe phase",
                layout.headers[0].source_name,
                layout.headers[0].station1_name,
                layout.headers[0].station2_name
            ),
            "Time since window start [s]",
            "Phase after joint delay/rate search [deg]",
            &(0..3)
                .map(|b| {
                    (
                        labels[b].to_owned(),
                        times
                            .iter()
                            .zip(&time_mean[b])
                            .map(|(&t, z)| (t, z.arg().to_degrees()))
                            .collect(),
                    )
                })
                .collect::<Vec<_>>(),
        )?;
        line_plot(
            &directory.join(format!("{prefix}_spectrum.png")),
            "Joint fringe spectrum (RF gap preserved)",
            "RF [MHz]",
            "Complex spectrum amplitude [COR units]",
            &(0..2)
                .map(|b| {
                    (
                        labels[b].to_owned(),
                        window
                            .frequency
                            .iter()
                            .enumerate()
                            .filter(|(j, _)| window.band[*j] == b)
                            .map(|(j, &rf)| (rf / 1e6, means[j].norm()))
                            .collect(),
                    )
                })
                .collect::<Vec<_>>(),
        )?;
        line_plot(
            &directory.join(format!("{prefix}_delay.png")),
            "Sparse-band delay response at selected common rate",
            "Delay [ns]",
            "Coherent amplitude [COR units]",
            &[(
                "Joint".to_owned(),
                fringe
                    .delays
                    .iter()
                    .zip(&fringe.delay_amplitude)
                    .filter(|&(d, _)| (*d - fringe.delay).abs() < 20e-9)
                    .map(|(&d, &a)| (d * 1e9, a))
                    .collect(),
            )],
        )?;
        line_plot(
            &directory.join(format!("{prefix}_rate.png")),
            "Common rate response (maximum over searched delay)",
            "Fringe rate at reference RF [Hz]",
            "Coherent amplitude [COR units]",
            &[(
                "Joint".to_owned(),
                fringe
                    .rates
                    .iter()
                    .zip(&fringe.rate_amplitude)
                    .filter(|&(r, _)| {
                        (*r - fringe.rate).abs() < 0.05_f64.max(3.0 / (rows as f64 * dt))
                    })
                    .map(|(&r, &a)| (r, a))
                    .collect(),
            )],
        )?;
        if args.npz || args.spectrum {
            let mut npz = NamedNpz::new(NpyMeta::new(
                "mbcor_joint",
                layout.headers[0].fft_point as u32,
                rows as u32,
            ));
            npz.add_f64_1d("frequency_hz", &window.frequency);
            npz.add_f64_1d("reference_hz", &[window.reference]);
            npz.add_f64_1d(
                "band_index",
                &window.band.iter().map(|&b| b as f64).collect::<Vec<_>>(),
            );
            npz.add_complex64_1d(
                "spectrum",
                &means
                    .iter()
                    .map(|z| C::new(z.re as f32, z.im as f32))
                    .collect::<Vec<_>>(),
            );
            npz.add_f64_1d("delay_s", &fringe.delays);
            npz.add_f64_1d("delay_amplitude", &fringe.delay_amplitude);
            npz.add_f64_1d("rate_hz", &fringe.rates);
            npz.add_f64_1d("rate_max_delay_amplitude", &fringe.rate_amplitude);
            npz.add_f64_1d(
                "time_timestamp",
                &times.iter().map(|t| window.start + t).collect::<Vec<_>>(),
            );
            npz.add_complex64_1d(
                "joint_time_visibility",
                &time_mean[2]
                    .iter()
                    .map(|z| C::new(z.re as f32, z.im as f32))
                    .collect::<Vec<_>>(),
            );
            npz.write(&directory.join(format!("{prefix}.npz")))?;
        }
        loop_phase.push((window.start - file_start, fringe.value.arg().to_degrees()));
    }
    results.flush()?;
    if args.add_plot && loop_phase.len() > 1 {
        line_plot(
            &directory.join(format!("{stem}_phase.png")),
            "Joint fringe window phases",
            "Window start since input start [s]",
            "Phase at window midpoint [deg]",
            &[("Joint".to_owned(), loop_phase)],
        )?;
    }
    println!("MBCOR joint results: {}", result_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn input(rows: usize, delay: f64, rate: f64) -> Vec<u8> {
        let mut bytes = vec![0u8; 544];
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&2u32.to_le_bytes());
        bytes[16..24].copy_from_slice(&1024e6f64.to_le_bytes());
        bytes[24..32].copy_from_slice(&7652e6f64.to_le_bytes());
        for (b, low) in [6600e6f64, 8192e6].iter().enumerate() {
            let h = &mut bytes[32 + b * 256..288 + b * 256];
            h[..4].copy_from_slice(&[0x83, 0xf9, 0xa2, 0x3e]);
            h[12..16].copy_from_slice(&1024000000i32.to_le_bytes());
            h[16..24].copy_from_slice(&low.to_le_bytes());
            h[24..28].copy_from_slice(&1024i32.to_le_bytes());
            h[28..32].copy_from_slice(&(rows as i32).to_le_bytes());
            h[32..40].copy_from_slice(b"YAMAGU32");
            h[80..88].copy_from_slice(b"YAMAGU34");
            h[128..135].copy_from_slice(b"TESTSRC");
        }
        for n in 0..rows {
            for low in [6600e6f64, 8192e6] {
                let mut sector = vec![0u8; 128];
                sector[..4].copy_from_slice(&(1000 + n as u32).to_le_bytes());
                sector[8..12].copy_from_slice(&(1001 + n as u32).to_le_bytes());
                sector[112..116].copy_from_slice(&1.0f32.to_le_bytes());
                bytes.extend(sector);
                for k in 0..512 {
                    let rf = low + k as f64 * 1e6;
                    let t = n as f64 - (rows - 1) as f64 / 2.0;
                    let z = C::from_polar(
                        1.0,
                        (TAU * ((rf - 7652e6) * delay + rf / 7652e6 * rate * t) + 0.3) as f32,
                    );
                    bytes.extend(z.re.to_le_bytes());
                    bytes.extend(z.im.to_le_bytes());
                }
            }
        }
        bytes
    }
    #[test]
    fn preserves_true_rf_gap_and_reference() {
        let data = input(4, 0.0, 0.0);
        let l = Layout::parse(&data).unwrap();
        let w = Window::read(&data, &l, 0, 4, &Args::default()).unwrap();
        assert_eq!(w.frequency.len(), 1024);
        assert_eq!(w.frequency[512] - w.frequency[511], 1081e6);
        assert_eq!(w.bins[512], 1592);
        assert_eq!(w.delay_fft, 4096);
        assert_eq!(w.reference, 7652e6);
        assert!((w.evaluate(0.0, 0.0).norm() - 1.0).abs() < 1e-6);
    }
    #[test]
    fn recovers_common_delay_rate_with_rf_scaling_and_phase() {
        let (delay, rate) = (3.371e-9, 0.081234);
        let data = input(32, delay, rate);
        let l = Layout::parse(&data).unwrap();
        let args = Args {
            iter: 7,
            rate_padding: 8,
            ..Args::default()
        };
        let w = Window::read(&data, &l, 0, 32, &args).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let found = search(&w, &args, &pool).unwrap();
        assert!((found.delay - delay).abs() < 2e-12, "delay {}", found.delay);
        assert!((found.rate - rate).abs() < 2e-5, "rate {}", found.rate);
        assert!((found.value.norm() - 1.0).abs() < 1e-5);
        assert!((found.value.arg() - 0.3).abs() < 1e-4);
    }
    #[test]
    fn rejects_truncated_and_wrong_version() {
        let mut data = input(4, 0.0, 0.0);
        data.pop();
        assert!(Layout::parse(&data).is_err());
        let mut data = input(4, 0.0, 0.0);
        data[8] = 2;
        assert!(Layout::parse(&data).is_err());
    }
    #[test]
    fn rejects_band_time_and_source_mismatch() {
        let mut data = input(4, 0.0, 0.0);
        let l = Layout::parse(&data).unwrap();
        data[l.offsets[1]] += 1;
        assert!(Window::read(&data, &l, 0, 4, &Args::default()).is_err());
        let mut data = input(4, 0.0, 0.0);
        data[288 + 128] = b'X';
        assert!(Layout::parse(&data).is_err());
    }
    #[test]
    fn never_fits_out_target_if_phase() {
        let mut data = input(4, 0.0, 0.0);
        let l = Layout::parse(&data).unwrap();
        for n in 0..4 {
            let offset = l.offsets[1] + n * l.stride + 128;
            for k in 0..512 {
                let o = offset + k * 8;
                let z = C::new(
                    f32::from_le_bytes(data[o..o + 4].try_into().unwrap()),
                    f32::from_le_bytes(data[o + 4..o + 8].try_into().unwrap()),
                ) * C::new(0.0, 1.0);
                data[o..o + 4].copy_from_slice(&z.re.to_le_bytes());
                data[o + 4..o + 8].copy_from_slice(&z.im.to_le_bytes());
            }
        }
        let w = Window::read(&data, &l, 0, 4, &Args::default()).unwrap();
        assert!((w.evaluate(0.0, 0.0).norm() - 0.5f64.sqrt()).abs() < 1e-6);
    }
    #[test]
    fn flags_rf_gap_without_changing_occupied_noise_weights() {
        let data = input(4, 0.0, 0.0);
        let l = Layout::parse(&data).unwrap();
        let args = Args {
            rfi: vec!["512,1592".to_owned()],
            ..Args::default()
        };
        let w = Window::read(&data, &l, 0, 4, &args).unwrap();
        assert_eq!(w.count, 4096);
        assert_eq!(w.values.len(), 1024);
        let args = Args {
            rfi: vec!["0,512".to_owned()],
            ..Args::default()
        };
        let w = Window::read(&data, &l, 0, 4, &args).unwrap();
        assert_eq!(w.values.len(), 512);
        assert!(w.band.iter().all(|&b| b == 1));
        assert!((w.evaluate(0.0, 0.0).norm() - 1.0).abs() < 1e-6);
    }
    #[test]
    fn first_difference_noise_has_correct_quadrature_normalization() {
        let data = input(64, 0.0, 0.0);
        let l = Layout::parse(&data).unwrap();
        let mut w = Window::read(&data, &l, 0, 64, &Args::default()).unwrap();
        let mut state = 17u64;
        let mut noise = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 32) as u32 as f64 / u32::MAX as f64 - 0.5) as f32 * 0.2
        };
        for channel in &mut w.values {
            for z in channel {
                *z = C::new(1.0 + noise(), noise());
            }
        }
        let expected = 0.1 / (3.0 * w.count as f64).sqrt();
        assert!((w.noise_proxy() / expected - 1.0).abs() < 0.02);
    }

    #[test]
    fn rate_correction_uses_file_origin_across_windows() {
        let rate = 0.037;
        let data = input(32, 0.0, rate);
        let l = Layout::parse(&data).unwrap();
        let args = Args {
            rate_correct: rate as f32,
            ..Args::default()
        };
        let first = Window::read(&data, &l, 0, 16, &args).unwrap();
        let second = Window::read(&data, &l, 16, 16, &args).unwrap();
        assert!((first.evaluate(0.0, 0.0) - second.evaluate(0.0, 0.0)).norm() < 1e-5);
    }
}
