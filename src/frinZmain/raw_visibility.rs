use std::error::Error;
use std::fs;
use std::io::Cursor;
use std::path::Path;

use crate::args::Args;
use crate::fft::apply_phase_correction_in_place_at_frequency;
use crate::header::parse_header;
use crate::input_support::read_input_bytes;
use crate::npy_output::{npz_sidecar_path, NamedNpz, NpyMeta};
use crate::plot;
use crate::read::read_visibility_data;
use num_complex::Complex;
type C32 = Complex<f32>;

/// Executes the raw visibility plotting.
pub fn run_raw_visibility_plot(args: &Args) -> Result<(), Box<dyn Error>> {
    //println!("# Starting raw visibility plotting...");

    let input_path = args.input.as_ref().unwrap();

    // --- Create Output Directory ---
    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
    let output_dir = parent_dir.join("frinZ").join("rawvis");
    fs::create_dir_all(&output_dir)?;
    let base_filename = input_path.file_stem().unwrap().to_str().unwrap();

    let buffer = read_input_bytes(input_path)?;
    let mut cursor = Cursor::new(buffer.as_slice());

    let header = parse_header(&mut cursor)?;

    let mut all_spectra: Vec<Vec<C32>> = Vec::new();
    let mut effective_integ_time = 1.0f32;
    for l1 in 0..header.number_of_sector {
        let (complex_vec, _, effective) = match read_visibility_data(
            &mut cursor,
            &header,
            1,  // length in sectors
            0,  // skip in sectors
            l1, // loop_idx, which acts as sector index here
            false,
            &[], // Add empty pp_flag_ranges
        ) {
            Ok(data) => data,
            Err(_) => break, // Stop if we can't read more data
        };
        if complex_vec.is_empty() {
            eprintln!("Warning: Empty sector {} found, stopping read.", l1);
            break;
        }
        effective_integ_time = effective;
        all_spectra.push(complex_vec);
    }

    if all_spectra.is_empty() {
        eprintln!("No visibility data found in the file.");
        return Ok(());
    }

    let amp_heatmap_filepath = output_dir.join(format!("{}_rawvis_heatmap_amp.png", base_filename));
    let phase_heatmap_filepath =
        output_dir.join(format!("{}_rawvis_heatmap_phase.png", base_filename));
    for legacy_filename in [
        format!("{}_heatmap_amp.png", base_filename),
        format!("{}_heatmap_phase.png", base_filename),
        format!("{}_heatmap_amp_phase.png", base_filename),
        format!("{}_scatter_real_imag.png", base_filename),
        format!("{}_scatter_amp_phase.png", base_filename),
    ] {
        let _ = fs::remove_file(output_dir.join(legacy_filename));
    }

    // Use a default sigma of 0.0 for blurring, as in the original frinZrawvis.
    plot::plot_spectrum_amplitude_heatmap(&amp_heatmap_filepath, &all_spectra, 0.0)?;
    plot::plot_spectrum_phase_heatmap(&phase_heatmap_filepath, &all_spectra, 0.0)?;
    let rows = all_spectra.len();
    let cols = all_spectra.first().map_or(0, Vec::len);

    // --raw-visibility normally shows the unmodified visibility.  When a
    // manual delay/rate (or higher Taylor term) is supplied, also render the
    // visibility after applying exactly that correction so the time/frequency
    // effect can be inspected side by side with the original data.
    let has_manual_correction = args.delay_correct != 0.0
        || args.rate_correct != 0.0
        || args.acel_correct != 0.0
        || args.jerk_correct != 0.0
        || args.snap_correct != 0.0;
    if has_manual_correction
        && rows > 0
        && cols > 0
        && all_spectra.iter().all(|row| row.len() == cols)
    {
        let mut corrected_flat: Vec<C32> = all_spectra
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        apply_phase_correction_in_place_at_frequency(
            &mut corrected_flat,
            cols,
            args.rate_correct,
            args.delay_correct,
            args.acel_correct,
            args.jerk_correct,
            args.snap_correct,
            effective_integ_time,
            header.sampling_speed as u32,
            header.fft_point as u32,
            0.0,
            header.observing_frequency,
        );
        let corrected_spectra: Vec<Vec<C32>> = corrected_flat
            .chunks(cols)
            .map(|row| row.to_vec())
            .collect();

        let correction_kind = match (args.delay_correct != 0.0, args.rate_correct != 0.0) {
            (true, true) => "delay_rate",
            (true, false) => "delay",
            (false, true) => "rate",
            (false, false) => "taylor",
        };
        let corrected_amp_heatmap_filepath = output_dir.join(format!(
            "{}_rawvis_corrected_{}_heatmap_amp.png",
            base_filename, correction_kind
        ));
        let corrected_phase_heatmap_filepath = output_dir.join(format!(
            "{}_rawvis_corrected_{}_heatmap_phase.png",
            base_filename, correction_kind
        ));
        plot::plot_spectrum_amplitude_heatmap(
            &corrected_amp_heatmap_filepath,
            &corrected_spectra,
            0.0,
        )?;
        plot::plot_spectrum_phase_heatmap(
            &corrected_phase_heatmap_filepath,
            &corrected_spectra,
            0.0,
        )?;
        println!(
            "Corrected raw visibility amplitude plot: {:?}",
            corrected_amp_heatmap_filepath
        );
        println!(
            "Corrected raw visibility phase plot: {:?}",
            corrected_phase_heatmap_filepath
        );

        if args.npz {
            let correction_flag = format!("rawvis_corrected_{correction_kind}");
            let mut corrected_npz = NamedNpz::new(NpyMeta::new(
                &correction_flag,
                header.fft_point as u32,
                header.number_of_sector as u32,
            ));
            let time_axis: Vec<f64> = (0..rows).map(|index| index as f64).collect();
            let channel_axis: Vec<f64> = (0..cols).map(|index| index as f64).collect();
            corrected_npz.add_f64_1d("sector_index", &time_axis);
            corrected_npz.add_f64_1d("channel_index", &channel_axis);
            corrected_npz.add_complex64_2d(
                "visibility",
                (rows, cols),
                corrected_spectra.iter().flatten().copied(),
            )?;
            corrected_npz.write(&npz_sidecar_path(
                &corrected_phase_heatmap_filepath,
                &correction_flag,
            ))?;
        }
    }
    if args.npz && rows > 0 && cols > 0 && all_spectra.iter().all(|row| row.len() == cols) {
        let time_axis: Vec<f64> = (0..rows).map(|index| index as f64).collect();
        let channel_axis: Vec<f64> = (0..cols).map(|index| index as f64).collect();
        let mut npz = NamedNpz::new(NpyMeta::new(
            "rawvis",
            header.fft_point as u32,
            header.number_of_sector as u32,
        ));
        npz.add_f64_1d("sector_index", &time_axis);
        npz.add_f64_1d("channel_index", &channel_axis);
        npz.add_complex64_2d(
            "visibility",
            (rows, cols),
            all_spectra.iter().flatten().copied(),
        )?;
        npz.write(&npz_sidecar_path(&amp_heatmap_filepath, "rawvis"))?;
    }

    Ok(())
}
