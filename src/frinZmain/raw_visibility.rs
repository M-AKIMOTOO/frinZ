use std::error::Error;
use std::fs;
use std::io::Cursor;
use std::path::Path;

use crate::args::Args;
use crate::header::parse_header;
use crate::input_support::read_input_bytes;
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
    for l1 in 0..header.number_of_sector {
        let (complex_vec, _, _) = match read_visibility_data(
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
        all_spectra.push(complex_vec);
    }

    if all_spectra.is_empty() {
        eprintln!("No visibility data found in the file.");
        return Ok(());
    }

    let amp_heatmap_filepath = output_dir.join(format!("{}_heatmap_amp.png", base_filename));
    let phase_heatmap_filepath = output_dir.join(format!("{}_heatmap_phase.png", base_filename));
    for legacy_filename in [
        format!("{}_heatmap_amp_phase.png", base_filename),
        format!("{}_scatter_real_imag.png", base_filename),
        format!("{}_scatter_amp_phase.png", base_filename),
    ] {
        let _ = fs::remove_file(output_dir.join(legacy_filename));
    }

    // Use a default sigma of 0.0 for blurring, as in the original frinZrawvis.
    plot::plot_spectrum_amplitude_heatmap(&amp_heatmap_filepath, &all_spectra, 0.0)?;
    plot::plot_spectrum_phase_heatmap(&phase_heatmap_filepath, &all_spectra, 0.0)?;

    Ok(())
}
