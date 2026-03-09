use std::error::Error;
use std::fs;
use std::io::Cursor;
use std::io::Read;
use std::path::Path;

use crate::args::Args;
use crate::header::parse_header;
use crate::plot;
use crate::read::read_visibility_data;
use crate::utils::safe_arg;
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

    let mut file = fs::File::open(input_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
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

    let heatmap_filename = format!("{}_heatmap_amp_phase.png", base_filename);
    let scatter_filename = format!("{}_scatter_real_imag.png", base_filename);
    let amp_phase_filename = format!("{}_scatter_amp_phase.png", base_filename);
    let heatmap_filepath = output_dir.join(&heatmap_filename);
    let scatter_filepath = output_dir.join(&scatter_filename);
    let amp_phase_filepath = output_dir.join(&amp_phase_filename);

    // Use a default sigma of 0.0 for blurring, as in the original frinZrawvis.
    plot::plot_spectrum_heatmaps(&heatmap_filepath, &all_spectra, 0.0)?;

    let mut real_values = Vec::new();
    let mut imag_values = Vec::new();
    let mut amp_values = Vec::new();
    let mut phase_values = Vec::new();

    for spectra in &all_spectra {
        for value in spectra {
            real_values.push(value.re);
            imag_values.push(value.im);
            amp_values.push(value.norm());
            phase_values.push(safe_arg(value));
        }
    }

    plot::plot_complex_scatter(&scatter_filepath, &real_values, &imag_values)?;
    plot::plot_amp_phase_scatter(&amp_phase_filepath, &amp_values, &phase_values)?;
    //plot::plot_complex_histograms(
    //    &hist_filepath,
    //    &hist_report_filepath,
    //    &real_values,
    //    &imag_values,
    //    &amp_values,
    //    &phase_values,
    //)?;

    //println!("#Raw visibility heatmaps saved to {} and {}", amp_filepath.display(), phase_filepath.display());

    Ok(())
}
