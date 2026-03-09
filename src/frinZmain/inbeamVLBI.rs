// inbeamVLBI.rs
// In-beam VLBI workflow entry point:
// - uses the standard frinZ delay-rate fringe analysis
// - optionally runs rate/acel fitting when requested via --search
use crate::args::Args;
use crate::plot::{write_add_plot_outputs, write_cumulate_outputs};
use crate::processing::process_cor_file;
use crate::search::run_acel_search_analysis;
use crate::utils;
use chrono::{DateTime, Utc};
use std::error::Error;
use std::io;
use std::path::Path;

pub fn run_inbeam_vlbi_analysis(
    args: &Args,
    time_flag_ranges: &[(DateTime<Utc>, DateTime<Utc>)],
    pp_flag_ranges: &[(u32, u32)],
) -> Result<(), Box<dyn Error>> {
    let input_path = args.input.as_ref().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "--in-beam requires an --input file")
    })?;

    println!("#INFO: Running in-beam VLBI workflow (standard delay-rate fringe search).");

    let has_rate_search = args.search.iter().any(|mode| mode == "rate");
    let has_acel_search = args.search.iter().any(|mode| mode == "acel");
    let acel_only = !args.search.is_empty()
        && args
            .search
            .iter()
            .all(|mode| mode == "acel" || mode == "rate");

    if acel_only {
        if args.length == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "--search=acel/rate requires --len in --in-beam mode",
            )
            .into());
        }
        if args.loop_ == 1 {
            eprintln!(
                "Warning: --search=acel/rate is used, but --loop is 1. Multiple loops are usually needed for fitting."
            );
        }
        let mut degrees = Vec::new();
        if has_rate_search {
            degrees.push(1);
        }
        if has_acel_search {
            degrees.push(2);
        }
        return run_acel_search_analysis(args, &degrees, time_flag_ranges, pp_flag_ranges);
    }

    let result = process_cor_file(input_path, args, time_flag_ranges, pp_flag_ranges, false)?;
    let parent_dir = input_path.parent().unwrap_or_else(|| Path::new(""));
    let inbeam_dir = parent_dir.join("frinZ").join("inbeamVLBI");
    write_cumulate_outputs(args, &result, &inbeam_dir)?;
    let base_filename = write_add_plot_outputs(args, &result, &inbeam_dir)?;

    if args.allan_deviance {
        utils::write_allan_deviation_outputs(
            &result.add_plot_phase,
            args.length as f32,
            result.header.observing_frequency,
            &result.header.source_name,
            &base_filename,
            &inbeam_dir,
        )?;
    }

    if has_acel_search || has_rate_search {
        if args.length == 0 {
            eprintln!(
                "Warning: --search includes 'acel'/'rate' but --length is not specified. Skipping acceleration search."
            );
        } else {
            if args.loop_ == 1 {
                eprintln!(
                    "Warning: --search includes 'acel'/'rate' but --loop is 1. Results may be unreliable."
                );
            }
            let mut degrees = Vec::new();
            if has_rate_search {
                degrees.push(1);
            }
            if has_acel_search {
                degrees.push(2);
            }
            run_acel_search_analysis(args, &degrees, time_flag_ranges, pp_flag_ranges)?;
        }
    }

    Ok(())
}
