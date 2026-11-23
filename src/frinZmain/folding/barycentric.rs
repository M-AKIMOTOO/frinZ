use chrono::{DateTime, Datelike, Utc};
use erfa::constants::{ERFA_AULT, ERFA_CMPS};
use erfa::earth::position_velocity_00;
use erfa::prenut::pn_matrix_06a;
use erfa::time::gst06;
use erfa::vectors_and_matrices::{inner_product, mat_mul_pvec};

const J2000_JD: f64 = 2_451_545.0;
const J2000_UNIX_S: f64 = 946_728_000.0;
const DAY_S: f64 = 86_400.0;
const ROTATION_RAD_PER_S: f64 = std::f64::consts::TAU * 1.002_737_811_911_354_6 / DAY_S;

/// Geometric barycentric light time for an ICRS direction and an ECEF site.
/// Earth orientation uses UT1≈UTC and omits polar motion.
pub(super) struct BarycentricRoemer {
    site_m: [f64; 3],
    direction: [f64; 3],
    anchor: Option<Anchor>,
}

struct Anchor {
    utc_unix_s: f64,
    earth_delay_s: f64,
    earth_delay_rate: f64,
    gst_rad: f64,
    direction_of_date: [f64; 3],
}

impl BarycentricRoemer {
    pub(super) fn new(site_m: [f64; 3], ra_rad: f64, dec_rad: f64) -> Self {
        let (sin_ra, cos_ra) = ra_rad.sin_cos();
        let (sin_dec, cos_dec) = dec_rad.sin_cos();
        Self {
            site_m,
            direction: [cos_ra * cos_dec, sin_ra * cos_dec, sin_dec],
            anchor: None,
        }
    }

    pub(super) fn delay(
        &mut self,
        sector_start: DateTime<Utc>,
        half_exposure_s: f64,
    ) -> Result<f64, &'static str> {
        let midpoint = sector_start.timestamp() as f64
            + f64::from(sector_start.timestamp_subsec_nanos()) * 1e-9
            + half_exposure_s;
        // At most 32 seconds from the ephemeris sample. The second-order
        // Earth-motion term over that interval is below the EPV00 accuracy.
        const INTERVAL_S: f64 = 64.0;
        let anchor_unix_s = (midpoint / INTERVAL_S).floor() * INTERVAL_S + INTERVAL_S * 0.5;
        if self
            .anchor
            .as_ref()
            .is_none_or(|anchor| anchor.utc_unix_s != anchor_unix_s)
        {
            let utc_days = (anchor_unix_s - J2000_UNIX_S) / DAY_S;
            let tt_days = utc_days + (tai_minus_utc(sector_start)? + 32.184) / DAY_S;
            // As allowed by EPV00, use TT in place of TDB in this ephemeris.
            let (out_of_range, _, [earth_pos, earth_vel]) = position_velocity_00(J2000_JD, tt_days);
            if out_of_range {
                return Err("folding barycentric ephemeris is outside the EPV00 date range");
            }
            let equinox = pn_matrix_06a(J2000_JD, tt_days);
            self.anchor = Some(Anchor {
                utc_unix_s: anchor_unix_s,
                earth_delay_s: inner_product(earth_pos, self.direction) * ERFA_AULT,
                earth_delay_rate: inner_product(earth_vel, self.direction) * ERFA_AULT / DAY_S,
                gst_rad: gst06(J2000_JD, utc_days, J2000_JD, tt_days, equinox),
                direction_of_date: mat_mul_pvec(equinox, self.direction),
            });
        }
        let anchor = self.anchor.as_ref().unwrap();
        let elapsed = midpoint - anchor.utc_unix_s;
        let (sin, cos) = (anchor.gst_rad + elapsed * ROTATION_RAD_PER_S).sin_cos();
        let [x, y, z] = self.site_m;
        let site_of_date = [cos * x - sin * y, sin * x + cos * y, z];
        Ok(anchor.earth_delay_s
            + elapsed * anchor.earth_delay_rate
            + inner_product(site_of_date, anchor.direction_of_date) / ERFA_CMPS)
    }
}

fn tai_minus_utc(epoch: DateTime<Utc>) -> Result<f64, &'static str> {
    // Post-1972 UTC steps from ERFA eraDat / IERS Bulletin C.
    const STEPS: &[(i32, u32)] = &[
        (1972, 1),
        (1972, 7),
        (1973, 1),
        (1974, 1),
        (1975, 1),
        (1976, 1),
        (1977, 1),
        (1978, 1),
        (1979, 1),
        (1980, 1),
        (1981, 7),
        (1982, 7),
        (1983, 7),
        (1985, 7),
        (1988, 1),
        (1990, 1),
        (1991, 1),
        (1992, 7),
        (1993, 7),
        (1994, 7),
        (1996, 1),
        (1997, 7),
        (1999, 1),
        (2006, 1),
        (2009, 1),
        (2012, 7),
        (2015, 7),
        (2017, 1),
    ];
    let steps = STEPS.partition_point(|&date| date <= (epoch.year(), epoch.month()));
    if steps == 0 {
        return Err("folding barycentric timing requires a UTC date in 1972 or later");
    }
    Ok(9.0 + steps as f64)
}
