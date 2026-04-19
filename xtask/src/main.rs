use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let Some(command) = args.next() else {
        print_usage();
        return Ok(());
    };

    let repo_root = repo_root()?;

    match command.to_string_lossy().as_ref() {
        "build" => {
            let cargo_args: Vec<OsString> = args.collect();
            let release = cargo_args.iter().any(|arg| arg == "--release");
            let profile = if release { "release" } else { "debug" };

            run_cargo_build(&cargo_args, &repo_root)?;
            copy_versioned_binary(&repo_root, profile, "frinZ", &repo_root.join("Cargo.toml"))?;
            copy_versioned_binary(
                &repo_root,
                profile,
                "gfrinZ",
                &repo_root.join("tools").join("Cargo.toml"),
            )?;
        }
        "install" => {
            let Some(bin_name) = args.next() else {
                print_usage();
                return Err("install command requires a binary name".into());
            };
            let bin_name = bin_name.to_string_lossy();
            install_versioned_binary(&repo_root, &bin_name)?;
        }
        _ => {
            print_usage();
            return Err(format!("unknown xtask command: {}", command.to_string_lossy()).into());
        }
    }

    Ok(())
}

fn print_usage() {
    eprintln!("usage: cargo build-versioned [cargo build options]");
    eprintln!("example: cargo build-versioned --release");
    eprintln!("usage: cargo frinZ");
    eprintln!("usage: cargo gfrinZ");
}

fn repo_root() -> Result<PathBuf, Box<dyn Error>> {
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR")
            .ok_or("CARGO_MANIFEST_DIR is not set; run this through cargo build-versioned")?,
    );
    Ok(manifest_dir
        .parent()
        .ok_or("xtask manifest directory has no parent")?
        .to_path_buf())
}

fn run_cargo_build(args: &[OsString], repo_root: &Path) -> Result<(), Box<dyn Error>> {
    let mut command = Command::new(env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo")));
    command
        .arg("build")
        .arg("--workspace")
        .arg("--exclude")
        .arg("xtask")
        .args(args)
        .current_dir(repo_root);

    let status = command.status()?;
    if !status.success() {
        return Err(format!("cargo build failed with status {status}").into());
    }
    Ok(())
}

fn install_versioned_binary(repo_root: &Path, bin_name: &str) -> Result<(), Box<dyn Error>> {
    let (manifest_path, install_path) = match bin_name {
        "frinZ" => (repo_root.join("Cargo.toml"), repo_root.to_path_buf()),
        "gfrinZ" => (
            repo_root.join("tools").join("Cargo.toml"),
            repo_root.join("tools"),
        ),
        other => return Err(format!("unsupported binary for versioned install: {other}").into()),
    };

    let version = package_version(&manifest_path)?;
    run_cargo_install(bin_name, &install_path)?;

    let bin_dir = cargo_install_bin_dir()?;
    let executable = executable_name(bin_name);
    let versioned = executable_name(&format!("{bin_name}-{version}"));
    let source = bin_dir.join(executable);
    let destination = bin_dir.join(versioned);

    fs::copy(&source, &destination)?;
    println!("copied {} -> {}", source.display(), destination.display());
    Ok(())
}

fn run_cargo_install(bin_name: &str, path: &Path) -> Result<(), Box<dyn Error>> {
    let mut command = Command::new(env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo")));
    command
        .arg("install")
        .arg("--force")
        .arg("--bin")
        .arg(bin_name)
        .arg("--path")
        .arg(path);

    let status = command.status()?;
    if !status.success() {
        return Err(format!("cargo install failed with status {status}").into());
    }
    Ok(())
}

fn cargo_install_bin_dir() -> Result<PathBuf, Box<dyn Error>> {
    if let Some(root) = env::var_os("CARGO_INSTALL_ROOT") {
        return Ok(PathBuf::from(root).join("bin"));
    }
    if let Some(cargo_home) = env::var_os("CARGO_HOME") {
        return Ok(PathBuf::from(cargo_home).join("bin"));
    }

    Ok(home_dir()?.join(".cargo").join("bin"))
}

fn home_dir() -> Result<PathBuf, Box<dyn Error>> {
    if let Some(home) = env::var_os("HOME") {
        return Ok(PathBuf::from(home));
    }
    if let Some(profile) = env::var_os("USERPROFILE") {
        return Ok(PathBuf::from(profile));
    }
    match (env::var_os("HOMEDRIVE"), env::var_os("HOMEPATH")) {
        (Some(drive), Some(path)) => {
            let mut home = PathBuf::from(drive);
            home.push(path);
            Ok(home)
        }
        _ => Err("could not determine home directory".into()),
    }
}

fn copy_versioned_binary(
    repo_root: &Path,
    profile: &str,
    base_name: &str,
    manifest_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let version = package_version(manifest_path)?;
    let executable = executable_name(base_name);
    let versioned = executable_name(&format!("{base_name}-{version}"));
    let profile_dir = repo_root.join("target").join(profile);
    let source = profile_dir.join(executable);
    let destination = profile_dir.join(versioned);

    fs::copy(&source, &destination)?;
    println!("copied {} -> {}", source.display(), destination.display());
    Ok(())
}

fn executable_name(base_name: &str) -> String {
    if cfg!(windows) {
        format!("{base_name}.exe")
    } else {
        base_name.to_string()
    }
}

fn package_version(manifest_path: &Path) -> Result<String, Box<dyn Error>> {
    let manifest = fs::read_to_string(manifest_path)?;
    let mut in_package = false;

    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed == "[package]" {
            in_package = true;
            continue;
        }
        if in_package && trimmed.starts_with('[') {
            break;
        }
        if in_package && trimmed.starts_with("version") {
            let (_, value) = trimmed
                .split_once('=')
                .ok_or_else(|| format!("invalid version line in {}", manifest_path.display()))?;
            let version = value.trim().trim_matches('"');
            if !version.is_empty() {
                return Ok(version.to_string());
            }
        }
    }

    Err(format!("package version not found in {}", manifest_path.display()).into())
}
