use anyhow::{ensure, Context, Result};
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

const ICON: &str = include_str!("icon.svg");

// Desktop entry values are unescaped before Exec quoting is processed.
// https://specifications.freedesktop.org/desktop-entry/latest/exec-variables.html
fn executable_value(executable: &Path) -> Result<String> {
    let path = executable
        .to_str()
        .context("The executable path must be UTF-8")?;
    ensure!(
        !path.contains(['\n', '\r', '=']),
        "The executable path contains a character unsupported by desktop entries"
    );
    let mut result = String::from("\"");
    for c in path.chars() {
        match c {
            '\\' => result.push_str("\\\\\\\\"),
            '"' | '`' | '$' => {
                result.push_str("\\\\");
                result.push(c);
            }
            '%' => result.push_str("%%"),
            _ => result.push(c),
        }
    }
    result.push('"');
    Ok(result)
}

fn install_at(executable: &Path, data_home: &Path) -> Result<PathBuf> {
    ensure!(executable.is_absolute(), "Use an absolute executable path");
    let executable = executable_value(executable)?;
    let applications = data_home.join("applications");
    let icons = data_home.join("icons/hicolor/scalable/apps");
    fs::create_dir_all(&applications)?;
    fs::create_dir_all(&icons)?;
    let desktop = applications.join("frinZtable.desktop");
    let entry = format!("[Desktop Entry]\nType=Application\nName=frinZtable\nGenericName=Fringe analysis spreadsheet\nGenericName[ja]=フリンジ解析結果の表・統計・グラフ\nComment=Explore frinZ TSV results, compare sources and export editable figures\nComment[ja]=frinZの解析結果を表で編集し、複数天体の統計とグラフを表示\nExec={executable} %f\nIcon=frinZtable\nTerminal=false\nCategories=Science;Astronomy;DataVisualization;\nKeywords=frinZ;TSV;VLBI;XRISM;statistics;spreadsheet;天体;統計;\nMimeType=text/tab-separated-values;text/plain;\nStartupNotify=false\nStartupWMClass=frinZtable\nActions=Help;\n\n[Desktop Action Help]\nName=frinZmain processing help\nName[ja]=frinZmainの処理一覧・仕様書\nExec={executable} --manual\n");
    fs::write(icons.join("frinZtable.svg"), ICON)?;
    fs::write(&desktop, entry)?;
    Ok(desktop)
}

pub fn install() -> Result<PathBuf> {
    ensure!(
        cfg!(target_os = "linux"),
        "Desktop application registration is supported on Linux"
    );
    let data_home = env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| env::var_os("HOME").map(|p| PathBuf::from(p).join(".local/share")))
        .context("HOME or an absolute XDG_DATA_HOME must be set")?;
    let executable = env::current_exe()?.canonicalize()?;
    let desktop = install_at(&executable, &data_home)?;
    match Command::new("update-desktop-database")
        .arg(desktop.parent().unwrap())
        .output()
    {
        Ok(output) if !output.status.success() => eprintln!(
            "Application registered; MIME cache update failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ),
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
            eprintln!("Application registered; MIME cache update failed: {error}")
        }
        _ => {}
    }
    Ok(desktop)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn desktop_registration_uses_actual_binary_and_single_file_handler() {
        let directory = tempfile::tempdir().unwrap();
        let path = install_at(Path::new("/tmp/共同研究/frinZ table"), directory.path()).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("Exec=\"/tmp/共同研究/frinZ table\" %f\n"));
        assert!(text.contains("Exec=\"/tmp/共同研究/frinZ table\" --manual"));
        assert!(text.contains("Terminal=false"));
        assert!(directory
            .path()
            .join("icons/hicolor/scalable/apps/frinZtable.svg")
            .exists());
        assert_eq!(
            executable_value(Path::new("/tmp/a$`\"\\%b")).unwrap(),
            concat!("\"/tmp/a", "\\\\$", "\\\\`", "\\\\\"", "\\\\\\\\", "%%b\"")
        );
        assert!(executable_value(Path::new("/tmp/a\nb")).is_err());
    }
}
