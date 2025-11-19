use crate::doctor::GIT_ALIASES;
use git2::Config;
use inquire::MultiSelect;
use std::env;
use std::error::Error;
use std::fs;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::process::Command;

#[cfg(target_os = "windows")]
mod windows_imports {
    pub use windows_registry::LOCAL_MACHINE;
}

#[cfg(target_os = "windows")]
use windows_imports::*;

pub fn setup_git() -> Result<(), Box<dyn Error>> {
    setup_config()?;
    setup_git_aliases()?;

    Ok(())
}

fn setup_config() -> Result<(), Box<dyn Error>> {
    let mut cfg_default = Config::open_default()?;

    cfg_default.set_str("core.eol", "lf")?;
    cfg_default.set_bool("core.safecrlf", false)?;
    cfg_default.set_str("core.autocrlf", "input")?;
    cfg_default.set_str("core.attributesFile", "~/.gitattributes")?;
    cfg_default.set_bool("core.longpaths", true)?;

    cfg_default.set_str("merge.conflictStyle", "diff3")?;
    cfg_default.set_str("merge.mergiraf.name", "mergiraf")?;
    #[rustfmt::skip]
    cfg_default.set_str("merge.mergiraf.driver", "mergiraf merge --git %O %A %B -s %S -x %X -y %Y -p %P -l %L")?;

    cfg_default.set_bool("absorb.oneFixupPerCommit", true)?;

    if let Some(home_path) = std::env::home_dir() {
        let fp = home_path.join(".gitattributes");
        if !fp.exists() {
            File::create(&fp)?;
        }

        let content = fs::read_to_string(&fp)?;
        if !content.lines().any(|line| line.contains("merge=mergiraf")) {
            let mut file = OpenOptions::new().append(true).open(&fp)?;
            writeln!(file, "* merge=mergiraf")?;
        }
    }

    Ok(())
}

fn setup_git_aliases() -> Result<(), Box<dyn Error>> {
    let mut cfg_default = Config::open_default()?;

    for (k, v) in &GIT_ALIASES {
        cfg_default.set_str(&format!("alias.{}", k), v)?;
    }

    Ok(())
}

pub fn setup_jupyter(xcling_path: Option<String>) -> Result<(), Box<dyn Error>> {
    setup_jupyter_cling_kernel()?;
    setup_jupyter_xcling_kernel(xcling_path)?;
    Ok(())
}

fn setup_jupyter_cling_kernel() -> Result<(), Box<dyn Error>> {
    let td = match env::temp_dir().join("jupyter-cling") {
        ref dir if dir.exists() => dir.clone(),
        dir => {
            fs::create_dir_all(&dir)?;
            dir
        }
    };

    let repo_url = "https://github.com/root-project/root";
    let repo_path = td.join("root");
    let kernel_path = "interpreter/cling/tools/Jupyter/kernel";
    let repo_kernel_path = repo_path.join(kernel_path);

    if repo_path.exists() {
        fs::remove_dir_all(&repo_path)?;
    }

    Command::new("git")
        .arg("clone")
        .arg("--filter=blob:none")
        .arg("--no-checkout")
        .arg("--depth")
        .arg("1")
        .arg(repo_url)
        .arg(&repo_path)
        .spawn()?
        .wait()?;

    Command::new("git")
        .arg("-C")
        .arg(&repo_path)
        .arg("sparse-checkout")
        .arg("init")
        .arg("--cone")
        .spawn()?
        .wait()?;

    Command::new("git")
        .arg("-C")
        .arg(&repo_path)
        .arg("sparse-checkout")
        .arg("set")
        .arg(kernel_path)
        .spawn()?
        .wait()?;

    let mut cmd = Command::new("git");
    cmd.arg("-C").arg(&repo_path).arg("checkout");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if output.status.success() {
        println!("{}", stdout);
    } else {
        eprintln!("{}", stderr);
    }

    let mut cmd = Command::new("jupyter-kernelspec");
    cmd.arg("install")
        .arg("--user")
        .arg("cling-cpp20")
        .current_dir(&repo_kernel_path);

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if output.status.success() {
        println!("{}", stdout);
    } else {
        eprintln!("{}", stderr);
    }

    Ok(())
}

fn setup_jupyter_xcling_kernel(xcling_path: Option<String>) -> Result<(), Box<dyn Error>> {
    if xcling_path.is_none() {
        println!("XCling kernel setup was skipped because xcling-path was not provided.");
        return Ok(());
    }

    const XCLING_KERNEL: &str = include_str!("../examples/xeus/cling/kernels/xcling/kernel.json");
    let kernel = XCLING_KERNEL.replace("{program}", &xcling_path.unwrap());

    let td = match env::temp_dir().join("jupyter-xcling") {
        ref dir if dir.exists() => dir.clone(),
        dir => {
            fs::create_dir_all(&dir)?;
            dir
        }
    };

    if td.exists() {
        fs::remove_dir_all(&td)?;
    }

    let kernel_dir = td.join("xcling");
    fs::create_dir_all(&kernel_dir)?;

    let kernel_path = kernel_dir.join("kernel.json");
    fs::write(&kernel_path, kernel)?;

    let mut cmd = Command::new("jupyter-kernelspec");
    cmd.arg("install").arg("--user").arg(&kernel_dir);

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if output.status.success() {
        println!("{}", stdout);
    } else {
        eprintln!("{}", stderr);
    }

    Ok(())
}

#[cfg(target_os = "windows")]
pub fn setup_cpu_priority() -> Result<(), Box<dyn Error>> {
    let options = vec!["Microsoft Defender"];

    let selections = MultiSelect::new("Select programs to set up:", options).prompt()?;

    for selection in selections {
        match selection {
            "Microsoft Defender" => {
                let key = LOCAL_MACHINE.create("SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Image File Execution Options\\MsMpEng.exe\\PerfOptions")?;
                key.set_u32("CpuPriorityClass", 1)?;
            }
            _ => {}
        }
    }

    Ok(())
}

#[cfg(not(target_os = "windows"))]
pub fn setup_cpu_priority() -> Result<(), Box<dyn Error>> {
    Ok(())
}
