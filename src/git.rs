use crate::doctor::GIT_ALIASES;
use git2::Config;
use std::error::Error;
use std::fs;
use std::fs::{File, OpenOptions};
use std::io::Write;

pub fn setup() -> Result<(), Box<dyn Error>> {
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
