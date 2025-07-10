use crate::doctor::GIT_ALIASES;
use git2::Config;
use std::error::Error;

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
    cfg_default.set_bool("core.longpaths", true)?;
    cfg_default.set_bool("absorb.oneFixupPerCommit", true)?;

    Ok(())
}

fn setup_git_aliases() -> Result<(), Box<dyn Error>> {
    let mut cfg_default = Config::open_default()?;

    for (k, v) in &GIT_ALIASES {
        cfg_default.set_str(&format!("alias.{}", k), v)?;
    }

    Ok(())
}
