#![cfg_attr(feature = "windows_subsystem", windows_subsystem = "windows")]

use clap::CommandFactory;
use clap::Parser;
use clap::Subcommand;
use git2::Repository;
use regex::Regex;
use std::error::Error;
use std::process::Command;

#[cfg(target_os = "windows")]
mod windows_imports {
    pub use std::env::current_exe;
    pub use std::os::windows::process::CommandExt;
    pub use which::which;
    pub use windows::Win32::Foundation::HWND;
    pub use windows::Win32::System::Console::GetConsoleWindow;
    pub use windows::Win32::UI::WindowsAndMessaging::ShowWindow;
    pub use windows::Win32::UI::WindowsAndMessaging::SW_HIDE;
    pub use windows_registry::CLASSES_ROOT;
    pub use windows_registry::CURRENT_USER;
}

#[cfg(target_os = "windows")]
use windows_imports::*;

mod doctor;
mod setup;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,

    #[arg(short = None, long, help = "Hide the console window.")]
    hide_console: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[clap(about = "Work with the context menu.")]
    ContextMenu {
        #[command(subcommand)]
        subcommand: Option<ContextMenuSubcommands>,
    },
    #[clap(about = "Configure Windows Search settings.")]
    Search {
        #[command(subcommand)]
        subcommand: Option<SearchSubcommands>,
    },
    #[clap(about = "Work with Visual Studio Code.")]
    Code {
        #[command(subcommand)]
        subcommand: Option<CodeSubcommands>,
    },
    #[clap(about = "Set up Git config, aliases, and Jupyter kernels.")]
    Setup {
        #[command(subcommand)]
        subcommand: Option<SetupSubcommands>,
    },
    #[clap(about = "Apply a patch to the latest non-fixup commit automatically.")]
    Fixup,
    #[clap(about = "Squash fixup commits into their targets.")]
    Squash,
    #[clap(about = "Check installation status and diagnose possible problems.")]
    Doctor,
}

#[derive(Subcommand, Debug)]
enum ContextMenuSubcommands {
    #[clap(about = "Clean up the context menu.")]
    Clean,

    #[clap(about = "Use the legacy context menu.")]
    Legacy,
}

#[derive(Subcommand, Debug)]
enum SearchSubcommands {
    #[clap(about = "Disable web search in Windows Search.")]
    DisableWebSearch,
}

#[derive(Subcommand, Debug)]
enum CodeSubcommands {
    #[clap(about = "Install the integrations.")]
    Install,

    #[clap(about = "Launch Visual Studio Code for files or folders within WSL.")]
    Launch { path: String },
}

#[derive(Subcommand, Debug)]
enum SetupSubcommands {
    #[clap(about = "Set up common Git config and useful aliases for common operations.")]
    Git,

    #[clap(about = "Set up Jupyter kernels.")]
    Jupyter {
        #[arg(
            short,
            long,
            help = "Specify the path to the XCling kernel executable."
        )]
        xcling_path: Option<String>,
    },
}

#[cfg(target_os = "windows")]
fn hide_console_window() {
    let window = unsafe { GetConsoleWindow() };

    if window == HWND::default() {
        return;
    }

    unsafe {
        let _ = ShowWindow(window, SW_HIDE);
    }
}

fn main() {
    let cli = Args::parse();

    if cli.hide_console {
        #[cfg(target_os = "windows")]
        hide_console_window();
    }

    match cli.command {
        Some(Commands::ContextMenu { subcommand }) => match subcommand {
            Some(ContextMenuSubcommands::Clean) => {
                #[cfg(target_os = "windows")]
                {
                    // Add to Favorites
                    let key = CLASSES_ROOT.create("*\\shell\\pintohomefile").unwrap();
                    key.set_string("ProgrammaticAccessOnly", "").unwrap();

                    // Pin to Quick Access
                    let key = CLASSES_ROOT.create("Folder\\shell\\pintohome").unwrap();
                    key.set_string("ProgrammaticAccessOnly", "").unwrap();

                    #[rustfmt::skip]
                    let key = CURRENT_USER.create("Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Advanced").unwrap();
                    key.set_u32("MaxUndoItems", 0).unwrap();

                    let key = CURRENT_USER.create("Software\\Microsoft\\Windows\\CurrentVersion\\Shell Extensions\\Blocked").unwrap();

                    // Move to OneDrive
                    key.set_string("{1FA0E654-C9F2-4A1F-9800-B9A75D744B00}", "")
                        .unwrap();

                    // OneDrive
                    key.set_string("{5250E46F-BB09-D602-5891-F476DC89B700}", "")
                        .unwrap();

                    // Pin to Start
                    key.set_string("{470C0EBD-5D73-4d58-9CED-E91E22E23282}", "")
                        .unwrap();

                    // Scan with Microsoft Defender
                    key.set_string("{09A47860-11B0-4DA5-AFA5-26D86198A780}", "")
                        .unwrap();

                    // Share with Skype
                    key.set_string("{776DBC8D-7347-478C-8D71-791E12EF49D8}", "")
                        .unwrap();

                    // Adobe Acrobat
                    key.set_string("{A6595CD1-BF77-430A-A452-18696685F7C7}", "")
                        .unwrap();

                    // Upload to MEGA
                    key.set_string("{0229E5E7-09E9-45CF-9228-0228EC7D5F17}", "")
                        .unwrap();
                }
            }
            Some(ContextMenuSubcommands::Legacy) => {
                #[cfg(target_os = "windows")]
                {
                    let key = CURRENT_USER.create("Software\\Classes\\CLSID\\{86ca1aa0-34aa-4e8b-a509-50c905bae2a2}\\InprocServer32").unwrap();
                    key.set_string("", "").unwrap();
                }
            }
            None => {
                Args::command()
                    .find_subcommand_mut("context-menu")
                    .unwrap()
                    .print_help()
                    .unwrap();
            }
        },
        Some(Commands::Search { subcommand }) => match subcommand {
            Some(SearchSubcommands::DisableWebSearch) => disable_web_search(),
            None => {
                Args::command()
                    .find_subcommand_mut("search")
                    .unwrap()
                    .print_help()
                    .unwrap();
            }
        },
        Some(Commands::Code { subcommand }) => match subcommand {
            Some(CodeSubcommands::Install) => {
                #[cfg(target_os = "windows")]
                {
                    let menu_text = "Open with Code (WSL)";
                    let code_path = which("code")
                        .unwrap()
                        .parent()
                        .and_then(|p| p.parent())
                        .map(|p| p.join("Code.exe"))
                        .unwrap();
                    let code_path_str = code_path.to_str().unwrap();
                    let exe_path = current_exe()
                        .unwrap()
                        .parent()
                        .map(|p| p.join("camp-nw.exe"))
                        .unwrap();
                    let exe_path_str = exe_path.to_str().unwrap();

                    let f_path = "*\\shell\\VSCode WSL";
                    let f_key = CLASSES_ROOT.create(f_path).unwrap();
                    f_key.set_string("", menu_text).unwrap();
                    f_key.set_string("Icon", code_path_str).unwrap();

                    let f_cmd_key = f_key.create("command").unwrap();
                    let f_cmd = format!("\"{exe_path_str}\" code launch \"%1\"");
                    f_cmd_key.set_string("", f_cmd.as_str()).unwrap();

                    let bg_path = "Directory\\Background\\shell\\VSCode WSL";
                    let bg_key = CLASSES_ROOT.create(bg_path).unwrap();
                    bg_key.set_string("", menu_text).unwrap();
                    bg_key.set_string("Icon", code_path_str).unwrap();

                    let bg_cmd_key = bg_key.create("command").unwrap();
                    let bg_cmd = format!("\"{exe_path_str}\" code launch \"%V\"");
                    bg_cmd_key.set_string("", bg_cmd.as_str()).unwrap();
                }
            }
            Some(CodeSubcommands::Launch { path }) => {
                if !path.starts_with("\\\\wsl$") {
                    return;
                }

                let pattern = r"\\\\wsl\$\\(.*?)(\\.*)";
                let re = Regex::new(pattern).unwrap();

                if let Some(captures) = re.captures(&path) {
                    let distro_name = captures.get(1).unwrap().as_str();
                    let path = captures.get(2).unwrap().as_str();

                    let remote = format!("wsl+{}", distro_name);
                    let new_path = path.replace("\\", "/");

                    let program = if cfg!(target_os = "windows") {
                        "code.cmd"
                    } else if cfg!(target_os = "linux") {
                        "code"
                    } else {
                        ""
                    };

                    if program.is_empty() {
                        return;
                    }

                    #[cfg(target_os = "windows")]
                    {
                        const CREATE_NO_WINDOW: u32 = 0x08000000;

                        Command::new(program)
                            .arg("--remote")
                            .arg(&remote)
                            .arg(&new_path)
                            .creation_flags(CREATE_NO_WINDOW)
                            .spawn()
                            .unwrap()
                            .wait()
                            .unwrap();
                    }

                    #[cfg(not(target_os = "windows"))]
                    Command::new(program)
                        .arg("--remote")
                        .arg(&remote)
                        .arg(&new_path)
                        .spawn()
                        .expect("");
                }
            }
            None => {
                Args::command()
                    .find_subcommand_mut("code")
                    .unwrap()
                    .print_help()
                    .unwrap();
            }
        },
        Some(Commands::Setup { subcommand }) => match subcommand {
            Some(SetupSubcommands::Git) => setup::setup_git().unwrap(),
            Some(SetupSubcommands::Jupyter { xcling_path }) => {
                setup::setup_jupyter(xcling_path).unwrap()
            }
            None => {
                Args::command()
                    .find_subcommand_mut("setup")
                    .unwrap()
                    .print_help()
                    .unwrap();
            }
        },
        Some(Commands::Fixup) => fixup().unwrap(),
        Some(Commands::Squash) => squash().unwrap(),
        Some(Commands::Doctor) => doctor::doctor(),
        None => {
            Args::command().print_help().unwrap();
        }
    }
}

fn disable_web_search() {
    #[cfg(target_os = "windows")]
    {
        CURRENT_USER
            .create("Software\\Policies\\Microsoft\\Windows\\Explorer")
            .and_then(|k| k.set_u32("DisableSearchBoxSuggestions", 1))
            .unwrap();
    }
}

fn fixup() -> Result<(), Box<dyn Error>> {
    let cwd = std::env::current_dir()?;
    let repo = Repository::discover(&cwd)?;
    let mut revwalk = repo.revwalk()?;

    let head_oid = repo.refname_to_id("HEAD")?;
    revwalk.push(head_oid)?;

    if let Ok(tail_oid) = repo.refname_to_id("refs/remotes/origin/master") {
        revwalk.hide(tail_oid)?;
    } else if let Ok(tail_oid) = repo.refname_to_id("refs/remotes/origin/main") {
        revwalk.hide(tail_oid)?;
    };

    for rev in revwalk {
        let oid = rev?;
        let commit = repo.find_commit(oid)?;
        let message = commit.message().unwrap_or("");

        if !message.starts_with("fixup!") {
            let sig = repo.signature()?;
            let head_commit = repo.find_commit(head_oid)?;
            let head_tree = head_commit.tree()?;

            let mut index = repo.index()?;
            let tree_oid = index.write_tree()?;
            let tree = repo.find_tree(tree_oid)?;

            if head_tree.id() == tree.id() {
                println!("No changes to commit.");
                break;
            }

            let commit_oid = repo.commit(
                Some("HEAD"),
                &sig,
                &sig,
                &format!("fixup! {}", message),
                &tree,
                &vec![&head_commit],
            )?;

            let commit = repo.find_commit(commit_oid)?;
            let summary = commit.summary().unwrap_or("");
            println!("{}", summary);
            break;
        }
    }

    Ok(())
}

fn squash() -> Result<(), Box<dyn Error>> {
    let cwd = std::env::current_dir()?;
    let repo = Repository::discover(cwd)?;

    let mut cmd = Command::new("git");
    cmd.arg("rebase").arg("--autosquash").arg("--autostash");

    if let Ok(_) = repo.refname_to_id("refs/remotes/origin/master") {
        cmd.arg("origin/master");
    } else if let Ok(_) = repo.refname_to_id("refs/remotes/origin/main") {
        cmd.arg("origin/main");
    } else {
        cmd.arg("--root");
    };

    let output = cmd.output()?;
    let mut success = output.status.success();
    let mut stderr = String::from_utf8_lossy(&output.stderr).to_string();

    // If a fixup to a commit results in the same commit as the previous one, the commit will be
    // empty. In this case, a manual override is needed to continue.
    while !success {
        if stderr.contains("--allow-empty") {
            println!("Empty commit detected. Continuing...");

            let mut cmd = Command::new("git");
            cmd.env("GIT_EDITOR", "true")
                .arg("rebase")
                .arg("--continue");

            let output = cmd.output()?;
            success = output.status.success();
            stderr = String::from_utf8_lossy(&output.stderr).to_string();
        } else {
            eprintln!("{}", stderr);
            break;
        }
    }

    Ok(())
}
