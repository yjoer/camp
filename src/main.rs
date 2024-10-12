#![cfg_attr(feature = "windows_subsystem", windows_subsystem = "windows")]

use clap::CommandFactory;
use clap::Parser;
use clap::Subcommand;
use regex::Regex;
use std::process::Command;
use windows::Win32::Foundation::HWND;
use windows::Win32::System::Console::GetConsoleWindow;
use windows::Win32::UI::WindowsAndMessaging::ShowWindow;
use windows::Win32::UI::WindowsAndMessaging::SW_HIDE;

#[cfg(target_os = "windows")]
mod windows_imports {
    pub use std::env::current_exe;
    pub use std::os::windows::process::CommandExt;
    pub use which::which;
    pub use windows_registry::CLASSES_ROOT;
}

#[cfg(target_os = "windows")]
use windows_imports::*;

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
    #[clap(about = "Work with Visual Studio Code.")]
    Code {
        #[command(subcommand)]
        subcommand: Option<CodeSubcommands>,
    },
}

#[derive(Subcommand, Debug)]
enum CodeSubcommands {
    #[clap(about = "Install the integrations.")]
    Install,

    #[clap(about = "Launch Visual Studio Code for files or folders within WSL.")]
    Launch { path: String },
}

fn hide_console_window() {
    let window = unsafe { GetConsoleWindow() };

    if window == HWND(0) {
        return;
    }

    unsafe {
        let _ = ShowWindow(window, SW_HIDE);
    }
}

fn main() {
    let cli = Args::parse();

    if cli.hide_console && cfg!(target_os = "windows") {
        hide_console_window();
    }

    match cli.command {
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
                            .expect("");
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
        None => {
            Args::command().print_help().unwrap();
        }
    }
}
