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
    pub use windows_registry::CURRENT_USER;
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
    #[clap(about = "Work with the context menu.")]
    ContextMenu {
        #[command(subcommand)]
        subcommand: Option<ContextMenuSubcommands>,
    },
    #[clap(about = "Work with Visual Studio Code.")]
    Code {
        #[command(subcommand)]
        subcommand: Option<CodeSubcommands>,
    },
}

#[derive(Subcommand, Debug)]
enum ContextMenuSubcommands {
    #[clap(about = "Clean up the context menu.")]
    Clean,

    #[clap(about = "Use the legacy context menu.")]
    Legacy,
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
        Some(Commands::ContextMenu { subcommand }) => match subcommand {
            Some(ContextMenuSubcommands::Clean) => {
                #[cfg(target_os = "windows")]
                {
                    // Add to Favorites
                    let key = CLASSES_ROOT.create("*\\shell\\pintohomefile").unwrap();
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
