#![cfg_attr(feature = "windows_subsystem", windows_subsystem = "windows")]

use std::error::Error;
use std::fs;
use std::path::Path;
use std::process::Command;

use clap::{Parser, Subcommand};
use git2::Repository;
use inquire::MultiSelect;
use regex::Regex;
use reqwest;

#[cfg(target_os = "windows")]
mod windows_imports {
	pub use std::env::current_exe;
	pub use std::os::windows::process::CommandExt;

	pub use which::which;
	pub use windows::Win32::Foundation::HWND;
	pub use windows::Win32::System::Console::GetConsoleWindow;
	pub use windows::Win32::UI::WindowsAndMessaging::{ShowWindow, SW_HIDE};
	pub use windows_registry::{CLASSES_ROOT, CURRENT_USER};
}

#[cfg(target_os = "windows")]
use windows_imports::*;

mod doctor;
mod monitor;
mod setup;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
	#[command(subcommand)]
	command: Commands,

	#[arg(short = None, long, help = "Hide the console window.")]
	hide_console: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
	#[clap(about = "Work with the context menu.")]
	ContextMenu {
		#[command(subcommand)]
		subcommand: ContextMenuSubcommands,
	},
	#[clap(about = "Configure Windows Search settings.")]
	Search {
		#[command(subcommand)]
		subcommand: SearchSubcommands,
	},
	#[clap(about = "Work with Visual Studio Code.")]
	Code {
		#[command(subcommand)]
		subcommand: CodeSubcommands,
	},
	#[clap(about = "Set up Git config, aliases, and Jupyter kernels.")]
	Setup {
		#[command(subcommand)]
		subcommand: SetupSubcommands,
	},
	#[clap(
		about = "Convert commits by the specified author into fixup commits targeting their nearest ancestor by a different author."
	)]
	Fold {
		#[arg(short, long, help = "Name of the author whose commits to fold.")]
		name: String,
	},
	#[clap(about = "Apply a patch to the latest non-fixup commit automatically.")]
	Fixup,
	#[clap(about = "Squash fixup commits into their targets.")]
	Squash,
	#[clap(about = "Check installation status and diagnose possible problems.")]
	Doctor,
	#[clap(about = "Change the display brightness on selected applications.")]
	Monitor {
		#[command(subcommand)]
		subcommand: Option<MonitorSubcommands>,
	},
}

#[derive(Subcommand, Debug)]
enum ContextMenuSubcommands {
	#[clap(about = "Set up the context menu.")]
	Setup,

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

	#[clap(about = "Set up CPU priority.")]
	CPUPriority,
}

#[derive(Subcommand, Debug)]
enum MonitorSubcommands {
	#[clap(about = "Launch the monitor service.")]
	Service,

	#[clap(about = "Install the monitor service.")]
	Install,

	#[clap(about = "Uninstall the monitor service.")]
	Uninstall,
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
		Commands::ContextMenu { subcommand } => match subcommand {
			ContextMenuSubcommands::Setup => setup_context_menu().unwrap(),
			ContextMenuSubcommands::Clean => context_menu().unwrap(),
			ContextMenuSubcommands::Legacy => {
				#[cfg(target_os = "windows")]
				{
					let key = CURRENT_USER.create("Software\\Classes\\CLSID\\{86ca1aa0-34aa-4e8b-a509-50c905bae2a2}\\InprocServer32").unwrap();
					key.set_string("", "").unwrap();
				}
			}
		},
		Commands::Search { subcommand } => match subcommand {
			SearchSubcommands::DisableWebSearch => disable_web_search(),
		},
		Commands::Code { subcommand } => match subcommand {
			CodeSubcommands::Install => {
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
			CodeSubcommands::Launch { path } => {
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
		},
		Commands::Setup { subcommand } => match subcommand {
			SetupSubcommands::Git => setup::setup_git().unwrap(),
			SetupSubcommands::Jupyter { xcling_path } => setup::setup_jupyter(xcling_path).unwrap(),
			SetupSubcommands::CPUPriority => setup::setup_cpu_priority().unwrap(),
		},
		Commands::Fold { name } => fold(&name).unwrap(),
		Commands::Fixup => fixup().unwrap(),
		Commands::Squash => squash().unwrap(),
		Commands::Doctor => doctor::doctor(),
		Commands::Monitor { subcommand } => match subcommand {
			Some(MonitorSubcommands::Service) => monitor::start_monitor_service().unwrap(),
			Some(MonitorSubcommands::Install) => monitor::install_monitor_service().unwrap(),
			Some(MonitorSubcommands::Uninstall) => monitor::uninstall_monitor_service().unwrap(),
			None => monitor::monitor(),
		},
	}
}

#[cfg(target_os = "windows")]
fn setup_context_menu() -> Result<(), Box<dyn Error>> {
	let options = vec!["Windows Terminal", "WSL"];

	let selections = MultiSelect::new("Select context menu items to set up:", options).prompt()?;

	for selection in selections {
		match selection {
			"Windows Terminal" => {
				let local_appdata = std::env::var("LOCALAPPDATA")?;
				let icon_path = Path::new(&local_appdata).join("Microsoft\\WindowsApps\\wt.ico");
				fs::create_dir_all(icon_path.parent().unwrap())?;

				if !icon_path.exists() {
					let url = "https://raw.githubusercontent.com/microsoft/terminal/master/res/terminal.ico";
					let bytes = reqwest::blocking::get(url)?.bytes()?;
					fs::write(icon_path, &bytes)?;
				}

				let key = CLASSES_ROOT.create("Directory\\Background\\shell\\WindowsTerminal")?;
				key.set_expand_string("", "Open in Terminal")?;
				key.set_expand_string("Icon", "%LOCALAPPDATA%\\Microsoft\\WindowsApps\\wt.ico")?;

				let cmd_key = key.create("command")?;
				cmd_key.set_string("", "wt.exe -d \"%V\"")?;
			}
			"WSL" => {
				let key = CLASSES_ROOT.create("Directory\\background\\shell\\WSL")?;
				match key.remove_value("Extended") {
					Ok(_) => {}
					Err(_) => {}
				};
				key.set_expand_string("Icon", "wsl.exe")?;
			}
			_ => {}
		}
	}

	Ok(())
}

#[cfg(not(target_os = "windows"))]
fn setup_context_menu() -> Result<(), Box<dyn Error>> {
	Ok(())
}

#[cfg(target_os = "windows")]
fn context_menu() -> Result<(), Box<dyn Error>> {
	let options = vec![
		"Add to Favorites",
		"Pin to Quick Access",
		"Undo Items",
		"Work Folders Context Menu Handler",
		"Move to OneDrive",
		"OneDrive",
		"Pin to Start",
		"Scan with Microsoft Defender",
		"Share with Skype",
		"Adobe Acrobat",
		"Upload to MEGA",
	];

	let selections = MultiSelect::new("Select context menu items to disable:", options).prompt()?;

	for selection in selections {
		let blocked_key = CURRENT_USER
			.create("Software\\Microsoft\\Windows\\CurrentVersion\\Shell Extensions\\Blocked")?;

		match selection {
			"Add to Favorites" => {
				let key = CLASSES_ROOT.create("*\\shell\\pintohomefile")?;
				key.set_string("ProgrammaticAccessOnly", "")?;
			}
			"Pin to Quick Access" => {
				let key = CLASSES_ROOT.create("Folder\\shell\\pintohome")?;
				key.set_string("ProgrammaticAccessOnly", "")?;
			}
			"Undo Items" => {
				let key = CURRENT_USER
					.create("Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Advanced")?;
				key.set_u32("MaxUndoItems", 0)?;
			}
			"Work Folders Context Menu Handler" => {
				blocked_key.set_string("{E61BF828-5E63-4287-BEF1-60B1A4FDE0E3}", "")?;
			}
			"Move to OneDrive" => {
				blocked_key.set_string("{1FA0E654-C9F2-4A1F-9800-B9A75D744B00}", "")?;
			}
			"OneDrive" => {
				blocked_key.set_string("{5250E46F-BB09-D602-5891-F476DC89B700}", "")?;
			}
			"Pin to Start" => {
				blocked_key.set_string("{470C0EBD-5D73-4d58-9CED-E91E22E23282}", "")?;
			}
			"Scan with Microsoft Defender" => {
				blocked_key.set_string("{09A47860-11B0-4DA5-AFA5-26D86198A780}", "")?;
			}
			"Share with Skype" => {
				blocked_key.set_string("{776DBC8D-7347-478C-8D71-791E12EF49D8}", "")?;
			}
			"Adobe Acrobat" => {
				blocked_key.set_string("{A6595CD1-BF77-430A-A452-18696685F7C7}", "")?;
			}
			"Upload to MEGA" => {
				blocked_key.set_string("{0229E5E7-09E9-45CF-9228-0228EC7D5F17}", "")?;
			}
			_ => {}
		}
	}

	Ok(())
}

#[cfg(not(target_os = "windows"))]
fn context_menu() -> Result<(), Box<dyn Error>> {
	Ok(())
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

fn fold(name: &str) -> Result<(), Box<dyn Error>> {
	let cwd = std::env::current_dir()?;
	let repo = Repository::discover(&cwd)?;

	let head_oid = repo.refname_to_id("HEAD")?;
	let head_commit = repo.find_annotated_commit(head_oid)?;

	let mut revwalk = repo.revwalk()?;
	revwalk.push_head()?;
	revwalk.set_sorting(git2::Sort::TOPOLOGICAL | git2::Sort::REVERSE)?;
	let root_oid = revwalk.next().unwrap()?;
	let root_commit = repo.find_annotated_commit(root_oid)?;

	let mut rebase = repo.rebase(Some(&head_commit), Some(&root_commit), None, None)?;

	while let Some(op) = rebase.next() {
		let oid = op?.id();
		let commit = repo.find_commit(oid)?;
		let author = commit.author();
		let author_name = author.name().unwrap_or("");
		let committer = commit.committer();
		let mut message = commit.message().unwrap_or("").to_string();

		if author_name == name {
			let mut revwalk_ = repo.revwalk()?;
			revwalk_.push(oid)?;

			for rev_ in revwalk_ {
				let oid_ = rev_?;
				let commit_ = repo.find_commit(oid_)?;
				let author_ = commit_.author();
				let author_name_ = author_.name().unwrap_or("");

				if author_name_ != name {
					message = format!("fixup! {}", commit_.message().unwrap_or(""));
					break;
				}
			}
		}

		rebase.commit(None, &committer, Some(&message))?;
	}

	rebase.finish(None)?;
	Ok(())
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

	// If a fixup to a commit results in the same commit as the previous one, the
	// commit will be empty. In this case, a manual override is needed to continue.
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
