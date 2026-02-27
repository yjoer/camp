use git2::Config;
use indexmap::IndexMap;
use owo_colors::OwoColorize;

#[cfg(target_os = "windows")]
mod windows_imports {
	pub use windows_registry::{CLASSES_ROOT, CURRENT_USER};
}

#[cfg(target_os = "windows")]
use windows_imports::*;

pub fn doctor() {
	#[cfg(target_os = "windows")]
	{
		println!("{}", "WINDOWS".bold());
		println!("{}", "Context Menu:".bold());
		for (k, v) in context_menu().iter() {
			println!("  {}: {}", k, v);
		}

		println!("\n{}", "Search:".bold());
		for (k, v) in search().iter() {
			println!("  {}: {}", k, v);
		}

		println!("\n{}", "Visual Studio Code:".bold());
		for (k, v) in code().iter() {
			println!("  {}: {}", k, v);
		}
	}

	println!("\n{}", "Git Config:".bold());
	for (k, v) in git_config().iter() {
		println!("  {}: {}", k, v);
	}
}

#[rustfmt::skip]
pub const GIT_ALIASES: [(&str, &str); 1] = [
	("sync", "!git fetch --prune && git rebase origin/master --autostash"),
];

#[cfg(target_os = "windows")]
fn context_menu() -> IndexMap<String, String> {
	// A &str has a static lifetime but does not persist long enough to be used by
	// another function. It is dropped at the end of this function.
	let mut map = IndexMap::<String, String>::new();

	let result = CURRENT_USER
		.open("Software\\Classes\\CLSID\\{86ca1aa0-34aa-4e8b-a509-50c905bae2a2}\\InprocServer32");

	match result {
		Ok(_) => map.insert("Classic Mode".to_string(), "Enabled".to_string()),
		Err(_) => map.insert("Classic Mode".to_string(), "Disabled".to_string()),
	};

	let result = CLASSES_ROOT
		.open("*\\shell\\pintohomefile")
		.and_then(|k| k.get_string("ProgrammaticAccessOnly"));

	match result {
		Ok(_) => map.insert("Add to Favorites".to_string(), "Disabled".to_string()),
		Err(_) => map.insert("Add to Favorites".to_string(), "Enabled".to_string()),
	};

	let result = CLASSES_ROOT
		.open("Folder\\shell\\pintohome")
		.and_then(|k| k.get_string("ProgrammaticAccessOnly"));

	match result {
		Ok(_) => map.insert("Pin to Quick Access".to_string(), "Disabled".to_string()),
		Err(_) => map.insert("Pin to Quick Access".to_string(), "Enabled".to_string()),
	};

	let result = CURRENT_USER
		.open("Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Advanced")
		.and_then(|k| k.get_u32("MaxUndoItems"));

	match result {
		Ok(v) => map.insert("Max Undo Items".to_string(), v.to_string()),
		Err(_) => map.insert("Max Undo Items".to_string(), "N/A".to_string()),
	};

	let path = "Software\\Microsoft\\Windows\\CurrentVersion\\Shell Extensions\\Blocked";

	let mut blocked_items = IndexMap::<&str, &str>::new();
	#[rustfmt::skip]
	blocked_items.insert("Work Folders Context Menu Handler", "{E61BF828-5E63-4287-BEF1-60B1A4FDE0E3}");
	blocked_items.insert("Move to OneDrive", "{1FA0E654-C9F2-4A1F-9800-B9A75D744B00}");
	blocked_items.insert("OneDrive", "{5250E46F-BB09-D602-5891-F476DC89B700}");
	blocked_items.insert("Pin to Start", "{470C0EBD-5D73-4d58-9CED-E91E22E23282}");
	#[rustfmt::skip]
	blocked_items.insert("Scan with Microsoft Defender", "{09A47860-11B0-4DA5-AFA5-26D86198A780}");
	blocked_items.insert("Share with Skype", "{776DBC8D-7347-478C-8D71-791E12EF49D8}");
	blocked_items.insert("Adobe Acrobat", "{A6595CD1-BF77-430A-A452-18696685F7C7}");
	blocked_items.insert("Upload to MEGA", "{0229E5E7-09E9-45CF-9228-0228EC7D5F17}");

	for (k, v) in blocked_items.iter() {
		let result = CURRENT_USER.open(path).and_then(|k| k.get_string(v));

		match result {
			Ok(_) => map.insert(k.to_string(), "Disabled".to_string()),
			Err(_) => map.insert(k.to_string(), "Enabled".to_string()),
		};
	}

	map
}

#[cfg(target_os = "windows")]
fn search() -> IndexMap<String, String> {
	let mut map = IndexMap::<String, String>::new();

	let result = CURRENT_USER
		.open("Software\\Policies\\Microsoft\\Windows\\Explorer")
		.and_then(|k| k.get_u32("DisableSearchBoxSuggestions"));

	match result {
		Ok(1) => map.insert("Web Search".to_string(), "Disabled".to_string()),
		Ok(_) => map.insert("Web Search".to_string(), "Enabled".to_string()),
		Err(_) => map.insert("Web Search".to_string(), "N/A".to_string()),
	};

	map
}

#[cfg(target_os = "windows")]
fn code() -> IndexMap<String, String> {
	let mut map = IndexMap::<String, String>::new();

	let result = CLASSES_ROOT
		.open("*\\shell\\VSCode WSL\\command")
		.and_then(|k| k.get_string(""));

	match result {
		Ok(v) => map.insert("Open File with WSL".to_string(), format!("Installed ({v})")),
		Err(_) => map.insert(
			"Open File with WSL".to_string(),
			"Not Installed".to_string(),
		),
	};

	let result = CLASSES_ROOT
		.open("Directory\\Background\\shell\\VSCode WSL\\command")
		.and_then(|k| k.get_string(""));

	match result {
		Ok(v) => map.insert(
			"Open Directory with WSL".to_string(),
			format!("Installed ({v})"),
		),
		Err(_) => map.insert(
			"Open Directory with WSL".to_string(),
			"Not Installed".to_string(),
		),
	};

	map
}

fn git_config() -> IndexMap<String, String> {
	let mut map = IndexMap::<String, String>::new();
	let cfg_default = Config::open_default().unwrap().snapshot().unwrap();

	let output = cfg_default.get_str("user.name").unwrap_or("N/A");
	map.insert("user.name".to_string(), output.to_string());

	let output = cfg_default.get_str("user.email").unwrap_or("N/A");
	map.insert("user.email".to_string(), output.to_string());

	let output = cfg_default.get_str("core.eol").unwrap_or("N/A");
	map.insert("core.eol".to_string(), output.to_string());

	let output = cfg_default.get_bool("core.safecrlf").unwrap_or(false);
	map.insert("core.safecrlf".to_string(), output.to_string());

	let output = cfg_default.get_str("core.autocrlf").unwrap_or("N/A");
	map.insert("core.autocrlf".to_string(), output.to_string());

	let output = cfg_default.get_str("core.attributesFile").unwrap_or("N/A");
	map.insert("core.attributesFile".to_string(), output.to_string());

	let output = cfg_default.get_bool("core.longpaths").unwrap_or(false);
	map.insert("core.longpaths".to_string(), output.to_string());

	let output = cfg_default.get_str("merge.conflictStyle").unwrap_or("N/A");
	map.insert("merge.conflictStyle".to_string(), output.to_string());

	let output = cfg_default.get_str("merge.mergiraf.name").unwrap_or("N/A");
	map.insert("merge.mergiraf.name".to_string(), output.to_string());

	let output = cfg_default
		.get_str("merge.mergiraf.driver")
		.unwrap_or("N/A");
	map.insert("merge.mergiraf.driver".to_string(), output.to_string());

	let output = cfg_default
		.get_bool("absorb.oneFixupPerCommit")
		.unwrap_or(false);
	map.insert("absorb.oneFixupPerCommit".to_string(), output.to_string());

	for (k, v) in &GIT_ALIASES {
		let output = cfg_default.get_str(&format!("alias.{}", k)).unwrap();

		if output == *v {
			map.insert(k.to_string(), format!("Enabled ({output})"));
		} else {
			map.insert(k.to_string(), format!("Disabled ({output})"));
		}
	}

	map
}
