use indexmap::IndexSet;
use std::ffi::c_void;
use std::ffi::OsString;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::ptr;
use std::slice;
use std::sync::mpsc;
use std::thread::sleep;
use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

#[cfg(target_os = "windows")]
mod windows_imports {
    pub use std::os::windows::ffi::OsStrExt;
    pub use std::os::windows::ffi::OsStringExt;
    pub use windows::core::BSTR;
    pub use windows::core::PWSTR;
    pub use windows::Win32::Devices::Display::DestroyPhysicalMonitors;
    pub use windows::Win32::Devices::Display::GetNumberOfPhysicalMonitorsFromHMONITOR;
    pub use windows::Win32::Devices::Display::GetPhysicalMonitorsFromHMONITOR;
    pub use windows::Win32::Devices::Display::SetMonitorBrightness;
    pub use windows::Win32::Devices::Display::PHYSICAL_MONITOR;
    pub use windows::Win32::Foundation::CloseHandle;
    pub use windows::Win32::Foundation::ERROR_SERVICE_DOES_NOT_EXIST;
    pub use windows::Win32::Foundation::HANDLE;
    pub use windows::Win32::Foundation::HMODULE;
    pub use windows::Win32::Foundation::HWND;
    pub use windows::Win32::Graphics::Gdi::MonitorFromWindow;
    pub use windows::Win32::Graphics::Gdi::MONITOR_DEFAULTTOPRIMARY;
    pub use windows::Win32::System::Com::CoCreateInstance;
    pub use windows::Win32::System::Com::CoInitializeEx;
    pub use windows::Win32::System::Com::CoSetProxyBlanket;
    pub use windows::Win32::System::Com::CoUninitialize;
    pub use windows::Win32::System::Com::CLSCTX_INPROC_SERVER;
    pub use windows::Win32::System::Com::COINIT_MULTITHREADED;
    pub use windows::Win32::System::Com::EOAC_NONE;
    pub use windows::Win32::System::Com::RPC_C_AUTHN_LEVEL_PKT;
    pub use windows::Win32::System::Com::RPC_C_IMP_LEVEL_IMPERSONATE;
    pub use windows::Win32::System::Environment::CreateEnvironmentBlock;
    pub use windows::Win32::System::Environment::DestroyEnvironmentBlock;
    pub use windows::Win32::System::ProcessStatus::GetModuleFileNameExW;
    pub use windows::Win32::System::RemoteDesktop::WTSActive;
    pub use windows::Win32::System::RemoteDesktop::WTSEnumerateSessionsW;
    pub use windows::Win32::System::RemoteDesktop::WTSFreeMemory;
    pub use windows::Win32::System::RemoteDesktop::WTSQueryUserToken;
    pub use windows::Win32::System::RemoteDesktop::WTS_SESSION_INFOW;
    pub use windows::Win32::System::Rpc::RPC_C_AUTHN_NONE;
    pub use windows::Win32::System::Rpc::RPC_C_AUTHN_WINNT;
    pub use windows::Win32::System::Threading::CreateProcessAsUserW;
    pub use windows::Win32::System::Threading::OpenProcess;
    pub use windows::Win32::System::Threading::TerminateProcess;
    pub use windows::Win32::System::Threading::CREATE_NO_WINDOW;
    pub use windows::Win32::System::Threading::CREATE_UNICODE_ENVIRONMENT;
    pub use windows::Win32::System::Threading::PROCESS_INFORMATION;
    pub use windows::Win32::System::Threading::PROCESS_QUERY_LIMITED_INFORMATION;
    pub use windows::Win32::System::Threading::STARTUPINFOW;
    pub use windows::Win32::System::Variant::VARIANT;
    pub use windows::Win32::System::Wmi::IWbemClassObject;
    pub use windows::Win32::System::Wmi::IWbemLocator;
    pub use windows::Win32::System::Wmi::WbemLocator;
    pub use windows::Win32::System::Wmi::WBEM_FLAG_FORWARD_ONLY;
    pub use windows::Win32::System::Wmi::WBEM_FLAG_RETURN_IMMEDIATELY;
    pub use windows::Win32::System::Wmi::WBEM_FLAG_RETURN_WBEM_COMPLETE;
    pub use windows::Win32::System::Wmi::WBEM_INFINITE;
    pub use windows::Win32::UI::Accessibility::SetWinEventHook;
    pub use windows::Win32::UI::Accessibility::UnhookWinEvent;
    pub use windows::Win32::UI::Accessibility::HWINEVENTHOOK;
    pub use windows::Win32::UI::WindowsAndMessaging::DispatchMessageW;
    pub use windows::Win32::UI::WindowsAndMessaging::GetMessageW;
    pub use windows::Win32::UI::WindowsAndMessaging::GetWindowThreadProcessId;
    pub use windows::Win32::UI::WindowsAndMessaging::TranslateMessage;
    pub use windows::Win32::UI::WindowsAndMessaging::EVENT_SYSTEM_FOREGROUND;
    pub use windows::Win32::UI::WindowsAndMessaging::EVENT_SYSTEM_MINIMIZEEND;
    pub use windows::Win32::UI::WindowsAndMessaging::MSG;
    pub use windows::Win32::UI::WindowsAndMessaging::WINEVENT_OUTOFCONTEXT;
    pub use windows_service::define_windows_service;
    pub use windows_service::service::ServiceAccess;
    pub use windows_service::service::ServiceControl;
    pub use windows_service::service::ServiceControlAccept;
    pub use windows_service::service::ServiceErrorControl;
    pub use windows_service::service::ServiceExitCode;
    pub use windows_service::service::ServiceInfo;
    pub use windows_service::service::ServiceStartType;
    pub use windows_service::service::ServiceState;
    pub use windows_service::service::ServiceStatus;
    pub use windows_service::service::ServiceType;
    pub use windows_service::service_control_handler;
    pub use windows_service::service_control_handler::ServiceControlHandlerResult;
    pub use windows_service::service_dispatcher;
    pub use windows_service::service_manager::ServiceManager;
    pub use windows_service::service_manager::ServiceManagerAccess;
}

#[cfg(target_os = "windows")]
use windows_imports::*;

const SERVICE_NAME: &str = "CampMonitor";
const SERVICE_TYPE: ServiceType = ServiceType::OWN_PROCESS;

#[cfg(target_os = "windows")]
pub fn monitor() {
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);

        let hook = SetWinEventHook(
            EVENT_SYSTEM_FOREGROUND,
            EVENT_SYSTEM_FOREGROUND,
            None,
            Some(handle_win_event),
            0,
            0,
            WINEVENT_OUTOFCONTEXT,
        );

        let minimize_hook = SetWinEventHook(
            EVENT_SYSTEM_MINIMIZEEND,
            EVENT_SYSTEM_MINIMIZEEND,
            None,
            Some(handle_win_event),
            0,
            0,
            WINEVENT_OUTOFCONTEXT,
        );

        let mut msg = MSG::default();
        while GetMessageW(&mut msg, Some(HWND::default()), 0, 0).into() {
            TranslateMessage(&msg).as_bool();
            DispatchMessageW(&msg);
        }

        UnhookWinEvent(hook).as_bool();
        UnhookWinEvent(minimize_hook).as_bool();
        CoUninitialize();
    }
}

#[cfg(not(target_os = "windows"))]
pub fn monitor() {}

unsafe extern "system" fn handle_win_event(
    _hwineventhook: HWINEVENTHOOK, _event: u32, hwnd: HWND, _idobject: i32, _idchild: i32,
    _ideventthread: u32, _dwmseventtime: u32,
) {
    let mut pid: u32 = 0;
    GetWindowThreadProcessId(hwnd, Some(&mut pid));

    let hproc = match OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) {
        Ok(h) => h,
        Err(_) => return,
    };
    let hmodule = HMODULE::default();
    let mut buf = vec![0u16; 260];
    GetModuleFileNameExW(Some(hproc), Some(hmodule), &mut buf);
    CloseHandle(hproc).unwrap();

    let exe_path = OsString::from_wide(&buf).to_string_lossy().to_string();
    let exe_path = exe_path.trim_end_matches('\0');

    let exe_file_name = Path::new(exe_path)
        .file_name()
        .map(|s| s.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    let set = IndexSet::from(["idea64.exe", "zed.exe"]);

    if set.contains(exe_file_name.as_str()) {
        set_brightness_wmi(25).unwrap();
    } else {
        set_brightness_wmi(20).unwrap();
    }
}

unsafe fn set_brightness_wmi(brightness: i32) -> Result<(), windows::core::Error> {
    let locator: IWbemLocator = CoCreateInstance(&WbemLocator, None, CLSCTX_INPROC_SERVER)?;
    let services = locator.ConnectServer(
        &BSTR::from("root\\wmi"),
        &BSTR::new(),
        &BSTR::new(),
        &BSTR::new(),
        0,
        &BSTR::new(),
        None,
    )?;

    CoSetProxyBlanket(
        &services,
        RPC_C_AUTHN_WINNT,
        RPC_C_AUTHN_NONE,
        None,
        RPC_C_AUTHN_LEVEL_PKT,
        RPC_C_IMP_LEVEL_IMPERSONATE,
        None,
        EOAC_NONE,
    )?;

    let enumerator = services.ExecQuery(
        &BSTR::from("WQL"),
        &BSTR::from("SELECT * FROM WmiMonitorBrightnessMethods"),
        WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY,
        None,
    )?;

    let mut monitors: Vec<Option<IWbemClassObject>> = vec![None];
    let mut returned: u32 = 0;
    let _ = enumerator.Next(WBEM_INFINITE, &mut monitors, &mut returned as *mut _);

    for monitor in monitors {
        let monitor = monitor.unwrap();

        let mut path = VARIANT::default();
        monitor.Get(&BSTR::from("__PATH"), 0, &mut path as *mut _, None, None)?;
        let path = path.Anonymous.Anonymous.Anonymous.bstrVal.clone();

        let mut class_obj: Option<IWbemClassObject> = None;
        services.GetObject(
            &BSTR::from("WmiMonitorBrightnessMethods"),
            WBEM_FLAG_RETURN_WBEM_COMPLETE,
            None,
            Some(&mut class_obj as *mut _),
            None,
        )?;

        let class_obj = class_obj.unwrap();
        let mut in_params: Option<IWbemClassObject> = Some(class_obj.SpawnInstance(0)?);
        let mut out_params: Option<IWbemClassObject> = None;
        class_obj.GetMethod(
            &BSTR::from("WmiSetBrightness"),
            0,
            &mut in_params as *mut _,
            &mut out_params as *mut _,
        )?;

        let in_params = in_params.unwrap();
        in_params.Put(&BSTR::from("Brightness"), 0, &VARIANT::from(brightness), 0)?;
        in_params.Put(&BSTR::from("Timeout"), 0, &VARIANT::from(0i32), 0)?;

        services.ExecMethod(
            &path,
            &BSTR::from("WmiSetBrightness"),
            Default::default(),
            None,
            &in_params,
            None,
            None,
        )?;
    }

    Ok(())
}

unsafe fn _set_brightness_ddc(hwnd: HWND, brightness: u32) {
    let hmonitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTOPRIMARY);

    let mut count: u32 = 0;
    GetNumberOfPhysicalMonitorsFromHMONITOR(hmonitor, &mut count).unwrap();

    let mut phys: Vec<PHYSICAL_MONITOR> = vec![PHYSICAL_MONITOR::default(); count as usize];
    GetPhysicalMonitorsFromHMONITOR(hmonitor, &mut phys).unwrap();

    for p in &phys {
        SetMonitorBrightness(p.hPhysicalMonitor, brightness);
    }

    DestroyPhysicalMonitors(&phys).unwrap();
}

#[cfg(target_os = "windows")]
pub fn start_monitor_service() -> windows_service::Result<()> {
    service_dispatcher::start(SERVICE_NAME, ffi_monitor_service)
}

#[cfg(not(target_os = "windows"))]
pub fn start_monitor_service() -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

#[cfg(target_os = "windows")]
define_windows_service!(ffi_monitor_service, monitor_service);

#[cfg(target_os = "windows")]
fn monitor_service(_arguments: Vec<OsString>) {
    if let Err(e) = monitor_service_handler() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let tmp_path = std::env::temp_dir().join(format!("camp_monitor_error_{}.log", ts));
        if let Ok(mut f) = File::create(&tmp_path) {
            let _ = writeln!(f, "Error: {}", e);
        }
    }
}

fn monitor_service_handler() -> Result<(), Box<dyn std::error::Error>> {
    let (shutdown_tx, shutdown_rx) = mpsc::channel();

    let event_handler = move |control_event| -> ServiceControlHandlerResult {
        match control_event {
            ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
            ServiceControl::Stop => {
                shutdown_tx.send(()).unwrap();
                ServiceControlHandlerResult::NoError
            }
            _ => ServiceControlHandlerResult::NotImplemented,
        }
    };

    let status_handle = service_control_handler::register(SERVICE_NAME, event_handler).unwrap();
    status_handle.set_service_status(ServiceStatus {
        service_type: SERVICE_TYPE,
        current_state: ServiceState::Running,
        controls_accepted: ServiceControlAccept::STOP,
        exit_code: ServiceExitCode::Win32(0),
        checkpoint: 0,
        wait_hint: Duration::default(),
        process_id: None,
    })?;

    let exe_path = std::env::current_exe()?;
    let command_line = format!("\"{}\" monitor", exe_path.display());

    let process_handle = unsafe { spawn_user_process(command_line)? };

    loop {
        match shutdown_rx.recv_timeout(Duration::from_secs(1)) {
            Ok(_) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                unsafe {
                    TerminateProcess(process_handle, 0)?;
                    CloseHandle(process_handle)?;
                }
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => (),
        };
    }

    status_handle.set_service_status(ServiceStatus {
        service_type: SERVICE_TYPE,
        current_state: ServiceState::Stopped,
        controls_accepted: ServiceControlAccept::empty(),
        exit_code: ServiceExitCode::Win32(0),
        checkpoint: 0,
        wait_hint: Duration::default(),
        process_id: None,
    })?;

    Ok(())
}

unsafe fn spawn_user_process(command_line: String) -> Result<HANDLE, windows::core::Error> {
    let mut sessions: *mut WTS_SESSION_INFOW = ptr::null_mut();
    let mut session_count: u32 = 0;
    let mut user_token: HANDLE = HANDLE::default();

    'retry: loop {
        WTSEnumerateSessionsW(None, 0, 1, &mut sessions, &mut session_count)?;

        let sessions_slice = slice::from_raw_parts(sessions, session_count as usize);
        for session in sessions_slice {
            if session.State != WTSActive {
                continue;
            }

            match WTSQueryUserToken(session.SessionId, &mut user_token) {
                Ok(_) => break 'retry,
                Err(_) => {}
            }
        }

        sleep(Duration::from_secs(3));
    }

    WTSFreeMemory(sessions as *mut c_void);

    let mut command_line_wide: Vec<u16> = OsString::from(&command_line)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();

    let mut environment: *mut c_void = ptr::null_mut();
    CreateEnvironmentBlock(&mut environment, Some(user_token), false)?;

    let mut desktop: Vec<u16> = OsString::from("winsta0\\default")
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();

    let startup_info = STARTUPINFOW {
        cb: std::mem::size_of::<STARTUPINFOW>() as u32,
        lpDesktop: PWSTR::from_raw(desktop.as_mut_ptr()),
        ..Default::default()
    };

    let mut process_info = PROCESS_INFORMATION::default();

    CreateProcessAsUserW(
        Some(user_token),
        None,
        Some(PWSTR::from_raw(command_line_wide.as_mut_ptr())),
        None,
        None,
        false,
        CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT,
        Some(environment),
        None,
        &startup_info,
        &mut process_info,
    )?;

    DestroyEnvironmentBlock(environment)?;
    CloseHandle(process_info.hThread)?;
    CloseHandle(user_token)?;
    Ok(process_info.hProcess)
}

#[cfg(target_os = "windows")]
pub fn install_monitor_service() -> Result<(), Box<dyn std::error::Error>> {
    let manager_access = ServiceManagerAccess::CONNECT | ServiceManagerAccess::CREATE_SERVICE;
    let service_manager = ServiceManager::local_computer(None::<&str>, manager_access)?;

    let service_binary_path = std::env::current_exe()?;

    let service_info = ServiceInfo {
        name: OsString::from(SERVICE_NAME),
        display_name: OsString::from("Camp Monitor"),
        service_type: SERVICE_TYPE,
        start_type: ServiceStartType::AutoStart,
        error_control: ServiceErrorControl::Normal,
        executable_path: service_binary_path,
        launch_arguments: vec![OsString::from("monitor"), OsString::from("service")],
        dependencies: vec![],
        account_name: None,
        account_password: None,
    };

    service_manager.create_service(&service_info, ServiceAccess::CHANGE_CONFIG)?;
    Ok(())
}

#[cfg(not(target_os = "windows"))]
pub fn install_monitor_service() -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

#[cfg(target_os = "windows")]
pub fn uninstall_monitor_service() -> Result<(), windows_service::Error> {
    use std::time::Instant;

    let manager_access = ServiceManagerAccess::CONNECT;
    let service_manager = ServiceManager::local_computer(None::<&str>, manager_access)?;

    let service_access = ServiceAccess::QUERY_STATUS | ServiceAccess::STOP | ServiceAccess::DELETE;
    let service = service_manager.open_service(SERVICE_NAME, service_access)?;
    service.delete()?;

    if service.query_status()?.current_state != ServiceState::Stopped {
        service.stop()?;
    }

    drop(service);

    let start = Instant::now();
    let timeout = Duration::from_secs(5);
    while start.elapsed() < timeout {
        let service = service_manager.open_service(SERVICE_NAME, ServiceAccess::QUERY_STATUS);
        if let Err(windows_service::Error::Winapi(e)) = service {
            if e.raw_os_error() == Some(ERROR_SERVICE_DOES_NOT_EXIST.0 as i32) {
                println!("{} is deleted.", SERVICE_NAME);
                return Ok(());
            }
        }

        sleep(Duration::from_secs(1));
    }

    println!("{} is marked for deletion.", SERVICE_NAME);
    Ok(())
}

#[cfg(not(target_os = "windows"))]
pub fn uninstall_monitor_service() -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}
