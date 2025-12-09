use indexmap::IndexSet;
use std::ffi::OsString;
use std::path::Path;

#[cfg(target_os = "windows")]
mod windows_imports {
    pub use std::os::windows::ffi::OsStringExt;
    pub use windows::core::BSTR;
    pub use windows::Win32::Devices::Display::DestroyPhysicalMonitors;
    pub use windows::Win32::Devices::Display::GetNumberOfPhysicalMonitorsFromHMONITOR;
    pub use windows::Win32::Devices::Display::GetPhysicalMonitorsFromHMONITOR;
    pub use windows::Win32::Devices::Display::SetMonitorBrightness;
    pub use windows::Win32::Devices::Display::PHYSICAL_MONITOR;
    pub use windows::Win32::Foundation::CloseHandle;
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
    pub use windows::Win32::System::ProcessStatus::GetModuleFileNameExW;
    pub use windows::Win32::System::Rpc::RPC_C_AUTHN_NONE;
    pub use windows::Win32::System::Rpc::RPC_C_AUTHN_WINNT;
    pub use windows::Win32::System::Threading::OpenProcess;
    pub use windows::Win32::System::Threading::PROCESS_QUERY_LIMITED_INFORMATION;
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
}

#[cfg(target_os = "windows")]
use windows_imports::*;

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

#[cfg(not(target_os = "windows"))]
pub fn monitor() {}
