//! Core GPU stat structures and NVML query logic
//! Ported from Python gpustat (https://github.com/wookayin/gpustat)

use chrono::{DateTime, Utc};
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::Nvml;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

const MB: u64 = 1024 * 1024;

/// Process information running on GPU
#[derive(Debug, Clone, Serialize)]
pub struct GpuProcessInfo {
    pub pid: u32,
    pub username: Option<String>,
    pub command: String,
    pub gpu_memory_usage: Option<u64>, // in MB
    /// True when username/command were resolved via Ngid-to-PID mapping
    pub username_from_ngid_mapping: bool,
    /// Real host PID when resolved via Ngid mapping (original pid was Ngid)
    pub real_pid: Option<u32>,
}

/// Single GPU statistics
#[derive(Debug, Clone, Serialize)]
pub struct GpuStat {
    pub index: u32,
    pub name: String,
    pub uuid: String,
    pub temperature: Option<u32>,
    pub fan_speed: Option<u32>,
    pub utilization: Option<u32>,
    pub utilization_enc: Option<u32>,
    pub utilization_dec: Option<u32>,
    pub power_draw: Option<u32>,  // Watts
    pub power_limit: Option<u32>, // Watts
    pub memory_used: u64,  // MB
    pub memory_total: u64, // MB
    pub processes: Option<Vec<GpuProcessInfo>>,
    pub available: bool,
}

/// Collection of GPU stats with host info
#[derive(Debug, Clone, Serialize)]
pub struct GpuStatCollection {
    pub hostname: String,
    pub query_time: DateTime<Utc>,
    pub driver_version: Option<String>,
    pub gpus: Vec<GpuStat>,
}

impl GpuStatCollection {
    /// Query all GPUs and return a new GpuStatCollection
    pub fn new_query(nvml: &Nvml, gpu_ids: Option<&[u32]>) -> Result<Self, nvml_wrapper::error::NvmlError> {
        let device_count = nvml.device_count()?;
        let hostname = hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        let driver_version = nvml.sys_driver_version().ok();

        let gpus_to_query: Vec<u32> = match gpu_ids {
            Some(ids) => ids.to_vec(),
            None => (0..device_count).collect(),
        };

        let mut gpus = Vec::new();
        for &index in &gpus_to_query {
            match get_gpu_info(nvml, index) {
                Ok(stat) => gpus.push(stat),
                Err(e) => {
                    gpus.push(GpuStat {
                        index,
                        name: format!("((Error: {}))", e),
                        uuid: String::new(),
                        temperature: None,
                        fan_speed: None,
                        utilization: None,
                        utilization_enc: None,
                        utilization_dec: None,
                        power_draw: None,
                        power_limit: None,
                        memory_used: 0,
                        memory_total: 0,
                        processes: None,
                        available: false,
                    });
                }
            }
        }

        Ok(Self {
            hostname,
            query_time: Utc::now(),
            driver_version,
            gpus,
        })
    }
}

/// Build mapping from Ngid/NStgid to host PID by scanning /proc.
/// On some systems, NVML returns Ngid instead of the real PID; this mapping
/// allows us to resolve to the actual process.
#[cfg(target_os = "linux")]
fn build_ngid_to_pid_mapping() -> HashMap<u32, u32> {
    let mut mapping = HashMap::new();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return mapping;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Ok(host_pid) = name.to_string_lossy().parse::<u32>() else {
            continue;
        };
        let status_path = entry.path().join("status");
        let Ok(status) = std::fs::read_to_string(&status_path) else {
            continue;
        };

        let mut ngid = None;
        let mut nstgid_values = Vec::new();

        for line in status.lines() {
            if let Some(value) = line.strip_prefix("Ngid:") {
                if let Ok(v) = value.trim().parse::<u32>() {
                    ngid = Some(v);
                }
                break;
            }
        }
        for line in status.lines() {
            if let Some(value) = line.strip_prefix("NStgid:") {
                for part in value.split_whitespace() {
                    if let Ok(v) = part.parse::<u32>() {
                        nstgid_values.push(v);
                    }
                }
                break;
            }
        }

        if let Some(ng) = ngid {
            if ng != 0 {
                mapping.insert(ng, host_pid);
            }
        }
        for ns_pid in nstgid_values {
            mapping.insert(ns_pid, host_pid);
        }
    }
    mapping
}

/// Get process info from PID (username, command).
/// When direct lookup fails (e.g. NVML returns Ngid instead of PID), retries
/// using Ngid/NStgid-to-PID mapping.
/// Returns (username, command, resolved_via_ngid_mapping, real_pid_when_mapped).
fn get_process_info(pid: u32) -> (Option<String>, String, bool, Option<u32>) {
    #[cfg(target_os = "linux")]
    {
        use std::path::Path;
        use procfs::process::Process;

        let try_lookup = |p: u32| -> (Option<String>, String) {
            let process = match Process::new(p as i32) {
                Ok(proc) => proc,
                Err(_) => return (None, "?".to_string()),
            };

            let username = process
                .uid()
                .ok()
                .and_then(|uid| users::get_user_by_uid(uid))
                .and_then(|u| u.name().to_str().map(|s| s.to_string()));

            let command = process
                .cmdline()
                .ok()
                .and_then(|cmdline| {
                    if cmdline.is_empty() {
                        process.stat().ok().map(|s| s.comm.clone())
                    } else {
                        Some(
                            Path::new(&cmdline[0])
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("?")
                                .to_string(),
                        )
                    }
                })
                .unwrap_or_else(|| "?".to_string());

            (username, command)
        };

        static MAPPING: OnceLock<HashMap<u32, u32>> = OnceLock::new();
        let mapping = MAPPING.get_or_init(build_ngid_to_pid_mapping);

        // When mapping says pid -> real_pid and real_pid != pid, the NVML "pid" is Ngid/ns-pid.
        // We must use the mapped result even if direct lookup "succeeded" (that would be a wrong
        // process when Ngid happens to equal some unrelated host PID).
        let mut result = try_lookup(pid);
        let mut from_mapping = false;
        let mut real_pid = None;
        if let Some(&rp) = mapping.get(&pid) {
            if rp != pid {
                result = try_lookup(rp);
                from_mapping = result.1 != "?";
                if from_mapping {
                    real_pid = Some(rp);
                }
            }
        } else if result.1 == "?" {
            // Not in mapping and direct failed - can't resolve
        }
        (result.0, result.1, from_mapping, real_pid)
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        (None, "?".to_string(), false, None)
    }
}

fn get_gpu_info(nvml: &Nvml, index: u32) -> Result<GpuStat, nvml_wrapper::error::NvmlError> {
    let device = nvml.device_by_index(index)?;

    // Basic info
    let name = device.name()?;
    let uuid = device.uuid()?;

    // Temperature (suppress not supported)
    let temperature = device
        .temperature(TemperatureSensor::Gpu)
        .ok();

    // Fan speed
    let fan_speed = device.fan_speed(0).ok();

    // Memory
    let memory = device.memory_info()?;
    let memory_used = memory.used / MB;
    let memory_total = memory.total / MB;

    // Utilization
    let utilization = device.utilization_rates().ok().map(|u| u.gpu);

    let utilization_enc = device.encoder_utilization().ok().map(|u| u.utilization);
    let utilization_dec = device.decoder_utilization().ok().map(|u| u.utilization);

    // Power (NVML returns milliwatts)
    let power_draw = device.power_usage().ok().map(|p| p / 1000);
    let power_limit = device.enforced_power_limit().ok().map(|p| p / 1000);

    // Processes - merge compute and graphics
    // None = NVML doesn't support process query (both APIs failed)
    // Some(vec) = API succeeded, vec can be empty (no processes on GPU)
    let comp_result = device.running_compute_processes();
    let graphics_result = device.running_graphics_processes();

    let processes: Option<Vec<GpuProcessInfo>> = if comp_result.is_err() && graphics_result.is_err() {
        None // Not Supported
    } else {
        let mut processes = Vec::new();
        let mut seen_pids = HashSet::new();

        for nv_process in comp_result
            .unwrap_or_default()
            .into_iter()
            .chain(graphics_result.unwrap_or_default())
        {
            if !seen_pids.insert(nv_process.pid) {
                continue;
            }

            let gpu_memory_mb = match &nv_process.used_gpu_memory {
                UsedGpuMemory::Used(bytes) => Some(*bytes / MB),
                UsedGpuMemory::Unavailable => None,
            };

            let (username, command, username_from_ngid_mapping, real_pid) = get_process_info(nv_process.pid);

            processes.push(GpuProcessInfo {
                pid: nv_process.pid,
                username,
                command,
                gpu_memory_usage: gpu_memory_mb,
                username_from_ngid_mapping,
                real_pid,
            });
        }
        Some(processes)
    };

    Ok(GpuStat {
        index,
        name,
        uuid,
        temperature,
        fan_speed,
        utilization,
        utilization_enc,
        utilization_dec,
        power_draw,
        power_limit,
        memory_used,
        memory_total,
        processes,
        available: true,
    })
}
