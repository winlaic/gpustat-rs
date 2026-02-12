//! Terminal display for GPU stats (colored output like Python gpustat)

use crate::core::{GpuProcessInfo, GpuStat, GpuStatCollection};
use colored::Colorize;
use std::io::IsTerminal;
use std::fmt;
use std::io::{self, Write};

const DEFAULT_GPUNAME_WIDTH: usize = 16;
const NOT_SUPPORTED: &str = "Not Supported";

/// Shorten string from left with ellipsis
fn shorten_left(text: &str, width: usize, placeholder: &str) -> String {
    if width == 0 {
        return String::new();
    }
    if text.len() <= width {
        return text.to_string();
    }
    if width <= placeholder.len() {
        return placeholder[..width].to_string();
    }
    format!("{}{}", placeholder, &text[text.len() - (width - placeholder.len())..])
}

/// Display options for GPU stats
#[derive(Debug, Clone, Default)]
pub struct DisplayOptions {
    pub show_cmd: bool,
    pub show_user: bool,
    pub show_pid: bool,
    pub show_fan_speed: bool,
    pub show_codec: bool,      // enc,dec or both
    pub show_power: bool,     // draw, limit or both
    pub show_power_limit: bool,
    pub no_processes: bool,
    pub no_header: bool,
    pub gpuname_width: Option<usize>,
    pub force_color: bool,
    pub no_color: bool,
}

fn opt_repr<T: fmt::Display>(v: Option<T>, none: &str) -> String {
    match v {
        Some(x) => x.to_string(),
        None => none.to_string(),
    }
}

/// Right-justify to width (like Python rjustify)
fn rjust<T: fmt::Display>(v: T, w: usize) -> String {
    format!("{0:>1$}", v, w)
}

impl GpuStat {
    fn format_line(
        &self,
        opts: &DisplayOptions,
        use_color: bool,
    ) -> String {
        let mut s = String::new();

        // [index]
        if use_color {
            s.push_str(&format!("[{}] ", self.index).cyan().to_string());
        } else {
            s.push_str(&format!("[{}] ", self.index));
        }

        // GPU name
        let gpu_width = opts.gpuname_width.unwrap_or(DEFAULT_GPUNAME_WIDTH);
        if gpu_width > 0 {
            let name = shorten_left(&self.name, gpu_width, "…");
            if use_color {
                let name_colored = if self.available {
                    name.blue().to_string()
                } else {
                    name.red().to_string()
                };
                s.push_str(&format!("{:>width$} | ", name_colored, width = gpu_width));
            } else {
                s.push_str(&format!("{:>width$} | ", name, width = gpu_width));
            }
        }

        // Temperature - rjust 3 then color (Python: CTemp < 50 → red, else bold_red)
        let temp_str = rjust(opt_repr(self.temperature.as_ref(), "??"), 3);
        if use_color {
            let temp_colored = match self.temperature {
                Some(t) if t < 50 => temp_str.red().to_string(),
                Some(_) => temp_str.bold().red().to_string(),
                _ => temp_str.to_string(),
            };
            s.push_str(&format!("{}°C, ", temp_colored));
        } else {
            s.push_str(&format!("{}°C, ", temp_str));
        }

        // Fan speed (optional) - rjust 3 (Python: FSpeed < 30 → cyan, else bold_cyan)
        if opts.show_fan_speed {
            let fan_str = rjust(opt_repr(self.fan_speed.as_ref(), "??"), 3);
            if use_color {
                let fan_colored = match self.fan_speed {
                    Some(f) if f < 30 => fan_str.cyan().to_string(),
                    _ => fan_str.bold().cyan().to_string(),
                };
                s.push_str(&format!("{} %, ", fan_colored));
            } else {
                s.push_str(&format!("{} %, ", fan_str));
            }
        }

        // Utilization - rjust 3 then color (Python: CUtil < 30 → green, else bold_green)
        // Build full "  XX %" string first so padding is correct, then color entire field
        let util_display = format!("{} %", rjust(opt_repr(self.utilization.as_ref(), "??"), 3));
        if use_color {
            let util_colored = match self.utilization {
                Some(u) if u < 30 => util_display.as_str().green().to_string(),
                _ => util_display.as_str().bold().green().to_string(),
            };
            s.push_str(&util_colored);
        } else {
            s.push_str(&util_display);
        }

        // Codec (optional) - rjust 3 for enc/dec (Python: < 50 → green, else bold_green)
        if opts.show_codec {
            let enc_str = rjust(opt_repr(self.utilization_enc.as_ref(), "??"), 3);
            let dec_str = rjust(opt_repr(self.utilization_dec.as_ref(), "??"), 3);
            s.push_str(" (");
            if use_color {
                let enc_c = match self.utilization_enc {
                    Some(u) if u < 50 => enc_str.green().to_string(),
                    _ => enc_str.bold().green().to_string(),
                };
                let dec_c = match self.utilization_dec {
                    Some(u) if u < 50 => dec_str.green().to_string(),
                    _ => dec_str.bold().green().to_string(),
                };
                s.push_str(&format!("E: {} %, D: {} %", enc_c, dec_c));
            } else {
                s.push_str(&format!("E: {} %, D: {} %", enc_str, dec_str));
            }
            s.push_str(")");
        }

        // Power (optional) - rjust 3 (Python: draw/limit < 0.4 → magenta, else bold_magenta)
        if opts.show_power {
            let pow_str = rjust(opt_repr(self.power_draw.as_ref(), "??"), 3);
            if use_color {
                let pow_colored = match (self.power_draw, self.power_limit) {
                    (Some(d), Some(l)) if l > 0 && (d as f32 / l as f32) < 0.4 => pow_str.magenta().to_string(),
                    _ => pow_str.bold().magenta().to_string(),
                };
                s.push_str(&format!(",  {} ", pow_colored));
            } else {
                s.push_str(&format!(",  {} ", pow_str));
            }
            if opts.show_power_limit {
                let limit_str = rjust(opt_repr(self.power_limit.as_ref(), "??"), 3);
                if use_color {
                    s.push_str(&format!("/ {} W", limit_str.magenta()));
                } else {
                    s.push_str(&format!("/ {} W", limit_str));
                }
            }
        }

        // Memory - rjust 5 for used/total (Python: CMemU bold_yellow, CMemT yellow)
        s.push_str(" | ");
        if use_color {
            let mem_used_str = rjust(self.memory_used.to_string(), 5);
            let mem_total_str = rjust(self.memory_total.to_string(), 5);
            s.push_str(&format!(
                "{} / {} MB",
                mem_used_str.bold().yellow(),
                mem_total_str.yellow()
            ));
        } else {
            s.push_str(&format!("{:>5} / {:>5} MB", self.memory_used, self.memory_total));
        }

        // Processes - only "(Not Supported)" when processes is None (NVML API failed)
        // When Some([]) (no processes on GPU), show nothing after " |"
        if !opts.no_processes {
            s.push_str(" |");
            match &self.processes {
                None => s.push_str(&format!(" ({})", NOT_SUPPORTED)),
                Some(procs) => {
                    for p in procs {
                        s.push_str(&format_process(p, opts, use_color));
                    }
                }
            }
        }

        s
    }
}

fn format_process(p: &GpuProcessInfo, opts: &DisplayOptions, use_color: bool) -> String {
    let mut s = String::new();
    s.push(' ');

    // Python: CUser = term.bold_black (gray for username)
    // Username resolved via Ngid mapping: green (.green() for terminal compatibility)
    let show_username = opts.show_user || !opts.show_cmd;
    if show_username {
        let username = p.username.as_deref().unwrap_or("--");
        if use_color {
            let username_str = if p.username_from_ngid_mapping {
                username.green().to_string()
            } else {
                username.bright_black().to_string()
            };
            s.push_str(&username_str);
        } else {
            s.push_str(username);
        }
    }
    if opts.show_cmd {
        if show_username {
            s.push(':');
        }
        let cmd = &p.command;
        if use_color {
            s.push_str(&cmd.cyan().to_string());
        } else {
            s.push_str(cmd);
        }
    }
    if opts.show_pid {
        let pid_str = match p.real_pid {
            Some(rp) => format!("{}->{}", p.pid, rp),
            None => p.pid.to_string(),
        };
        s.push_str(&format!("/{}", pid_str));
    }
    let mem_str = match p.gpu_memory_usage {
        Some(m) => m.to_string(),
        None => "?".to_string(),
    };
    if use_color {
        s.push_str(&format!("({}M)", mem_str.yellow()));
    } else {
        s.push_str(&format!("({}M)", mem_str));
    }
    s
}

impl GpuStatCollection {
    /// Print formatted GPU stats to stdout
    pub fn print_formatted(
        &self,
        opts: &DisplayOptions,
    ) -> io::Result<()> {
        let use_color = if opts.no_color {
            false
        } else if opts.force_color {
            true
        } else {
            std::io::stdout().is_terminal()
        };

        let gpu_width = opts.gpuname_width.unwrap_or_else(|| {
            self.gpus
                .iter()
                .map(|g| g.name.len())
                .max()
                .unwrap_or(0)
                .max(DEFAULT_GPUNAME_WIDTH)
        });

        let mut opts = opts.clone();
        opts.gpuname_width = Some(gpu_width);

        // Header
        if !opts.no_header {
            let timestr = self.query_time.format("%Y-%m-%d %H:%M:%S");
            let driver = self.driver_version.as_deref().unwrap_or("N/A");
            if use_color {
                println!(
                    "{}  {}  {}",
                    self.hostname.bold().white(),
                    timestr,
                    driver.dimmed()
                );
            } else {
                println!("{}  {}  {}", self.hostname, timestr, driver);
            }
        }

        // Body
        for gpu in &self.gpus {
            println!("{}", gpu.format_line(&opts, use_color));
        }

        if self.gpus.is_empty() {
            if use_color {
                println!("{}", "(No GPUs are available)".yellow());
            } else {
                println!("(No GPUs are available)");
            }
        }

        io::stdout().flush()
    }
}
