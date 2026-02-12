//! gpustat-rs: A Rust reimplementation of gpustat
//! Monitor your NVIDIA GPU status, like htop but for GPUs.

mod core;
mod display;

use clap::Parser;
use display::DisplayOptions;
use std::process;
use std::time::Duration;
use std::thread;

#[derive(Parser)]
#[command(name = "gpustat")]
#[command(about = "A monitoring tool for NVIDIA GPUs", long_about = None)]
struct Args {
    /// Comma-separated GPU indices to query (e.g. "0,1,2"). Default: all GPUs.
    #[arg(long, value_name = "IDS")]
    id: Option<String>,

    /// Print as JSON
    #[arg(long)]
    json: bool,

    /// Show all: cmd, user, pid, fan, codec, power
    #[arg(short = 'a', long)]
    show_all: bool,

    /// Display command name of running process
    #[arg(short = 'c', long)]
    show_cmd: bool,

    /// Display full command and CPU stats of running process
    #[arg(short = 'f', long)]
    show_full_cmd: bool,

    /// Display username of running process
    #[arg(short = 'u', long)]
    show_user: bool,

    /// Display PID of running process
    #[arg(short = 'p', long)]
    show_pid: bool,

    /// Display GPU fan speed
    #[arg(short = 'F', long)]
    show_fan_speed: bool,

    /// Show encoder/decoder utilization [possible values: enc, dec, enc,dec]
    #[arg(short = 'e', long, value_name = "CODEC")]
    show_codec: Option<Option<String>>,

    /// Show power usage [possible values: draw, limit, draw,limit]
    #[arg(short = 'P', long, value_name = "POWER")]
    show_power: Option<Option<String>>,

    /// Do not display header
    #[arg(long)]
    no_header: bool,

    /// Width for GPU name column (0 to hide)
    #[arg(long, value_name = "N")]
    gpuname_width: Option<usize>,

    /// Do not display running process information
    #[arg(long)]
    no_processes: bool,

    /// Force colored output
    #[arg(long, alias = "color")]
    force_color: bool,

    /// Suppress colored output
    #[arg(long)]
    no_color: bool,

    /// Use watch mode; seconds between updates (default: 1.0)
    #[arg(short = 'i', long = "interval", value_name = "SECONDS")]
    watch: Option<Option<f64>>,

    /// Print version
    #[arg(short = 'v', long)]
    version: bool,
}

fn main() {
    let args = Args::parse();

    if args.version {
        println!("gpustat-rs 0.1.0");
        return;
    }

    if args.force_color && args.no_color {
        eprintln!("Error: --force-color and --no-color cannot be used together");
        process::exit(1);
    }

    if args.json && args.watch.is_some() {
        eprintln!("Error: --json and --interval cannot be used together");
        process::exit(1);
    }

    // Parse GPU IDs
    let gpu_ids: Option<Vec<u32>> = args.id.as_ref().map(|s| {
        s.split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect()
    });

    // Build display options
    let mut opts = DisplayOptions {
        show_cmd: args.show_cmd || args.show_all,
        show_user: args.show_user || args.show_all,
        show_pid: args.show_pid || args.show_all,
        show_fan_speed: args.show_fan_speed || args.show_all,
        show_codec: args.show_codec.is_some() || args.show_all,
        show_power: args.show_power.is_some() || args.show_all,
        show_power_limit: args.show_power.as_ref()
            .map(|o| o.as_ref().map(|s| s.contains("limit")).unwrap_or(true))
            .unwrap_or(args.show_all),
        no_processes: args.no_processes,
        no_header: args.no_header,
        gpuname_width: args.gpuname_width,
        force_color: args.force_color,
        no_color: args.no_color,
    };

    // Handle show_power: "draw", "limit", "draw,limit"
    if let Some(ref power_opt) = args.show_power {
        opts.show_power = true;
        opts.show_power_limit = power_opt
            .as_ref()
            .map(|s| s.contains("limit"))
            .unwrap_or(true);
    }

    let interval = match args.watch {
        Some(Some(secs)) => Duration::from_secs_f64(secs.max(0.1)),
        Some(None) => Duration::from_secs_f64(1.0),
        None => Duration::ZERO,
    };

    let run_once = interval == Duration::ZERO;

    loop {
        match run_gpustat(&gpu_ids, &opts, args.json) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("Error querying NVIDIA devices: {}", e);
                process::exit(1);
            }
        }

        if run_once {
            break;
        }

        thread::sleep(interval);

        // Clear screen for watch mode (cursor to 0,0 and clear)
        if !args.json {
            print!("\x1b[H\x1b[J");
        }
    }
}

fn run_gpustat(
    gpu_ids: &Option<Vec<u32>>,
    opts: &DisplayOptions,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let nvml = nvml_wrapper::Nvml::init()?;
    let stats = core::GpuStatCollection::new_query(
        &nvml,
        gpu_ids.as_deref(),
    )?;

    if json {
        println!("{}", serde_json::to_string_pretty(&stats)?);
    } else {
        stats.print_formatted(opts)?;
    }

    Ok(())
}
