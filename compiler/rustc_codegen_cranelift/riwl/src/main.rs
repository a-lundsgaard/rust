//! CLI shim over riwl::cli::run_args.

fn main() {
    if let Err(e) = riwl::cli::run_args(std::env::args().skip(1).collect()) {
        eprintln!("riwl: error: {e:#}");
        std::process::exit(1);
    }
}
