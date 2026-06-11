
pub mod candlestick;
pub mod indicators;
pub mod backtest;
pub mod brkga;
pub mod indicator_cache;
mod utils;

use candlestick::Candlestick;
use backtest::{Backtest, RunMode};
use brkga::{BRKGA, BrkgaConfig, FitnessExecutor};

fn main() {
    let csv_path = "scripts/data_collector/ETHUSDT-5m.csv";
    
    let candles = match candlestick::load_candlesticks(csv_path) {
        Ok(candles) => candles,
        Err(_) => {
            println!("Could't load the file {}. Check if the file exists and has the required csv structure.", csv_path);
            return;
        }
    };
    println!("found {} candles inside {}", candles.len(), csv_path);

    let seed: u64 = 18988547;
    let frac_bot: f32 = 0.10;  // γ = 10% mutantes (paper seção 4.4)
    let frac_top: f32 = 0.25;  // β = 25% elite (paper seção 4.4)
    let pop_size : usize = 30000; // N = 30000 (paper seção 4.4)
    let max_iter: usize = 1000;   // itermax = 1000 (paper seção 4.4)
    let elit_rate : f32 = 0.70;   // α = 70% (paper seção 4.4)

    run_experiment(candles, seed, (frac_top, frac_bot, pop_size, max_iter, elit_rate));
}

fn run_experiment(candles: Vec<Candlestick>, seed: u64, config: BrkgaConfig) {
    println!("Running backtest with {} divisions", 12);
    
    let backtest_engine = Backtest::new(candles, 12, 0.005, 0.0001); // taxa 0.01% (paper seção 4.5)
    let mut brkga = BRKGA::new(seed, 35, config, // 35 genes: 1 alavancagem + 17 long + 17 short (paper seção 3.6)
        FitnessExecutor::new(backtest_engine, RunMode::Training), 500);
    brkga.run();
}