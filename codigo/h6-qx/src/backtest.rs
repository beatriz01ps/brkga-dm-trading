mod trade;
mod trade_rule;
pub mod strategy;

use trade::Trade;
use crate::candlestick::Candlestick;
use strategy::TradingStrategy;
use crate::utils::split_number_in_points;
use self::strategy::SingleStrategy;

pub struct Backtest {
    candlesticks: Vec<Candlestick>,
    all_ranges: Vec<(u32, u32)>,
    training_indices: Vec<usize>,
    validation_indices: Vec<usize>,
    slipage_percentage: f32,
    initialization_candles: u32,
    fee_percentage: f32,
    initial_usd_balance: f32,
}

#[derive(PartialEq, Eq, Copy, Clone)]
pub enum RunMode {
    Training,
    Validation,
}

impl Backtest {
    pub fn new(candlesticks: Vec<Candlestick>, divisions: u8, slipage_percentage: f32, fee_percentage: f32) -> Backtest {
        let all_ranges = split_number_in_points(candlesticks.len() as u32, divisions as u32);

        // paper seção 4.7: 2 blocos de validação — um no meio, um no fim
        let val_middle = all_ranges.len() / 2;
        let val_end = all_ranges.len() - 1;
        let mut training_indices = Vec::with_capacity(all_ranges.len().saturating_sub(2));
        let mut validation_indices = Vec::with_capacity(2);
        for i in 0..all_ranges.len() {
            if i == val_middle || i == val_end {
                validation_indices.push(i);
            } else {
                training_indices.push(i);
            }
        }

        Backtest {
            candlesticks,
            all_ranges,
            training_indices,
            validation_indices,
            fee_percentage,
            slipage_percentage,
            initial_usd_balance: 10_000.0,
            initialization_candles: 100,
        }
    }

    pub fn all_ranges(&self) -> &[(u32, u32)] { &self.all_ranges }

    pub fn candles_clone(&self) -> Vec<Candlestick> { self.candlesticks.clone() }

    pub fn run(&self, mode: RunMode, model: &mut SingleStrategy) -> f32 {
        let indices = if mode == RunMode::Training {
            &self.training_indices
        } else {
            &self.validation_indices
        };

        let mut trade_count = 0;
        let mut total_profit: f32 = 0.0;

        for &global_idx in indices {
            let range = self.all_ranges[global_idx];
            model.set_range(global_idx);

            let mut balance = self.initial_usd_balance;
            let mut current_trade: Option<Trade> = None;

            // warmup: apenas MACD precisa de streaming; outros indicadores são pré-calculados
            for i in range.0..range.0 + self.initialization_candles {
                model.warmup(&self.candlesticks[i as usize]);
            }

            for i in range.0 + self.initialization_candles..range.1 {
                let current_candle = &self.candlesticks[i as usize];
                model.new_candlestick(i as usize, current_candle);

                if current_trade.is_none() {
                    if let Some(ts) = model.should_start_trade() {
                        let balance_debit = model.percentage_amount_per_trade() * balance;
                        let mut new_trade = Trade::open(ts.0, balance_debit,
                            current_candle, model.leverage(), self.slipage_percentage, self.fee_percentage);
                        new_trade.takeprofit(ts.1);
                        new_trade.stoploss(ts.2);
                        current_trade = Some(new_trade);
                        balance -= balance_debit;
                        trade_count += 1;
                    }
                } else if let Some(trade) = current_trade.as_mut() {
                    if trade.is_liquidation_reached(current_candle) {
                        let result = trade.close_on_liquidation(current_candle);
                        total_profit += result;
                        balance += trade.initial_position_size + result;
                        current_trade = None;
                    } else if trade.is_stoploss_reached(current_candle) {
                        let result = trade.close_on_stoploss(current_candle);
                        total_profit += result;
                        balance += trade.initial_position_size + result;
                        current_trade = None;
                    } else if trade.is_takeprofit_reached(current_candle) {
                        let result = trade.close_on_takeprofit(current_candle);
                        total_profit += result;
                        balance += trade.initial_position_size + result;
                        current_trade = None;
                    }
                }
            }
            model.reset();
        }

        if trade_count == 0 {
            -self.initial_usd_balance
        } else {
            total_profit
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_candlesticks_for_validation_and_training() {
        let candlesticks = (0..100).map(|_| Candlestick::new()).collect();
        let backtest_engine = Backtest::new(candlesticks, 20, 0.005, 0.01);
        assert_eq!(backtest_engine.validation_indices.len(), 2);
        assert_eq!(
            backtest_engine.training_indices.len() + backtest_engine.validation_indices.len(),
            backtest_engine.all_ranges.len()
        );
    }
}
