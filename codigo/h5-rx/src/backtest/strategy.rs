use std::sync::Arc;

use crate::utils::map_range;
use crate::candlestick::Candlestick;
use crate::backtest::trade::TradeDirection;
use crate::backtest::trade_rule::TradeRule;
use crate::indicator_cache::IndicatorCache;

pub trait TradingStrategy {
    fn set_range(&mut self, range_idx: usize);
    fn warmup(&mut self, candle: &Candlestick);
    fn new_candlestick(&mut self, idx: usize, candle: &Candlestick);
    fn should_start_trade(&mut self) -> Option<(TradeDirection, f32, f32)>;
    fn reset(&mut self);
    fn percentage_amount_per_trade(&self) -> f32;
    fn leverage(&self) -> u8;
}

pub struct SingleStrategy {
    leverage: u8,
    long_rule: TradeRule,
    short_rule: TradeRule,
    percentage_amount_per_trade: f32,
    start_long_trade: bool,
    start_short_trade: bool,
}

impl TradingStrategy for SingleStrategy {
    fn set_range(&mut self, range_idx: usize) {
        self.long_rule.set_range(range_idx);
        self.short_rule.set_range(range_idx);
    }

    fn warmup(&mut self, candle: &Candlestick) {
        self.long_rule.warmup(candle);
        self.short_rule.warmup(candle);
    }

    fn new_candlestick(&mut self, idx: usize, candle: &Candlestick) {
        self.start_long_trade  = self.long_rule.evaluate(idx, candle);
        self.start_short_trade = self.short_rule.evaluate(idx, candle);
    }

    fn should_start_trade(&mut self) -> Option<(TradeDirection, f32, f32)> {
        if self.start_long_trade && self.start_short_trade {
            return None;
        } else if self.start_long_trade {
            return Some((TradeDirection::Long,
                self.long_rule.evaluate_take_profit(),
                self.long_rule.evaluate_stop_loss()));
        } else if self.start_short_trade {
            return Some((TradeDirection::Short,
                self.short_rule.evaluate_take_profit(),
                self.short_rule.evaluate_stop_loss()));
        }
        None
    }

    fn reset(&mut self) {
        self.long_rule.reset();
        self.short_rule.reset();
    }

    fn percentage_amount_per_trade(&self) -> f32 { self.percentage_amount_per_trade }
    fn leverage(&self) -> u8 { self.leverage }
}

impl SingleStrategy {
    pub fn decode(cromossome: &[f32], cache: &Arc<IndicatorCache>) -> Self {
        if cromossome.len() != 35 {
            panic!("the cromossome must have {} genes, but it had {}", 35, cromossome.len());
        }
        for &g in cromossome {
            if g < 0.0 || g > 1.0 {
                panic!("gene out of range: {}", g);
            }
        }
        SingleStrategy {
            leverage: map_range((1.0, 60.0), cromossome[0]) as u8,
            long_rule:  TradeRule::new(TradeDirection::Long,  &cromossome[1..=17], cache),
            short_rule: TradeRule::new(TradeDirection::Short, &cromossome[18..=34], cache),
            percentage_amount_per_trade: 0.015,
            start_long_trade: false,
            start_short_trade: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_single_strategy() {
        let strategy_leverage = map_range((1.0, 60.0), 0.3) as u8;
        assert_eq!(strategy_leverage, 18);
    }
}
