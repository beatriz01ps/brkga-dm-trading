use std::sync::Arc;

use crate::candlestick::Candlestick;
use crate::backtest::trade::TradeDirection;
use crate::utils::{map_range, percentage_difference};
use crate::indicator_cache::IndicatorCache;
use crate::indicators::moving_average_convergence_divergence::MovingAverageConvergenceDivergence as Macd;
use crate::indicators::traits::Next;

pub struct TradeRule {
    direction: TradeDirection,

    rsi_per_range:        Vec<Arc<Vec<f32>>>,
    ema_per_range:        Vec<Arc<Vec<f32>>>,
    sma_per_range:        Vec<Arc<Vec<f32>>>,
    atr_tp_per_range:     Vec<Arc<Vec<f32>>>,
    atr_sl_per_range:     Vec<Arc<Vec<f32>>>,
    tp_extreme_per_range: Vec<Arc<Vec<f32>>>,
    sl_extreme_per_range: Vec<Arc<Vec<f32>>>,

    macd: Macd,

    macd_target_value: f32,
    rsi_lower_bound: f32,
    rsi_higher_bound: f32,
    pub ema_min_percentage_diff_from_price: f32,
    pub sma_max_percentage_diff_from_ema: f32,
    take_profit_target_atr: f32,
    stop_loss_target_atr: f32,

    ranges: Vec<(u32, u32)>,
    current_range_idx: usize,
    current_relative_idx: usize,
}

impl TradeRule {
    pub fn new(direction: TradeDirection, cromossome: &[f32], cache: &Arc<IndicatorCache>) -> Self {
        let atr_tp_period = map_range((2.0, 100.0), cromossome[1]) as usize;
        let atr_sl_period = map_range((2.0,  50.0), cromossome[3]) as usize;
        let rsi_period    = map_range((0.0, 100.0), cromossome[4]) as usize;
        let rsi_b1        = map_range((0.0, 100.0), cromossome[5]);
        let rsi_b2        = map_range((0.0, 100.0), cromossome[6]);
        let macd_fast     = map_range((2.0, 100.0), cromossome[7]) as usize;
        let macd_slow     = map_range((2.0, 100.0), cromossome[8]) as usize;
        let macd_sig      = map_range((2.0, 100.0), cromossome[9]) as usize;
        let ema_period    = map_range((2.0, 100.0), cromossome[11]) as usize;
        let sma_period    = map_range((2.0, 100.0), cromossome[13]) as usize;
        let tp_period     = map_range((1.0, 100.0), cromossome[15]) as usize;
        let sl_period     = map_range((1.0, 100.0), cromossome[16]) as usize;

        let ranges = cache.ranges().to_vec();
        let num_ranges = ranges.len();

        let mut rsi_per_range        = Vec::with_capacity(num_ranges);
        let mut ema_per_range        = Vec::with_capacity(num_ranges);
        let mut sma_per_range        = Vec::with_capacity(num_ranges);
        let mut atr_tp_per_range     = Vec::with_capacity(num_ranges);
        let mut atr_sl_per_range     = Vec::with_capacity(num_ranges);
        let mut tp_extreme_per_range = Vec::with_capacity(num_ranges);
        let mut sl_extreme_per_range = Vec::with_capacity(num_ranges);

        for r in 0..num_ranges {
            rsi_per_range.push(cache.get_rsi(r, rsi_period));
            ema_per_range.push(cache.get_ema(r, ema_period));
            sma_per_range.push(cache.get_sma_close(r, sma_period));
            atr_tp_per_range.push(cache.get_atr(r, atr_tp_period));
            atr_sl_per_range.push(cache.get_atr(r, atr_sl_period));
            match direction {
                TradeDirection::Long => {
                    tp_extreme_per_range.push(cache.get_roll_max_high(r, tp_period));
                    sl_extreme_per_range.push(cache.get_roll_min_low(r, sl_period));
                }
                TradeDirection::Short => {
                    tp_extreme_per_range.push(cache.get_roll_min_low(r, tp_period));
                    sl_extreme_per_range.push(cache.get_roll_max_high(r, sl_period));
                }
            }
        }

        TradeRule {
            direction,
            rsi_per_range, ema_per_range, sma_per_range,
            atr_tp_per_range, atr_sl_per_range,
            tp_extreme_per_range, sl_extreme_per_range,
            macd: Macd::new(macd_fast.max(1), macd_slow.max(1), macd_sig.max(1)),
            macd_target_value: map_range((-1000.0, 1000.0), cromossome[10]),
            rsi_lower_bound:   rsi_b1.max(rsi_b2),
            rsi_higher_bound:  rsi_b1.min(rsi_b2),
            ema_min_percentage_diff_from_price: map_range((0.1, 100.0), cromossome[12]),
            sma_max_percentage_diff_from_ema:   map_range((0.1, 100.0), cromossome[14]),
            take_profit_target_atr: map_range((0.1, 20.0), cromossome[0]),
            stop_loss_target_atr:   map_range((0.1, 20.0), cromossome[2]),
            ranges,
            current_range_idx: 0,
            current_relative_idx: 0,
        }
    }

    pub fn set_range(&mut self, range_idx: usize) {
        self.current_range_idx = range_idx;
        self.macd.reset();
    }

    pub fn warmup(&mut self, candle: &Candlestick) {
        self.macd.next(candle);
    }

    pub fn reset(&mut self) {}

    pub fn evaluate(&mut self, idx: usize, candle: &Candlestick) -> bool {
        let range_start = self.ranges[self.current_range_idx].0 as usize;
        let relative_idx = idx - range_start;
        self.current_relative_idx = relative_idx;

        let r = self.current_range_idx;
        if relative_idx >= self.rsi_per_range[r].len() { return false; }

        let macd_out = self.macd.next(candle);
        let rsi = self.rsi_per_range[r][relative_idx];
        let ema = self.ema_per_range[r][relative_idx];
        let sma = self.sma_per_range[r][relative_idx];
        let pd_ema = percentage_difference(ema, candle.close);
        let pd_sma = percentage_difference(sma, ema);

        // E1 (original, resultado-e1.log): macd && rsi && pd_ema || pd_sma
        // E2 (abaixo, resultado-e2.log): (macd && rsi) || (pd_ema && pd_sma)
        (macd_out.signal > self.macd_target_value &&
        rsi > self.rsi_higher_bound && rsi < self.rsi_lower_bound) ||
        (pd_ema >= self.ema_min_percentage_diff_from_price &&
        pd_sma <= self.sma_max_percentage_diff_from_ema)
    }

    pub fn evaluate_take_profit(&self) -> f32 {
        let r    = self.current_range_idx;
        let ridx = self.current_relative_idx;
        let atr  = self.atr_tp_per_range[r].get(ridx).copied().unwrap_or(0.0);
        let ext  = self.tp_extreme_per_range[r].get(ridx).copied().unwrap_or(0.0);
        match self.direction {
            TradeDirection::Long  => ext + atr * self.take_profit_target_atr,
            TradeDirection::Short => ext - atr * self.take_profit_target_atr,
        }
    }

    pub fn evaluate_stop_loss(&self) -> f32 {
        let r    = self.current_range_idx;
        let ridx = self.current_relative_idx;
        let atr  = self.atr_sl_per_range[r].get(ridx).copied().unwrap_or(0.0);
        let ext  = self.sl_extreme_per_range[r].get(ridx).copied().unwrap_or(0.0);
        match self.direction {
            TradeDirection::Long  => ext - atr * self.stop_loss_target_atr,
            TradeDirection::Short => ext + atr * self.stop_loss_target_atr,
        }
    }
}
