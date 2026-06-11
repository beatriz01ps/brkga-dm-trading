use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::candlestick::Candlestick;
use crate::indicators::traits::Next;
use crate::indicators::relative_strength_index::RelativeStrengthIndex;
use crate::indicators::exponential_moving_average::ExponentialMovingAverage;
use crate::indicators::simple_moving_average::SimpleMovingAverage;
use crate::indicators::average_true_range::AverageTrueRange;

// MACD não é cacheado: 3 parâmetros livres em [2,100] → até 99³ ≈ 970k combinações.
// Com 30k indivíduos aleatórios na gen 0, quase todas são únicas → risco de OOM.
// Todos os outros indicadores são cacheados por (range_idx, period).
// A série de range_idx=r é calculada a partir de candles[ranges[r].0..ranges[r].1],
// replicando exatamente o comportamento original de reset por range.
//
// Bound de memória (pior caso, todos os 99 períodos populados):
//   6 tipos × num_ranges × 99 períodos × (range_len × 4 bytes)
// Com 11 ranges de ~17.500 candles cada:
//   6 × 11 × 99 × 17.500 × 4 ≈ 274 MB

pub struct IndicatorCache {
    candles:       Vec<Candlestick>,
    ranges:        Vec<(u32, u32)>,
    rsi:           RwLock<HashMap<(usize, usize), Arc<Vec<f32>>>>,
    ema:           RwLock<HashMap<(usize, usize), Arc<Vec<f32>>>>,
    sma_close:     RwLock<HashMap<(usize, usize), Arc<Vec<f32>>>>,
    atr:           RwLock<HashMap<(usize, usize), Arc<Vec<f32>>>>,
    roll_max_high: RwLock<HashMap<(usize, usize), Arc<Vec<f32>>>>,
    roll_min_low:  RwLock<HashMap<(usize, usize), Arc<Vec<f32>>>>,
}

impl IndicatorCache {
    pub fn new(candles: Vec<Candlestick>, ranges: Vec<(u32, u32)>) -> Self {
        Self {
            candles,
            ranges,
            rsi:           RwLock::new(HashMap::new()),
            ema:           RwLock::new(HashMap::new()),
            sma_close:     RwLock::new(HashMap::new()),
            atr:           RwLock::new(HashMap::new()),
            roll_max_high: RwLock::new(HashMap::new()),
            roll_min_low:  RwLock::new(HashMap::new()),
        }
    }

    pub fn ranges(&self) -> &[(u32, u32)] { &self.ranges }

    pub fn get_rsi(&self, range_idx: usize, period: usize) -> Arc<Vec<f32>> {
        let (start, end) = self.ranges[range_idx];
        self.get1(&self.rsi, (range_idx, period), || {
            let mut ind = RelativeStrengthIndex::new(period.max(1));
            self.candles[start as usize..end as usize]
                .iter().map(|c| ind.next(c.close)).collect()
        })
    }

    pub fn get_ema(&self, range_idx: usize, period: usize) -> Arc<Vec<f32>> {
        let (start, end) = self.ranges[range_idx];
        self.get1(&self.ema, (range_idx, period), || {
            let mut ind = ExponentialMovingAverage::new(period.max(1));
            self.candles[start as usize..end as usize]
                .iter().map(|c| ind.next(c)).collect()
        })
    }

    pub fn get_sma_close(&self, range_idx: usize, period: usize) -> Arc<Vec<f32>> {
        let (start, end) = self.ranges[range_idx];
        self.get1(&self.sma_close, (range_idx, period), || {
            let mut ind = SimpleMovingAverage::new(period.max(1));
            self.candles[start as usize..end as usize]
                .iter().map(|c| ind.next(c.close)).collect()
        })
    }

    pub fn get_atr(&self, range_idx: usize, period: usize) -> Arc<Vec<f32>> {
        let (start, end) = self.ranges[range_idx];
        self.get1(&self.atr, (range_idx, period), || {
            let mut ind = AverageTrueRange::new(period.max(1));
            self.candles[start as usize..end as usize]
                .iter().map(|c| ind.next(c)).collect()
        })
    }

    pub fn get_roll_max_high(&self, range_idx: usize, period: usize) -> Arc<Vec<f32>> {
        let (start, end) = self.ranges[range_idx];
        self.get1(&self.roll_max_high, (range_idx, period), || {
            let highs: Vec<f32> = self.candles[start as usize..end as usize]
                .iter().map(|c| c.high).collect();
            rolling_max(&highs, period.max(1))
        })
    }

    pub fn get_roll_min_low(&self, range_idx: usize, period: usize) -> Arc<Vec<f32>> {
        let (start, end) = self.ranges[range_idx];
        self.get1(&self.roll_min_low, (range_idx, period), || {
            let lows: Vec<f32> = self.candles[start as usize..end as usize]
                .iter().map(|c| c.low).collect();
            rolling_min(&lows, period.max(1))
        })
    }

    // Compute dentro do write lock: apenas uma thread paga o custo por (range, period).
    fn get1(
        &self,
        cache: &RwLock<HashMap<(usize, usize), Arc<Vec<f32>>>>,
        key: (usize, usize),
        compute: impl Fn() -> Vec<f32>,
    ) -> Arc<Vec<f32>> {
        {
            let g = cache.read().unwrap();
            if let Some(v) = g.get(&key) { return v.clone(); }
        }
        let mut g = cache.write().unwrap();
        if let Some(v) = g.get(&key) { return v.clone(); }
        let series = Arc::new(compute());
        g.insert(key, series.clone());
        series
    }
}

fn rolling_max(values: &[f32], period: usize) -> Vec<f32> {
    use std::collections::VecDeque;
    let mut result = Vec::with_capacity(values.len());
    let mut dq: VecDeque<usize> = VecDeque::new();
    for i in 0..values.len() {
        while dq.front().map_or(false, |&f| f + period <= i) { dq.pop_front(); }
        while dq.back().map_or(false, |&b| values[b] <= values[i]) { dq.pop_back(); }
        dq.push_back(i);
        result.push(values[*dq.front().unwrap()]);
    }
    result
}

fn rolling_min(values: &[f32], period: usize) -> Vec<f32> {
    use std::collections::VecDeque;
    let mut result = Vec::with_capacity(values.len());
    let mut dq: VecDeque<usize> = VecDeque::new();
    for i in 0..values.len() {
        while dq.front().map_or(false, |&f| f + period <= i) { dq.pop_front(); }
        while dq.back().map_or(false, |&b| values[b] >= values[i]) { dq.pop_back(); }
        dq.push_back(i);
        result.push(values[*dq.front().unwrap()]);
    }
    result
}
