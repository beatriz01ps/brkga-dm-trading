use rayon::prelude::*;
use rand::prelude::*;
use rand_pcg::{Pcg64, Lcg128Xsl64};
use std::time::{Instant, Duration};
use std::sync::Arc;
use std::fs;
use std::io::Write as IoWrite;
use crate::candlestick::{Candlestick, load_candlesticks};
use crate::backtest::{Backtest, RunMode};
use crate::backtest::strategy::SingleStrategy;
use crate::indicator_cache::IndicatorCache;
use crate::xmeans::xmeans_fit;

pub struct BRKGA {
    fraction_top: f32,
    fraction_bottom: f32,
    population_size: usize,
    max_iterations: usize,
    elitism_rate: f32,
    rng: Lcg128Xsl64,
    cromossome_size: usize,
    population: Vec<Individual>,
    fitness_executor: FitnessExecutor,
    warmup_generations: usize,
    dm_history_size: usize,
    elite_history: Vec<Vec<f32>>,
    validation_top_k: usize,
}

trait FitnessFunction {
    fn fitness(&self, cromossome: Vec<f32>) -> f32;
}

pub type BrkgaConfig = (f32, f32, usize, usize, f32);

impl BRKGA {
    pub fn new(seed: u64, cromossome_size: usize, config: BrkgaConfig, fitness_executor: FitnessExecutor, validation_top_k: usize) -> Self {
        Self {
            fraction_top: config.0,
            fraction_bottom: config.1,
            population_size: config.2,
            max_iterations: config.3,
            elitism_rate: config.4,
            cromossome_size,
            fitness_executor,
            rng: Pcg64::seed_from_u64(seed),
            population: vec![],
            warmup_generations: 30,
            dm_history_size: 20,
            elite_history: vec![],
            validation_top_k,
        }
    }

    fn initial_population(&mut self) -> Vec<Individual> {
        (0..self.population_size).map(|_| self.random_individual()).collect()
    }

    pub fn random_individual(&mut self) -> Individual {
        Individual::new((0..self.cromossome_size).map(|_| self.rng.gen_range(0.0..1.0)).collect())
    }

    fn generate_mutants(&mut self) -> Vec<Individual> {
        let amount_mutants = (self.population_size as f32 * self.fraction_bottom) as usize;
        (0..amount_mutants).map(|_| self.random_individual()).collect()
    }

    fn get_elite_population(&self) -> Vec<Individual> {
        let amount_elite = (self.population_size as f32 * self.fraction_top) as usize;
        let elite_start = self.population_size - amount_elite;
        self.population[elite_start..].to_vec()
    }

    fn sort_population(&mut self) {
        self.population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
    }

    fn calculate_population_fitness(&mut self) {
        self.population.par_iter_mut()
            .filter(|ind| ind.fitness.is_none())
            .for_each(|ind| {
                ind.fitness = Some(self.fitness_executor.calculate_fitness(ind.cromossome.as_slice()));
            });
    }

    fn crossover(&mut self, index_elite_parent: usize, index_non_elite_parent: usize) -> Individual {
        let mut child = Individual::new(Vec::with_capacity(self.cromossome_size));
        for i in 0..self.cromossome_size {
            if self.rng.gen_range(0.0..1.0) < self.elitism_rate {
                child.cromossome.push(self.population[index_elite_parent].cromossome[i]);
            } else {
                child.cromossome.push(self.population[index_non_elite_parent].cromossome[i]);
            }
        }
        child
    }

    fn evolve_population(&mut self, generation: usize) -> Vec<Individual> {
        if generation >= self.warmup_generations {
            let best_cromossome = self.population[self.population_size - 1].cromossome.clone();
            self.elite_history.push(best_cromossome);
            if self.elite_history.len() > self.dm_history_size {
                self.elite_history.remove(0);
            }
        }

        let mut new_population: Vec<Individual> = Vec::with_capacity(self.population_size);
        new_population.extend(self.generate_mutants());

        if !self.elite_history.is_empty() {
            let dm_cromossome = self.compute_dm_x();
            new_population.push(Individual::new(dm_cromossome));
        }

        new_population.extend(self.get_elite_population());
        let amount_to_reproduce = self.population_size - new_population.len();
        for _ in 0..amount_to_reproduce {
            let e = self.get_random_elite();
            let n = self.get_random_non_elite();
            new_population.push(self.crossover(e, n));
        }
        new_population
    }

    /// DMC-GRASP-X: per-dimension 1D x-means; pick first cluster's mean for each gene.
    fn compute_dm_x(&mut self) -> Vec<f32> {
        let max_clusters = 40;
        let mut result = Vec::with_capacity(self.cromossome_size);
        for gene_idx in 0..self.cromossome_size {
            let values: Vec<Vec<f32>> = self.elite_history.iter()
                .map(|c| vec![c[gene_idx]])
                .collect();
            let clusters = xmeans_fit(&values, max_clusters, &mut self.rng);
            let selected = &clusters[0];
            let n = selected.len() as f32;
            let mean = selected.iter().map(|&i| values[i][0]).sum::<f32>() / n;
            result.push(mean);
        }
        result
    }

    fn get_random_elite(&mut self) -> usize {
        let amount_elite = (self.population_size as f32 * self.fraction_top) as usize;
        let elite_start = self.population_size - amount_elite;
        self.rng.gen_range(elite_start..self.population_size)
    }

    fn get_random_non_elite(&mut self) -> usize {
        let amount_elite = (self.population_size as f32 * self.fraction_top) as usize;
        let elite_start = self.population_size - amount_elite;
        self.rng.gen_range(0..elite_start)
    }

    pub fn run(&mut self) {
        println!("Starting BRKGA with a population of {}", self.population_size);
        fs::create_dir_all("results").ok();
        init_csv("results/generations.csv");

        let start = Instant::now();
        self.population = self.initial_population();

        let mut best_training_ever: f32 = f32::NEG_INFINITY;
        let mut best_validation_ever: f32 = f32::NEG_INFINITY;
        let mut best_training_gen: usize = 0;
        let mut best_validation_gen: usize = 0;

        for i in 0..self.max_iterations {
            let gen_start = Instant::now();
            self.calculate_population_fitness();
            self.sort_population();
            let gen_time = gen_start.elapsed();
            let total_time = start.elapsed();

            let (train_fit, val_fit) = self.show_details(
                i, best_training_ever, best_validation_ever, gen_time, total_time);

            if train_fit > best_training_ever {
                best_training_ever = train_fit;
                best_training_gen = i;
            }
            if val_fit > best_validation_ever {
                best_validation_ever = val_fit;
                best_validation_gen = i;
            }

            let dm_active = i >= self.warmup_generations;
            append_csv("results/generations.csv", i, train_fit,
                self.population[self.population_size/2].fitness.unwrap(),
                self.population[0].fitness.unwrap(),
                val_fit, gen_time, total_time, dm_active);

            self.population = self.evolve_population(i);
        }

        let duration = start.elapsed();
        println!("\n--- Resultado Final ---");
        println!("Melhor treino ever:    {:10.2} (geração {})", best_training_ever, best_training_gen);
        println!("Melhor validação ever: {:10.2} (geração {})", best_validation_ever, best_validation_gen);
        println!("Tempo total: {:?}", duration);
    }

    fn show_details(&self, generation: usize, best_training_ever: f32, best_validation_ever: f32,
                    gen_time: Duration, total_time: Duration) -> (f32, f32) {
        let best   = &self.population[self.population_size - 1];
        let median = &self.population[self.population_size / 2];
        let worst  = &self.population[0];

        let val_best = self.fitness_executor.validation_fitness(best.cromossome.as_slice());

        let k = self.validation_top_k.min(self.population_size);
        let val_topk = self.population[self.population_size - k..]
            .iter()
            .map(|ind| self.fitness_executor.validation_fitness(ind.cromossome.as_slice()))
            .fold(f32::NEG_INFINITY, f32::max);

        println!(
            "Gen {:4}: train={:9.2} | median={:9.2} | worst={:9.2} | val={:9.2} | val_top{}={:9.2} | best_train_ever={:9.2} | best_val_ever={:9.2} | gen={:.1}s | total={:.0}s",
            generation,
            best.fitness.unwrap(), median.fitness.unwrap(), worst.fitness.unwrap(),
            val_best, k, val_topk, best_training_ever, best_validation_ever,
            gen_time.as_secs_f32(), total_time.as_secs_f32()
        );

        (best.fitness.unwrap(), val_topk)
    }
}

fn init_csv(path: &str) {
    let mut f = fs::File::create(path).unwrap();
    writeln!(f, "generation,train_best,train_median,train_worst,val_best,gen_time_s,total_time_s,dm_active").ok();
}

fn append_csv(path: &str, gen: usize, train_best: f32, train_median: f32,
              train_worst: f32, val_best: f32, gen_time: Duration, total_time: Duration, dm_active: bool) {
    let mut f = fs::OpenOptions::new().append(true).open(path).unwrap();
    writeln!(f, "{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{}",
        gen, train_best, train_median, train_worst, val_best,
        gen_time.as_secs_f32(), total_time.as_secs_f32(), dm_active as u8).ok();
}

pub struct Individual {
    pub fitness: Option<f32>,
    pub cromossome: Vec<f32>,
}

impl Individual {
    pub fn new(cromossome: Vec<f32>) -> Self {
        Self { fitness: None, cromossome }
    }
}

impl Clone for Individual {
    fn clone(&self) -> Individual {
        Individual { fitness: self.fitness, cromossome: self.cromossome.clone() }
    }
}

pub struct FitnessExecutor {
    backtester: Backtest,
    mode: RunMode,
    cache: Arc<IndicatorCache>,
}

impl FitnessExecutor {
    pub fn new(backtester: Backtest, mode: RunMode) -> Self {
        let all_ranges = backtester.all_ranges().to_vec();
        let candles = backtester.candles_clone();
        let cache = Arc::new(IndicatorCache::new(candles, all_ranges));
        Self { backtester, mode, cache }
    }

    pub fn calculate_fitness(&self, cromossome: &[f32]) -> f32 {
        let mut model = SingleStrategy::decode(cromossome, &self.cache);
        self.backtester.run(self.mode, &mut model)
    }

    pub fn validation_fitness(&self, cromossome: &[f32]) -> f32 {
        let mut model = SingleStrategy::decode(cromossome, &self.cache);
        self.backtester.run(RunMode::Validation, &mut model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_brkga() {
        let config: BrkgaConfig = (0.1, 0.2, 15000, 1000, 0.6);
        let candles = load_candlesticks("test_files/ADAUSDT-30m.csv").unwrap();
        let backtest_engine = Backtest::new(candles, 12, 0.005, 0.02);
        let brkga = BRKGA::new(1223, 35, config, FitnessExecutor::new(backtest_engine, RunMode::Training), 50);
        assert_eq!(brkga.cromossome_size, 35);
        assert_eq!(brkga.fraction_top, 0.1);
        assert_eq!(brkga.fraction_bottom, 0.2);
        assert_eq!(brkga.population_size, 15000);
        assert_eq!(brkga.max_iterations, 1000);
        assert_eq!(brkga.elitism_rate, 0.6);
    }

    #[test]
    fn test_generate_random_invidiual() {
        let config: BrkgaConfig = (0.1, 0.2, 15000, 1000, 0.6);
        let candles = load_candlesticks("test_files/ADAUSDT-30m.csv").unwrap();
        let backtest_engine = Backtest::new(candles, 12, 0.005, 0.02);
        let mut brkga = BRKGA::new(59841, 35, config, FitnessExecutor::new(backtest_engine, RunMode::Training), 50);
        let first_individual = brkga.random_individual();
        assert_eq!(first_individual.cromossome.len(), 35);
        for i in 0..first_individual.cromossome.len() {
            assert!(first_individual.cromossome[i] >= 0.0 && first_individual.cromossome[i] <= 1.0);
        }
        assert_eq!(0.66094065, first_individual.cromossome[0]);
        assert_eq!(0.24245393, first_individual.cromossome[34]);
        let second_individual = brkga.random_individual();
        assert_ne!(first_individual.cromossome, second_individual.cromossome);
    }
}
