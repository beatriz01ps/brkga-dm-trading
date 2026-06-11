use rand::prelude::*;
use rand_pcg::Lcg128Xsl64;
use std::f32::consts::PI;

fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

pub fn kmeans_plusplus_init(data: &[Vec<f32>], k: usize, rng: &mut Lcg128Xsl64) -> Vec<Vec<f32>> {
    if data.is_empty() || k == 0 { return vec![]; }
    let k = k.min(data.len());
    let mut centers: Vec<Vec<f32>> = Vec::with_capacity(k);

    let first = rng.gen_range(0..data.len());
    centers.push(data[first].clone());

    for _ in 1..k {
        let dists: Vec<f32> = data.iter().map(|x| {
            centers.iter().map(|c| sq_dist(x, c)).fold(f32::MAX, f32::min)
        }).collect();

        let total: f32 = dists.iter().sum();
        if total <= 0.0 {
            centers.push(data[rng.gen_range(0..data.len())].clone());
            continue;
        }

        let mut threshold = rng.gen_range(0.0..total);
        let mut chosen = data.len() - 1;
        for (i, &d) in dists.iter().enumerate() {
            threshold -= d;
            if threshold <= 0.0 { chosen = i; break; }
        }
        centers.push(data[chosen].clone());
    }
    centers
}

fn assign_clusters(data: &[Vec<f32>], centers: &[Vec<f32>]) -> Vec<Vec<usize>> {
    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); centers.len()];
    for (i, point) in data.iter().enumerate() {
        let nearest = centers.iter().enumerate()
            .min_by(|(_, a), (_, b)| sq_dist(point, a).partial_cmp(&sq_dist(point, b)).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        clusters[nearest].push(i);
    }
    clusters
}

fn recompute_centers(data: &[Vec<f32>], clusters: &[Vec<usize>]) -> Vec<Vec<f32>> {
    let d = data.first().map(|v| v.len()).unwrap_or(1);
    clusters.iter().map(|cluster| {
        if cluster.is_empty() { return vec![0.5; d]; }
        let n = cluster.len() as f32;
        (0..d).map(|di| cluster.iter().map(|&i| data[i][di]).sum::<f32>() / n).collect()
    }).collect()
}

pub fn kmeans_fit(data: &[Vec<f32>], init_centers: Vec<Vec<f32>>, max_iter: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let mut centers = init_centers;
    let mut clusters = assign_clusters(data, &centers);
    for _ in 0..max_iter {
        let new_centers = recompute_centers(data, &clusters);
        let converged = centers.iter().zip(new_centers.iter()).all(|(a, b)| sq_dist(a, b) < 1e-10);
        centers = new_centers;
        clusters = assign_clusters(data, &centers);
        if converged { break; }
    }
    (clusters, centers)
}

fn bic_score(data: &[Vec<f32>], clusters: &[Vec<usize>], centers: &[Vec<f32>]) -> f32 {
    let n = data.len() as f32;
    let k = clusters.len() as f32;
    let d = data.first().map(|v| v.len()).unwrap_or(1) as f32;

    if n <= k { return f32::NEG_INFINITY; }

    let rss: f32 = clusters.iter().zip(centers.iter())
        .map(|(cl, c)| cl.iter().map(|&i| sq_dist(&data[i], c)).sum::<f32>())
        .sum();

    let sigma2 = rss / ((n - k) * d).max(1e-10);
    if sigma2 <= 0.0 { return f32::INFINITY; }

    let log_like: f32 = clusters.iter().map(|cl| {
        let nj = cl.len() as f32;
        if nj == 0.0 { 0.0 } else { nj * (nj / n).ln() }
    }).sum::<f32>()
        - n * d / 2.0 * (2.0 * PI).ln()
        - n * d / 2.0 * sigma2.ln()
        - rss / (2.0 * sigma2);

    let num_params = k * d + k;
    log_like - 0.5 * num_params * n.ln()
}

/// X-means: BIC-guided recursive splitting starting from k=2.
/// Returns cluster assignments (Vec of Vecs of indices into `data`).
pub fn xmeans_fit(data: &[Vec<f32>], max_clusters: usize, rng: &mut Lcg128Xsl64) -> Vec<Vec<usize>> {
    let n = data.len();
    if n == 0 { return vec![]; }
    if n == 1 { return vec![vec![0]]; }
    if max_clusters < 2 { return vec![(0..n).collect()]; }

    let init = kmeans_plusplus_init(data, 2.min(n), rng);
    let (mut clusters, mut centers) = kmeans_fit(data, init, 100);

    loop {
        let current_k = clusters.len();
        if current_k >= max_clusters { break; }

        let mut new_centers: Vec<Vec<f32>> = Vec::new();
        let mut split_occurred = false;

        for ci in 0..current_k {
            if clusters[ci].len() < 2 {
                new_centers.push(centers[ci].clone());
                continue;
            }

            let sub_data: Vec<Vec<f32>> = clusters[ci].iter().map(|&i| data[i].clone()).collect();
            let child_init = kmeans_plusplus_init(&sub_data, 2, rng);
            let (child_clusters, child_centers) = kmeans_fit(&sub_data, child_init, 100);

            let parent_cluster = vec![(0..sub_data.len()).collect::<Vec<_>>()];
            let bic1 = bic_score(&sub_data, &parent_cluster, &[centers[ci].clone()]);
            let bic2 = bic_score(&sub_data, &child_clusters, &child_centers);

            if bic2 > bic1 && new_centers.len() + 2 <= max_clusters {
                new_centers.extend(child_centers);
                split_occurred = true;
            } else {
                new_centers.push(centers[ci].clone());
            }
        }

        if !split_occurred { break; }

        let (new_clusters, new_centers_upd) = kmeans_fit(data, new_centers, 100);
        clusters = new_clusters;
        centers = new_centers_upd;
    }

    clusters
}
