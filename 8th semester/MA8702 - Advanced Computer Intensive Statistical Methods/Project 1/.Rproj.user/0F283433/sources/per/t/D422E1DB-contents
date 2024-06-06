data {
  int<lower=0> Pump; // Pump
  int y[Pump]; // Number of failures
  real t[Pump]; // Time
}

parameters {
  real<lower=0> alpha; // Shape parameter
  real<lower=0> beta; // Rate parameter
  real<lower=0> lambda[Pump]; // Intensity
}

transformed parameters {
  real lambdat[Pump]; // Intensity * time
  for (i in 1:Pump) {
    lambdat[i] = lambda[i] * t[i];
  }
}

model {
// log of hyper-priors
  target += exponential_lpdf(alpha | 1);
  target += gamma_lpdf(beta | 0.1, 1);
// log of prior
  target += gamma_lpdf(lambda | alpha, beta);
// log of likelihood
  target += poisson_lpmf(y | lambdat);
}

