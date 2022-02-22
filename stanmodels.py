hybrid_gershman = """
data {
  int<lower=1> N;      // Number of subjects
  int<lower=1> T;      // Number of trials
  int<lower=1, upper=T> Tsub[N]; // number of trials per subject
  real<lower=-1, upper=2> choice[N, T]; // choice
  int<lower=-1, upper=2> choiceidx[N, T]; // choiceidx
  int game[N, T]; // game
  int reward[N, T];  // reward
}
transformed data {
  vector[2] initV;  // initial values for EV
  real oSD;
  initV = rep_vector(0.5, 2);
  oSD = 0.433;
}
parameters {
  // Hyper(group)-parameters
  vector[3] mu_pr;
  vector<lower=0>[3] sigma;

  // Subject-level raw parameters (for Matt trick)
  vector[N] gamma_pr;
  vector[N] beta_pr;
  vector[N] initSD_pr;
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0, upper=2>[N]       gamma; // directed exploration weight
  vector<lower=0, upper=2>[N]       beta; // random exploration weight
  vector<lower=0, upper=15>[N]      initSD; // initial uncertainty

  for (i in 1:N) {
    gamma[i] = Phi_approx(mu_pr[1] + sigma[1] * gamma_pr[i]) * 2;
    beta[i]   = Phi_approx(mu_pr[2] + sigma[2] * beta_pr[i]) * 2;
    initSD[i] = Phi_approx(mu_pr[3] + sigma[3] * initSD_pr[i]) * 15;
  }
}

model {
  // Hyperparameters
  mu_pr  ~ normal(0, 1);
  sigma ~ cauchy(0, 5);

  // Individual parameters for non-centered parameterization
  gamma_pr ~ normal(0, 1);
  beta_pr ~ normal(0, 1);
  initSD_pr ~ normal(0, 1);

  // Begin subject loop
  for (i in 1:N) {
    vector[2] ev; // expected value
    vector[2] vari; // expected variance
    int gamemem; // game memory
    real K; // Kalman gain
    real PE; // prediction error
    gamemem = 0;
    for (t in 1:(Tsub[i])) {
      if (game[i, t] != gamemem) {
        ev = initV;
        vari = rep_vector(initSD[i]^2, 2);
        gamemem = game[i, t];
      }
      if (t >= 5) {
        real center = beta[i] * (ev[1] - ev[2]) / sqrt(vari[1] + vari[2]) + gamma[i] * (sqrt(vari[1]) - sqrt(vari[2]));
        choice[i, t] ~ normal(center, 1);
      }
      PE = reward[i, t] - ev[choiceidx[i, t]];
      K = vari[choiceidx[i, t]] / (vari[choiceidx[i, t]] + oSD^2);
      ev[choiceidx[i, t]] += K * PE;
      vari[choiceidx[i, t]] -= K * vari[choiceidx[i, t]];
    }
  } // end of subject loop
}

generated quantities {
  // For group level parameters
  real<lower=0, upper=2>    mu_gamma; // directed exploration weight
  real<lower=0, upper=2>    mu_beta; // random exploration weight
  real<lower=0, upper=15>   mu_initSD;     // initial uncertainty

  // For log likelihood calculation
  real log_lik[N];
  real ev[N, T+1, 2]; // expected probability of reward
  real vari[N, T+1, 2]; // expected variance
  real K[N, T]; // Kalman gain

  // Assign group level parameter values
  mu_gamma = Phi_approx(mu_pr[1]) * 2;
  mu_beta = Phi_approx(mu_pr[2]) * 2;
  mu_initSD = Phi_approx(mu_pr[3]) * 15;

  { // local section, this saves time and space
    // Begin subject loop
    for (i in 1:N) {
      real PE; // prediction error
      int gamemem; // game memory
      log_lik[i] = 0;
      gamemem = 0;
      for (t in 1:(Tsub[i])) {
        if (game[i, t] != gamemem) {
          ev[i, t, 1] = initV[1];
          ev[i, t, 2] = initV[2];
          vari[i, t, 1] = initSD[i]^2;
          vari[i, t, 2] = initSD[i]^2;
          gamemem = game[i, t];
        }
        if (t >= 5) {
          log_lik[i] += normal_lpdf(choice[i, t] | beta[i] * (ev[i, t, 1] - ev[i, t, 2]) / sqrt(vari[i, t, 1] + vari[i, t, 2]) + gamma[i] * (sqrt(vari[i, t, 1]) - sqrt(vari[i, t, 2])), 1);
        }
        PE = reward[i, t] - ev[i, t, choiceidx[i, t]];
        K[i, t] = vari[i, t, choiceidx[i, t]] / (vari[i, t, choiceidx[i, t]] + oSD^2);
        ev[i, t+1, 1] = ev[i, t, 1];
        ev[i, t+1, 2] = ev[i, t, 2];
        vari[i, t+1, 1] = vari[i, t, 1];
        vari[i, t+1, 2] = vari[i, t, 2];
        ev[i, t+1, choiceidx[i, t]] = ev[i, t, choiceidx[i, t]] + K[i, t] * PE;
        vari[i, t+1, choiceidx[i, t]] = vari[i, t, choiceidx[i, t]] - K[i, t] * vari[i, t, choiceidx[i, t]];
      }
    }
  }
}
"""
