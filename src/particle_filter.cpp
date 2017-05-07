/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

std::default_random_engine gen;

double bivariate_pdf(double mu_x, double mu_y, double x, double y, double std_x, double std_y) {
    const double std_x2 = 2 * pow(std_x, 2);
    const double std_y2 = 2 * pow(std_y, 2);
    const double diff_x2 = pow(x - mu_x, 2);
    const double diff_y2 = pow(y - mu_y, 2);
    const double C = 1. / (2. * M_PI * std_x * std_y);
    const double prob = C * exp(-(diff_x2 / std_x2 + diff_y2 / std_y2));
    return prob;
};

LandmarkObs transformObservation(Particle p, LandmarkObs obs) {
    LandmarkObs updatedObs = {obs.id, obs.x, obs.y};
    updatedObs.x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
    updatedObs.y = obs.x * sin(p.theta) + obs.y * cos(p.theta) + p.y;
    return updatedObs;
};

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = 10;

	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	std::normal_distribution<double> dist_x(x, std_x);
	std::normal_distribution<double> dist_y(y, std_y);
	std::normal_distribution<double> dist_theta(theta, std_theta);

	for (int i = 0; i < num_particles; i++) {
		Particle sample;
		sample.id = i;
		sample.x = dist_x(gen);
		sample.y = dist_y(gen);
		sample.theta = dist_theta(gen);
		sample.weight = 1.;

		weights.push_back(sample.weight);
		particles.push_back(sample);
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	std::normal_distribution<double> dist_x(0., std_x);
	std::normal_distribution<double> dist_y(0., std_y);
	std::normal_distribution<double> dist_theta(0., std_theta);

	for (int i = 0; i < num_particles; i++) {
		Particle& p = particles[i];
		if(fabs(yaw_rate) > 1e-5) {
            const double yaw_dt = yaw_rate * delta_t;
            const double v_over_yaw = velocity / yaw_rate;
			p.x += v_over_yaw * (sin(p.theta + yaw_dt) - sin(p.theta)) + dist_x(gen);
			p.y += v_over_yaw * (cos(p.theta) - cos(p.theta + yaw_dt)) + dist_y(gen);
            p.theta += yaw_dt + dist_theta(gen);
		} else {
            const double v_dt = velocity * delta_t;
			p.x += v_dt * cos(p.theta) + dist_x(gen);
            p.y += v_dt * sin(p.theta) + dist_y(gen);
            p.theta += dist_theta(gen);
		}
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    if(!predicted.empty()) {
        for(int i = 0; i < observations.size(); i++) {
            int id = 0;
            LandmarkObs obs = observations[i];
            double min_range = dist(obs.x, obs.y, predicted[0].x, predicted[0].y);
            for(int j = 0; j < predicted.size(); j++) {
                double range = dist(obs.x, obs.y, predicted[j].x, predicted[j].y);
                if(range < min_range) {
                    min_range = range;
                    id = j;
                }
            }
            observations[i].id = id;
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	for (int i = 0; i < num_particles; i++) {
		Particle& p = particles[i];
		std::vector<LandmarkObs> transformed_obs;
		for(int j = 0; j < observations.size(); j++) {
			transformed_obs.push_back(transformObservation(p, observations[j]));
		}

		std::vector<LandmarkObs> predicted;
		for(int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			Map::single_landmark_s landmark = map_landmarks.landmark_list[j];
			if(dist(p.x, p.y, landmark.x_f, landmark.y_f) <= sensor_range) {
				LandmarkObs obs = {landmark.id_i, landmark.x_f, landmark.y_f};
				predicted.push_back(obs);
			}
		}

		if(!predicted.empty()) {
			dataAssociation(predicted, transformed_obs);
			double weight = 1.;
			for(int j = 0; j < transformed_obs.size(); j++) {
				const LandmarkObs& obs = transformed_obs[j];
				const LandmarkObs& lm = predicted[obs.id];
                double prob = bivariate_pdf(lm.x, lm.y, obs.x, obs.y, std_landmark[0], std_landmark[1]);
				weight = weight * prob;
			}
			p.weight = weight;
		} else {
			p.weight = 0.;
		}
        weights[i] = p.weight;
	}
}

void ParticleFilter::resample() {
	std::discrete_distribution<> distr(weights.begin(), weights.end());
	std::vector<Particle> updated;
	for(int i = 0; i < num_particles; i++) {
		int id = distr(gen);
		updated.push_back(particles[id]);
	}
	particles = updated;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::trunc);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << " " << particles[i].weight << "\n";
	}
	dataFile.close();
}
