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

double bivariate_pdf(double mu_x, double mu_y, double x, double y, double std_x, double std_y) {
    const double std_x2 = pow(std_x, 2);
    const double std_y2 = pow(std_y, 2);
    const double diff_x2 = pow(x - mu_x, 2);
    const double diff_y2 = pow(y - mu_y, 2);
    return exp(-0.5f * (diff_x2 / std_x2 + diff_y2 / std_y2)) / (2.0f * M_PI * std_x * std_y);
};

LandmarkObs transformObservation(Particle p, LandmarkObs obs) {
    LandmarkObs updatedObs = {obs.id, obs.x, obs.y};
    updatedObs.x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
    updatedObs.y = obs.x * sin(p.theta) + obs.y * cos(p.theta) + p.y;
    return updatedObs;
};

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	std::default_random_engine gen;

	num_particles = 100;

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
		sample.weight = 1.0;

		weights.push_back(sample.weight);
		particles.push_back(sample);
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	std::default_random_engine gen;

	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	std::normal_distribution<double> dist_x(0, std_x);
	std::normal_distribution<double> dist_y(0, std_y);
	std::normal_distribution<double> dist_theta(0, std_theta);

	for (int i = 0; i < num_particles; i++) {
		Particle sample = particles[i];
		if(yaw_rate > 1e-3) {
			double theta0 = sample.theta;
			sample.theta += yaw_rate * delta_t + dist_theta(gen);
			sample.x += (sin(sample.theta) - sin(theta0)) * velocity / yaw_rate + dist_x(gen);
			sample.y += (cos(theta0) - cos(sample.theta)) * velocity / yaw_rate + dist_y(gen);
		} else {
			sample.theta += dist_theta(gen);
			sample.x += velocity * delta_t * sin(sample.theta) + dist_x(gen);
			sample.y -= velocity * delta_t * cos(sample.theta) + dist_y(gen);
		}
		particles[i] = sample;
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
		Particle p = particles[i];
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
			double weight = 1.0f;
			for(int j = 0; j < transformed_obs.size(); j++) {
				LandmarkObs obs = transformed_obs[j];
				weight = weight * bivariate_pdf(predicted[obs.id].x, predicted[obs.id].y, obs.x, obs.y, std_landmark[0], std_landmark[1]);
			}
			particles[i].weight = weight;
		} else {
			particles[i].weight = 0.0f;
		}
        weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	std::default_random_engine gen;
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
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
