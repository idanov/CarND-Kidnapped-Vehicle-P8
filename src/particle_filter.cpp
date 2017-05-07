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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	std::default_random_engine gen;

	num_particles = 100;

	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	std::normal_distribution<double> dist_x(x, std_x);
	std::normal_distribution<double> dist_y(x, std_y);
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
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
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
