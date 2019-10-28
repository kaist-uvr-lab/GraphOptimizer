#pragma once

#ifndef GRAPH_OPTIMIZER_ROBUSTKERNEL_H
#define GRAPH_OPTIMIZER_ROBUSTKERNEL_H

#include <Eigen/Core>

namespace GraphOptimizer {
	class RobustKernel {
	public:
		RobustKernel() {}
		virtual ~RobustKernel() {}
		virtual void robustify(double squaredError, Eigen::Vector3d& rho)=0;
		virtual void SetDelta(double delta) {
			_delta = delta;
			_deltaSquared = delta*delta;
		}
		double GetDelta() {
			return _delta;
		}
	protected:
		double _delta;
		double _deltaSquared;
	};

	class HuberKernel : public RobustKernel {
	public:
		void robustify(double squaredError, Eigen::Vector3d& rho) {
			if (squaredError <= _deltaSquared) { // inlier
				rho[0] = squaredError;
				rho[1] = 1.;
				rho[2] = 0.;
			}
			else { // outlier
				double sqrte = sqrt(squaredError); // absolut value of the error
				rho[0] = 2 * sqrte*_delta - _deltaSquared; // rho(e)   = 2 * delta * e^(1/2) - delta^2
				rho[1] = _delta / sqrte;        // rho'(e)  = delta / sqrt(e)
				rho[2] = -0.5 * rho[1] / squaredError;    // rho''(e) = -1 / (2*e^(3/2)) = -1/2 * (delta/e) / e
			}
		}
	};
}

#endif