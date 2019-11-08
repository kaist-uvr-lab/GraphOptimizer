//
// Created by UVR-KAIST on 2019-06-26.
//

#pragma once

#ifndef GRAPH_OPTIMIZER_VERTEX_H
#define GRAPH_OPTIMIZER_VERTEX_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace cv;

namespace GraphOptimizer {
	
	//여러 vertex가 동일한 파라메터를 접근하게 되는데 그 부분을 쉽게 하기 위하여 빼냄.
	//타입은 double임.
	
	class Vertex {
	public:
		Vertex();
		Vertex(int idx, int _size, bool _b);
		virtual ~Vertex();

		//MPVertex에서 이용
		int GetIndex() {
			return index;
		}
		void SetFixed(bool b) {
			bFixed = b;
		}
		bool GetFixed() {
			return bFixed;
		}
		Eigen::VectorXd GetParam() {
			return param;
		}
		Eigen::MatrixXd GetJacobian() {
			return j;
		}
		void SetJacobian(Eigen::MatrixXd J) {
			j = J;
		}
		void SetHessian(Eigen::MatrixXd I, Eigen::VectorXd r);
		void ResetHessian();
		void CheckHessian();
		Eigen::MatrixXd GetHessian();
		Eigen::MatrixXd GetInverseHessian();
		Eigen::VectorXd GetB();
		void SetDifferential(Eigen::VectorXd _d);
		Eigen::VectorXd GetDifferential();
		void AddWeight(double w);
		int GetSize();
	public:
		
		virtual void SetParam() = 0;
		virtual void UpdateParam();
		virtual void RestoreData() = 0;
		virtual void* GetPointer() = 0;
		virtual void Accept();
		virtual void Reject();
	public:
		int mnOptimizerVertexIndex;
		Eigen::MatrixXd H, Hold, H2;
		Eigen::VectorXd b, Bold, b2;
	protected:
		int index, dsize;
		bool bFixed;
		
		Eigen::MatrixXd j;
		Eigen::VectorXd d;
		Eigen::VectorXd param, paramOld;
	};
		
}

#endif //ANDROIDOPENCVPLUGINPROJECT_VERTEX_H
