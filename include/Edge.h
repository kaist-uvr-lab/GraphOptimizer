#pragma once

#ifndef GRAPH_OPTIMIZER_EDGE_H
#define GRAPH_OPTIMIZER_EDGE_H


#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <RobustKernel.h>
//#include "../include/Vertex.h"

using namespace cv;

namespace GraphOptimizer {
	
	typedef std::pair<int, int> HessianSubBlockIdx;
	class Vertex;
	class Edge {
	public:
		Edge();
		Edge(int rsize);
		virtual ~Edge() {}

		void SetLevel(int l) {
			prevLevel = level;
			level = l;
		}
		int GetLevel() {
			return level;
		}
		void ReterunLevel() {
			level = prevLevel;
		}
		void SetMeasurement(Eigen::VectorXd _m) {
			measurment = _m;
		}
		Eigen::VectorXd GetMeasurement() {
			return measurment;
		}
		void SetInformation(cv::Mat _i);
		Eigen::MatrixXd GetInforamtion() {
			return info;
		}
		double GetError() {
			return err;
		}
		void Accept() {
			if(mvSubBlockIdxs.size()>0)
				std::copy(mvSubHs.begin(), mvSubHs.end(), mvSubHolds.begin());
		}
		void Reject() {
			if (mvSubBlockIdxs.size()>0)
				std::copy(mvSubHolds.begin(), mvSubHolds.end(), mvSubHs.begin());
		}
		//vector subh의 크기를 설정함.
		void SetSubHessian();
		void AddVertex(Vertex* pVertex) {
			mvpVertices.push_back(pVertex);
		}
		Vertex* GetVertex(int idx) {
			return mvpVertices[idx];
		}
	public:
		virtual double CalcError() = 0;
		virtual void CalcJacobian() = 0;
		virtual void SetHessian() = 0; //버텍스가 2개 다 설정되면 서브 매트릭스 설정이 필요함
		
		//virtual void SetSubspaceHessian(cv::Mat& HgmHmmInv, cv::Mat& Hmg) = 0;
		//virtual void EraseEdge() = 0;
		//virtual void SetEdge() = 0;
	//private:
		//int combination(int n, int r) { if (n == r || r == 0) return 1; else return combination(n - 1, r - 1) + combination(n - 1, r); }
	public:
		//edge에 따라 다름. R, S, fx, fy, x,y,z 등
		//cv::Mat subH, subHOld;
		std::vector<Eigen::MatrixXd> mvSubHs, mvSubHolds;
		std::vector<HessianSubBlockIdx> mvSubBlockIdxs;
		RobustKernel* mpRobustKernel;
	protected:
		//Vertex* mpVertex1;
		//Vertex* mpVertex2;
		std::vector<Vertex*> mvpVertices;
		double err;
		Eigen::MatrixXd info;
		Eigen::VectorXd residual;
		Eigen::VectorXd measurment;
		double w;

		int level, prevLevel;
		/////////////이 위에는 무조건 있어야 하는 것

	};

}

#endif