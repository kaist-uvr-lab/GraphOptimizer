#pragma once

#ifndef GRAPH_OPTIMIZER_OPTIMIZER_H
#define GRAPH_OPTIMIZER_OPTIMIZER_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace cv;

namespace GraphOptimizer {

	class Vertex;
	class Edge;

	class Optimizer {
	public:
		Optimizer();
		virtual ~Optimizer();
		//void AddVertex(Vertex* pVertex);
		void AddVertex(Vertex* pVertex, bool bType = true);
		void AddEdge(Edge* pEdge);
		void Accept();
		void Reject();
	public:
		virtual void Optimize(int trial, int level, bool bShowStatus = false)=0;
	protected:
		int CheckError(double currErr, double prevErr);
	public:
		int mnBlockHessianIdx1, mnBlockHessianIdx2; //for schur complement
		int mnBlockVertexIdx; //vector index for schur complement
		Eigen::MatrixXd Haa, Hbb, Hab, Hba;
		Eigen::VectorXd Ba, Bb;
		Eigen::MatrixXd HbbInv, HabHbbInv;
		bool bSchur;
	protected:
		std::vector<Vertex*> mvpVertices;
		std::vector<Edge*> mvpEdges;
		
	};

	class GNOptimizer : public Optimizer {
	public:
		GNOptimizer();
		virtual ~GNOptimizer();
		void Optimize(int trial, int level, bool bShowStatus);
	private:
		void SolveProblem();
		bool SolveProblem(Vertex* pFVertex);
	protected:
		
	};
	class LMOptimizer : public Optimizer {
	public:
		LMOptimizer();
		virtual ~LMOptimizer();
		void Optimize(int trial, int level, bool bShowStatus);
	private:
		void SolveProblem();
	protected:
		double damping;
	};
}

#endif //ANDROIDOPENCVPLUGINPROJECT_OPTIMIZER_H