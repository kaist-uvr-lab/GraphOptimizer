#include <Edge.h>
#include <opencv2\core\eigen.hpp>

GraphOptimizer::Edge::Edge() :level(0), prevLevel(0), err(0), w(1.0) {
	info.setIdentity(1, 1);
	mpRobustKernel = nullptr;
}
GraphOptimizer::Edge::Edge(int rsize) :level(0), prevLevel(0), err(0), w(1.0) {
	info.setIdentity(rsize, rsize);
	mpRobustKernel = nullptr;
}

void GraphOptimizer::Edge::SetInformation(cv::Mat _i) {
	cv::cv2eigen(_i, info);
}

void GraphOptimizer::Edge::SetSubHessian() {
	int Ncom = 0;
	if (mvpVertices.size()>1) {
		for (int i = 0; i < mvpVertices.size() - 1; i++) {
			for (int j = i + 1; j < mvpVertices.size(); j++) {
				mvSubBlockIdxs.push_back(std::make_pair(i, j));
			}
		}
		Ncom = mvSubBlockIdxs.size();
	}
	mvSubHolds = std::vector<Eigen::MatrixXd>(Ncom);
	mvSubHs = std::vector<Eigen::MatrixXd>(Ncom);
}