

#include <Vertex.h>

GraphOptimizer::Vertex::Vertex():bFixed(false){}

GraphOptimizer::Vertex::Vertex(int idx, int _size, bool _b) :index(idx), bFixed(_b), dsize(_size) {
	H = Eigen::MatrixXd(dsize, dsize);
	b = Eigen::VectorXd(dsize);
	ResetHessian();
}
GraphOptimizer::Vertex::~Vertex() {}

void GraphOptimizer::Vertex::SetHessian(Eigen::MatrixXd I, Eigen::VectorXd r) {
	H += j.transpose()*I*j;
	b += j.transpose()*I*r;
}

void GraphOptimizer::Vertex::ResetHessian() {
	H.setZero();
	b.setZero();
}

void GraphOptimizer::Vertex::CheckHessian() {
	if (H.isZero()) {
		H.setIdentity();
		b.setZero();
	}
}

Eigen::MatrixXd GraphOptimizer::Vertex::GetHessian() {
	//if(bFixed)
	//    return cv::Mat::eye(dsize, dsize, CV_64FC1);

	return H;
}

Eigen::MatrixXd GraphOptimizer::Vertex::GetInverseHessian() {
	/*Eigen::MatrixXd res;
	if (dsize == 1) {

	std::cout << "test inv ::" << H << std::endl << H.inverse() << std::endl;
	}
	else {

	}*/

	return H.inverse();
}
Eigen::VectorXd GraphOptimizer::Vertex::GetB() {
	//if(bFixed)
	//    return cv::Mat::zeros(dsize, 1, CV_64FC1);
	return b;
}
void GraphOptimizer::Vertex::SetDifferential(Eigen::VectorXd _d) {
	d = _d;
}
Eigen::VectorXd GraphOptimizer::Vertex::GetDifferential() {
	return d;
}
void GraphOptimizer::Vertex::AddWeight(double w) {
	H += H.diagonal().asDiagonal()*w;
}
int GraphOptimizer::Vertex::GetSize() { return dsize; }

void GraphOptimizer::Vertex::UpdateParam() {
	param += d;
	ResetHessian();
}

void GraphOptimizer::Vertex::Accept() {
	Hold = H;
	Bold = b;
	paramOld = param;
}
void GraphOptimizer::Vertex::Reject() {
	b = Bold;
	H = Hold;
	param = paramOld;
}