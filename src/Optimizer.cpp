#include <Optimizer.h>
#include <Edge.h>
#include <Vertex.h>

GraphOptimizer::Optimizer::Optimizer():bSchur(false){}
GraphOptimizer::Optimizer::~Optimizer(){}

GraphOptimizer::GNOptimizer::GNOptimizer():Optimizer(){}
GraphOptimizer::GNOptimizer::~GNOptimizer() {}

GraphOptimizer::LMOptimizer::LMOptimizer():damping(1.0), Optimizer() {}
GraphOptimizer::LMOptimizer::~LMOptimizer() {}

//void GraphOptimizer::Optimizer::AddVertex(Vertex* pVertex) {
//	//mvpVertices.push_back(pVertex);
//	AddVertex(pVertex, false);
//}
void GraphOptimizer::Optimizer::AddVertex(Vertex* pVertex, bool bType) {
	//type1 mnIdx1 - pose plane
	//type2 mnIdx2 - map
	if (pVertex->GetFixed())
		return;

	if (bType) {
		pVertex->mnOptimizerVertexIndex = mnBlockHessianIdx1;
		mnBlockHessianIdx1 += pVertex->GetSize();
	}
	else {
		if (mnBlockHessianIdx2 == 0)
			mnBlockVertexIdx = mvpVertices.size();
		pVertex->mnOptimizerVertexIndex = mnBlockHessianIdx2;
		mnBlockHessianIdx2 += pVertex->GetSize();
	}
	mvpVertices.push_back(pVertex);
}
void GraphOptimizer::Optimizer::AddEdge(Edge* pEdge) {
	if (bSchur)
		pEdge->SetSubHessian();
	mvpEdges.push_back(pEdge);
}

int GraphOptimizer::Optimizer::CheckError(double currErr, double prevErr) {
	double diffErr = prevErr - currErr;
	int res = 2;
	if (diffErr < 0.0f) {
		res = 1;
	}
	else if ((diffErr) <= 0.0005f) {
		res = 0;
	}
	return res;
}

void GraphOptimizer::Optimizer::Accept() {
	for (int i = 0; i < mvpVertices.size(); i++) {
		mvpVertices[i]->Accept();
	}
	for (int i = 0; i < mvpEdges.size(); i++) {
		mvpEdges[i]->Accept();
	}
}
void GraphOptimizer::Optimizer::Reject() {
	for (int i = 0; i < mvpVertices.size(); i++) {
		mvpVertices[i]->Reject();
	}
	for (int i = 0; i < mvpEdges.size(); i++) {
		mvpEdges[i]->Reject();
	}
}

void GraphOptimizer::GNOptimizer::Optimize(int nTrial, int level, bool bShowStatus) {
	double sumPrevErr = DBL_MAX;
	//optimize method
	for (int trial = 0; trial < nTrial; trial++) {
		//set hessian matrix
		if(bShowStatus)
			std::cout << "trial::" << trial << "::start" << std::endl;
		double sumErr = 0.0;

		for (int i = 0; i < mvpEdges.size(); i++) {
			if (mvpEdges[i]->GetLevel()>level)
				continue;
			sumErr += mvpEdges[i]->CalcError();
			mvpEdges[i]->CalcJacobian();
			mvpEdges[i]->SetHessian();
			
		}
		//check err
		int nCheck = CheckError(sumErr, sumPrevErr);
		if (nCheck == 0) {
			if (bShowStatus)
				std::cout << "PoseOptimization::break::1::" << trial << "::" << sumPrevErr << "::" << sumErr << std::endl;
			Accept();
			break;
		}
		else if (nCheck == 1) {
			if (bShowStatus)
				std::cout << "PoseOptimization::break::2::" << trial << "::" << sumPrevErr << "::" << sumErr << std::endl;
			//mvpVertices[0]->Reject();
			Reject();
			break;

		}
		else {
			if (bShowStatus)
				std::cout << "PoseOptimization::progressing::" << trial << "::" << sumPrevErr << "::" << sumErr << std::endl;
			sumPrevErr = sumErr;
			Accept();
		}

		if (trial == nTrial - 1)
			return;
		SolveProblem();
		//bool bres = SolveProblem(mvpVertices[0]);
		if (bShowStatus)
			std::cout << "PoseOptimization::" << trial << "::solve::"  << std::endl;
		
	}
}

void GraphOptimizer::LMOptimizer::Optimize(int nTrial, int level, bool bShowStatus) {
	double sumPrevErr = DBL_MAX;
	//optimize method
	for (int trial = 0; trial < nTrial; trial++) {
		//set hessian matrix
		if (bShowStatus)
			std::cout << "LM::trial::" << trial << "::start" << std::endl;
		double sumErr = 0.0;

		for (int i = 0; i < mvpEdges.size(); i++) {
			if (mvpEdges[i]->GetLevel()>level)
				continue;
			double err= mvpEdges[i]->CalcError();
			//std::cout << "edge::" << i << "::" << err << std::endl;
			sumErr += err;
			mvpEdges[i]->CalcJacobian();
			//std::cout << "jacobian" << std::endl;
			mvpEdges[i]->SetHessian();
			//std::cout << "hesisan" << std::endl;

		}
		if (bShowStatus)
			std::cout << "LM::trial::" << trial << "::Edge::end" << std::endl;

		//check err
		int nCheck = CheckError(sumErr, sumPrevErr);
		if (nCheck == 0) {
			if (bShowStatus)
				std::cout << "LM::PoseOptimization::break::1::" << trial << "::" << sumPrevErr << "::" << sumErr << std::endl;
			Accept();
			break;
		}
		else if (nCheck == 1) {
			if (bShowStatus)
				std::cout << "LM::PoseOptimization::break::2::" << trial << "::"<<damping<<", " << sumPrevErr << "::" << sumErr << std::endl;
			//mvpVertices[0]->Reject();
			Reject();
			damping *= 10.0;
			//break;
		}
		else {
			if (bShowStatus)
				std::cout << "LM::PoseOptimization::progressing::" << trial << "::" << damping << ", " << sumPrevErr << "::" << sumErr << std::endl;
			sumPrevErr = sumErr;
			Accept();
			damping *= 0.5;
		}

		if (trial == nTrial - 1)
			return;
		SolveProblem();
		//bool bres = SolveProblem(mvpVertices[0]);
		if (bShowStatus)
			std::cout << "LM::PoseOptimization::" << trial << "::solve::" << std::endl;
		
	}
}

bool GraphOptimizer::GNOptimizer::SolveProblem(Vertex* pFVertex) {
	Eigen::VectorXd DeltaState = pFVertex->GetHessian().bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(-1 * pFVertex->GetB());
	
	//std::cout << "H=" << pFVertex->GetHessian() << std::endl;
	//std::cout << "b=" << pFVertex->GetB() << std::endl;
	////std::cout << "H2=" << pFVertex->H2 << std::endl;
	////std::cout << "b2=" << pFVertex->b2 << std::endl;
	//std::cout << "delta=" << DeltaState << std::endl;

	//Eigen::VectorXd DeltaState2 = pFVertex->H2.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(-1 * pFVertex->b2);
	//std::cout << "delta2=" << DeltaState2 << std::endl;
	pFVertex->SetDifferential(DeltaState);
	pFVertex->UpdateParam();
	return true;
}

void GraphOptimizer::GNOptimizer::SolveProblem() {
	if (mnBlockHessianIdx2 > 0) {
		//std::cout << "Index ::" << mnBlockHessianIdx1 << ", " << mnBlockHessianIdx2 << "::" << mnBlockVertexIdx << std::endl;
		Haa = Eigen::MatrixXd::Zero(mnBlockHessianIdx1, mnBlockHessianIdx1);
		Hab = Eigen::MatrixXd::Zero(mnBlockHessianIdx1, mnBlockHessianIdx2);
		Hbb = Eigen::MatrixXd::Zero(mnBlockHessianIdx2, mnBlockHessianIdx2);
		HbbInv = Eigen::MatrixXd::Zero(mnBlockHessianIdx2, mnBlockHessianIdx2);
		HabHbbInv = Eigen::MatrixXd::Zero(mnBlockHessianIdx1, mnBlockHessianIdx2);
		Ba = Eigen::VectorXd::Zero(mnBlockHessianIdx1);
		Bb = Eigen::VectorXd::Zero(mnBlockHessianIdx2);
		//Hab && Hbb && HbbInv



		//Haa, Ba
		for (int i = 0; i < mnBlockVertexIdx; i++) {
			Vertex* v = mvpVertices[i];
			if (v->GetFixed())
				continue;
			v->CheckHessian();
			Haa.block(v->mnOptimizerVertexIndex, v->mnOptimizerVertexIndex, v->GetSize(), v->GetSize()) = v->GetHessian();
			Ba.block(v->mnOptimizerVertexIndex, 0, v->GetSize(), 1) = v->GetB();
		}
		for (int i = mnBlockVertexIdx; i < mvpVertices.size(); i++) {
			Vertex* v = mvpVertices[i];
			if (v->GetFixed())
				continue;
			v->CheckHessian();
			Hbb.block(v->mnOptimizerVertexIndex, v->mnOptimizerVertexIndex, v->GetSize(), v->GetSize()) = v->GetHessian();
			HbbInv.block(v->mnOptimizerVertexIndex, v->mnOptimizerVertexIndex, v->GetSize(), v->GetSize()) = v->GetInverseHessian();
			Bb.block(v->mnOptimizerVertexIndex, 0, v->GetSize(), 1) = v->GetB();
		}
		
		for (int i = 0; i < mvpEdges.size(); i++)
		{
			Edge* e = mvpEdges[i];
			for (int j = 0; j < e->mvSubBlockIdxs.size(); j++) {
				
				HessianSubBlockIdx idx = e->mvSubBlockIdxs[j];
				Vertex* v1 = e->GetVertex(idx.first);
				Vertex* v2 = e->GetVertex(idx.second);

				if (v1->GetFixed() || v2->GetFixed())
					continue;

				Eigen::MatrixXd temp = e->mvSubHs[j];
				Eigen::MatrixXd subHinv = HbbInv.block(v2->mnOptimizerVertexIndex, v2->mnOptimizerVertexIndex, v2->GetSize(), v2->GetSize());
				
				HabHbbInv.block(v1->mnOptimizerVertexIndex, v2->mnOptimizerVertexIndex, v1->GetSize(), v2->GetSize()) = temp*subHinv;
				Hab.block(v1->mnOptimizerVertexIndex, v2->mnOptimizerVertexIndex, v1->GetSize(), v2->GetSize()) = temp;

			}
		}
		Hba = Hab.transpose();
		Eigen::MatrixXd Haa2 = HabHbbInv*Hba;
		//schur complement

		Eigen::MatrixXd HaaSchur = Haa - Haa2;
		Eigen::VectorXd BaSchur = -Ba + HabHbbInv*Bb;
		//std::cout << "HHHHH" << std::endl;
		//std::cout << HaaSchur << std::endl << BaSchur << std::endl;
		Eigen::VectorXd Da = HaaSchur.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(BaSchur);
		//std::cout <<"Da="<< Da << std::endl;
		Eigen::VectorXd BbSchur = -Bb - Hba*Da;

		for (int i = 0; i < mnBlockVertexIdx; i++) {
			Vertex* v = mvpVertices[i];
			Eigen::VectorXd tDa = Da.block(v->mnOptimizerVertexIndex, v->mnOptimizerVertexIndex, v->GetSize(), v->GetSize());
			v->SetDifferential(tDa);
			v->UpdateParam();
		}

		//update param
		for (int i = mnBlockVertexIdx; i < mvpVertices.size(); i++) {
			Eigen::VectorXd Bb2 = BbSchur.block(mvpVertices[i]->mnOptimizerVertexIndex, 0, mvpVertices[i]->GetSize(), 1);
			Eigen::VectorXd Db = mvpVertices[i]->GetHessian().bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Bb2);
			//std::cout << "Db::" << i << "=" << Db << std::endl;
			Vertex* v = mvpVertices[i];
			v->SetDifferential(Db);
			v->UpdateParam();
		}
	}
	else if (mnBlockHessianIdx2 == 0 && mvpVertices.size() == 1){
		SolveProblem(mvpVertices[0]);
	}
	else {

	}
}

void GraphOptimizer::LMOptimizer::SolveProblem() {
	if (mnBlockHessianIdx2 > 0) {
		std::cout << "Index ::" << mnBlockHessianIdx1 << ", " << mnBlockHessianIdx2 << "::" << mnBlockVertexIdx << std::endl;
		Haa = Eigen::MatrixXd::Zero(mnBlockHessianIdx1, mnBlockHessianIdx1);
		Hab = Eigen::MatrixXd::Zero(mnBlockHessianIdx1, mnBlockHessianIdx2);
		Hbb = Eigen::MatrixXd::Zero(mnBlockHessianIdx2, mnBlockHessianIdx2);
		HbbInv = Eigen::MatrixXd::Zero(mnBlockHessianIdx2, mnBlockHessianIdx2);
		HabHbbInv = Eigen::MatrixXd::Zero(mnBlockHessianIdx1, mnBlockHessianIdx2);
		Ba = Eigen::VectorXd::Zero(mnBlockHessianIdx1);
		Bb = Eigen::VectorXd::Zero(mnBlockHessianIdx2);
		//Hab && Hbb && HbbInv

		//Haa, Ba
		for (int i = 0; i < mnBlockVertexIdx; i++) {
			Vertex* v = mvpVertices[i];
			v->AddWeight(damping);
			v->CheckHessian();
			Haa.block(v->mnOptimizerVertexIndex, v->mnOptimizerVertexIndex, v->GetSize(), v->GetSize()) = v->GetHessian();
			Ba.block(v->mnOptimizerVertexIndex, 0, v->GetSize(), 1) = v->GetB();
		}
		
		for (int i = mnBlockVertexIdx; i < mvpVertices.size(); i++) {
			Vertex* v = mvpVertices[i];
			v->AddWeight(damping);
			v->CheckHessian();
			Hbb.block(v->mnOptimizerVertexIndex, v->mnOptimizerVertexIndex, v->GetSize(), v->GetSize()) = v->GetHessian();
			HbbInv.block(v->mnOptimizerVertexIndex, v->mnOptimizerVertexIndex, v->GetSize(), v->GetSize()) = v->GetInverseHessian();
			Bb.block(v->mnOptimizerVertexIndex, 0, v->GetSize(), 1) = v->GetB();
		}
		
		//std::cout << "Haa=[" << Haa << std::endl << "]" << std::endl;
		//std::cout << "Bpose=[" << Ba << std::endl << "]" << std::endl;
		for (int i = 0; i < mvpEdges.size(); i++)
		{
			Edge* e = mvpEdges[i];
			//std::cout << "Edge::" << i <<"== "<< e->mvSubBlockIdxs.size()<< std::endl;
			for (int j = 0; j < e->mvSubBlockIdxs.size(); j++) {

				HessianSubBlockIdx idx = e->mvSubBlockIdxs[j];
				Vertex* v1 = e->GetVertex(idx.first);
				Vertex* v2 = e->GetVertex(idx.second);
				if (v1->GetFixed() || v2->GetFixed())
					continue;
				Eigen::MatrixXd temp = e->mvSubHs[j];
				Eigen::MatrixXd subHinv = HbbInv.block(v2->mnOptimizerVertexIndex, v2->mnOptimizerVertexIndex, v2->GetSize(), v2->GetSize());

				HabHbbInv.block(v1->mnOptimizerVertexIndex, v2->mnOptimizerVertexIndex, v1->GetSize(), v2->GetSize()) = temp*subHinv;
				Hab.block(v1->mnOptimizerVertexIndex, v2->mnOptimizerVertexIndex, v1->GetSize(), v2->GetSize()) = temp;

			}
		}
		
		Hba = Hab.transpose();
		Eigen::MatrixXd Haa2 = HabHbbInv*Hba;
		//schur complement
		
		Eigen::MatrixXd HaaSchur = Haa - Haa2;
		Eigen::VectorXd BaSchur = -Ba + HabHbbInv*Bb;
		//std::cout << "HHHHH" << std::endl;
		
		//std::cout << HaaSchur << std::endl << BaSchur << std::endl;
		Eigen::VectorXd Da = HaaSchur.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(BaSchur);
		//std::cout <<"Da="<< Da << std::endl;
		Eigen::VectorXd BbSchur = -Bb - Hba*Da;
		
		for (int i = 0; i < mnBlockVertexIdx; i++) {
			Vertex* v = mvpVertices[i];
			Eigen::VectorXd tDa = Da.block(v->mnOptimizerVertexIndex, 0, v->GetSize(),1);
			//std::cout << "Da::" << Da << std::endl;
			//std::cout << tDa << std::endl;
			v->SetDifferential(tDa);
			v->UpdateParam();
		}
		//update param
		for (int i = mnBlockVertexIdx; i < mvpVertices.size(); i++) {
			Eigen::VectorXd Bb2 = BbSchur.block(mvpVertices[i]->mnOptimizerVertexIndex, 0, mvpVertices[i]->GetSize(), 1);
			Eigen::VectorXd Db = mvpVertices[i]->GetHessian().bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(Bb2);
			//std::cout << "Db::" << i << "=" << Db << std::endl;
			Vertex* v = mvpVertices[i];
			v->SetDifferential(Db);
			v->UpdateParam();
		}
		
	}
	else if (mnBlockHessianIdx2 == 0 && mvpVertices.size() == 1) {
		mvpVertices[0]->AddWeight(damping);
		Eigen::VectorXd DeltaState = mvpVertices[0]->GetHessian().bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(-1 * mvpVertices[0]->GetB());
		mvpVertices[0]->SetDifferential(DeltaState);
		mvpVertices[0]->UpdateParam();
	}
	else {

	}
}