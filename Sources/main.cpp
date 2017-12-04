#include <iostream>
#include <vector>
#include <tuple>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>  
#include <chrono>
#include <ctime>
#include <filesystem>
#include <string>
#include <regex>
#include <cstdint>
#include "Utilities.h"
#include "AlphaPathMoves.h"
#include "TS_Logger.h"

using namespace Utilities;
namespace fs = std::experimental::filesystem::v1;
std::mutex unittests_queue_mtx;
std::queue<std::string> unittests_queue;



double getWpq_Euclidean(int64_t p, int64_t q, const void * data) 
{  
	return 1; 
};
double getWpq_Canny( int64_t p, int64_t q, const void * data) 
{ 
	const uint32_t * cannydata = reinterpret_cast<const uint32_t *>(data);
	return (cannydata != nullptr && ((cannydata[p] == 1 && cannydata[q] == 1) || (cannydata[p] == 0 && cannydata[q] == 0))) ? 1 : 0.125; 
};
double getWpq_Intensity( int64_t p, int64_t q, const void * data)
{ 
	const double * voldata = reinterpret_cast<const double *>(data);
	return exp(-pow2(voldata[p] - voldata[q]) / 10000); 
};

int run_unittestcase(std::string src_path)
{
	Array3D<double> vol(src_path + "vol.mat");
	Array3D<int32_t> hhmask(src_path + "hhogsmask.mat");
	Array3D<uint32_t> canny(src_path + "canny.mat");
	Array2D<int64_t> dataterms(src_path + "dataterms.mat");
	Array3D<uint32_t> initlabeling(src_path + "initlabeling.mat");
	Array2D<double> hhog_theta(src_path + "hhog_theta.mat");
	Array2D<double> smooth_lambda(src_path + "smooth_lambda.mat");
	Array2D<double> hierarchical_tree_weights(src_path + "hierarchical_tree.mat");
	Array2D<uint32_t> hhog_wsize(src_path + "hhog_wsize.mat");
	Array2D<uint32_t> smooth_wsize(src_path + "smooth_wsize.mat");
	Array2D<uint32_t> min_margins(src_path + "min_margins.mat");
	Array2D<uint32_t> exp_ordering(src_path + "expansion_ordering.mat");
	Array2D<int64_t> gt_sol_energy(src_path + "energy.mat");

	uint32_t n_labels = hierarchical_tree_weights.X; //number of labels (also hierarchical.X==hierarchical.Y=number of labels)
	int64_t dims[3];
	dims[0] = vol.X;
	dims[1] = vol.Y;
	dims[2] = vol.Z;

	
	
	using gctype = int64_t;
	using PathMoves = GC::AlphaPathMoves< gctype, gctype, gctype>;
	using MaxSolver = MaxflowSolver<gctype, gctype, gctype>;
	PathMoves * optimizer = nullptr;
	Array3D<uint32_t> * sol_labeling = nullptr;
	gctype sol_energy;
	
	try
	{

		optimizer = new PathMoves(dims, n_labels); //it only accepts 3D dims, for 2D examples set dim[2] to 1

		//[required] set the unary potentials (dataterms)
		//dataterms is an array of size n_lables x n_voxels
		optimizer->setDataTerms(&dataterms);

		////// set smoothness paramters //////////
		//[required] set smoothness neighbourhood window size  (which should be an odd number)   
		optimizer->setSmoothnessNeighbourhoodWindowSize(smooth_wsize.data[0]);        
		
		//[required] set smoothness parameter lambda and the hierarchal tree-weights.
		//Tree weights must be an adjacency matrix of size n_labels x n_labels.
		//A Row represent a parent node, and the columns represent the childeren of that node.
		//parent -> child weight could be >=0, while -1 means they are no connected.
		optimizer->setTreeWeights(smooth_lambda.data[0], &hierarchical_tree_weights); 

		//[optional] specify the function to be called to compute the discontinuity weights between pixels p and q, i.e. w_pq.
		//The function signature is double getWpq(int64_t p, int64_t q, const void * data);
		//Also, specity a pointer to the data that will be sent to the function along with indxies of pixels p and q.
		optimizer->setWpqFunction(getWpq_Canny, canny.data);
		
		//[required] set initial solution
		//The initial solution must be valid, i.e. hierarchal structure and hhog prior must not be violated. 
		//It as always safe to start from the following trivial solution: 
		//everything is background except for the set pixels in the hhog-mask constraints they are assinged to their corresponding labels.
		optimizer->setInitialLabeling(&initlabeling);


		//[optional]set min-margins and (optinal)the method used to compute them
		//min_margins is an array of size n_lables x 1, labels with no min_marigns should have 0 min_margins.
		//Also, this function can not be called before setting the hierarchal tree via setTreeWeights.
		optimizer->setMinimumMargins(&min_margins, PathMoves::MinMarginMethod::CUDA_BASED);

		//[optional] set hhog parameters
		//hhog neghbourhood window size (odd number), hhog mask (used to generate hhog vector fields), shape tightness parameter theta [0-90].
		//hhog mask[p]=i means that pixel p is part of label_i seed. hhog mask[p]=-1 means that it is not part of any label's seed.
		//theta is in degrees and it always best to stay out of extreem values, like 0 and 90. For best perfomance theta should be 45+/-20.
		optimizer->setHedgehogAttributes(hhog_wsize.data[0], &hhmask, hhog_theta.data[0]);

		//[optional] set the order in which lables are expanded during PathMoves.
		//This is required in order to be able to replicate results, as in the case of Unittesting.
		optimizer->setExpansionOrdering(&exp_ordering);

		//[required] run pathmoves and specify the desired max-flow solver. 
		//The currently supported solvers are IBFS, BK and QPBO. IBFS proved to be the fastest but uses more memeory than BK.
		//Also, DO NOT use IBFS for floating point precision flow/capcity types, it could be unstable and might not terminate.
		sol_labeling = optimizer->runPathMoves(MaxSolver::SolverName::IBFS, sol_energy);

	}catch (const std::exception& e) {
		bgn_log << LogType::ERROR << "The following Unittest threw an exception : \n" << src_path <<"\n"<<e.what()<<"\n"<<end_log;
		if (sol_labeling)
			delete sol_labeling;
		if (optimizer)
			delete optimizer;
		return false;
	}

	//Unittest, validation
	if (gt_sol_energy.data[0] != sol_energy)
		return false;

	if (sol_labeling)
		delete sol_labeling;

	if (optimizer)
		delete optimizer;

	return true;
}

void thread_worker()
{
	while (true)
	{
		unittests_queue_mtx.lock();
		if (unittests_queue.empty())
		{
			unittests_queue_mtx.unlock();
			return;
		}
		std::string instance_path = unittests_queue.front();
		unittests_queue.pop();
		unittests_queue_mtx.unlock();
		
		bgn_log << LogType::INFO_LEVEL0 << "Starting instance " + instance_path + "\n" << end_log;
		bool result=run_unittestcase(instance_path);
		if (result)
			bgn_log << LogType::INFO_LEVEL0 << "UT " + instance_path + " Succeeded\n" << end_log;
		else
			bgn_log << LogType::ERROR << "UT " + instance_path + " Failed\n" << end_log;
	} 
}
int main(int argc, char *argv[])
{
	
	if (argc<2)
	{
		bgn_log << LogType::ERROR << " Usage .... \n" << end_log;
		bgn_log << LogType::ERROR << " AlphaPathMoves unittests_path [number of threads] \n" << end_log;
		throw("Parameters were not set correctly");
	}
	std::string unittest_path(argv[1]);


	int n_threads = 3;
	if (argc == 3)
		n_threads = atoi(argv[2]);

	for (fs::path p : fs::directory_iterator(unittest_path))
		unittests_queue.push(p.u8string() + "\\");

	std::vector <std::thread> working_threads;
	for (int i = 0; i < n_threads; i++)
		working_threads.emplace_back(thread_worker);

	for (int i = 0; i < n_threads; i++)
		working_threads[i].join();
	return 0;
}
