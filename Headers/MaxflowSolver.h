#ifndef __MAXFLOWSOLVER_H__
#define __MAXFLOWSOLVER_H__
#include "BK\graph.h"
#include "QPBO\QPBO.h"
#include "IBFS\ibfs.h"
#include "BasicGraph.h"
#include <time.h>
#include <cinttypes>
#include "TS_Logger.h"

//#define MFS_DEBUG
template class IBFSGraph<long long, long long, long long>;
template <class nweightType, class tweightType, class flowtype> class MaxflowSolver
{
public :
	enum class SolverName { BK, IBFS, QPBO_SLVR };
private:
	SolverName name; //1 BK graph cut, 2 IBFS, 3 QPBO
	size_t n_edges, n_nodes;
	double maxflow_runtime, construction_runtime, total_runtime;
	// BK library
	Graph<nweightType, tweightType, flowtype> * bk_gc;
	flowtype BK_solver(BasicGraph<nweightType, tweightType> *, char *);

	// IBFS library
	IBFSGraph<nweightType, tweightType, flowtype> * ibfs_gc;
	flowtype IBFS_solver(BasicGraph<nweightType, tweightType> *, char *);

	// QPBO library
	QPBO<flowtype> * qpbo;
	flowtype QPBO_solver(BasicGraph<nweightType, tweightType> *, char *);
	int run_id;
public:
	MaxflowSolver(size_t, size_t, SolverName); //those are only estiamte the actual # of nodes/edges in the basicgraph could be bigger.
	static SolverName getDefaultSolver();
	void getRuntimes(double &, double &, double&);
	~MaxflowSolver();
	flowtype solve(BasicGraph<nweightType, tweightType> *, char *);
	flowtype computEnergy(BasicGraph<nweightType, tweightType> * graph, char * labeling);
};

template <class nweightType, class tweightType, class flowtype>
flowtype  MaxflowSolver<nweightType, tweightType, flowtype>::computEnergy(BasicGraph<nweightType, tweightType> * graph,char * labeling)
{
	BasicGraph<nweightType, tweightType>::t_links_struct tlinks = graph->get_tlinks();
	BasicGraph<nweightType, tweightType>::n_links_struct nlinks = graph->get_nlinks();
	size_t n_nodes = graph->getNumberOfNodes();

	int s = 0, t=1;
	flowtype unary_energy = 0, pairwise_energy = 0;
	for (size_t link_id = 0; link_id < tlinks.n_tlinks; ++link_id)
		if (labeling[tlinks.node_ids[link_id]] == t)
			unary_energy += tlinks.weights[link_id * 2 + 0];
		else
			unary_energy += tlinks.weights[link_id * 2 + 1];
	for (size_t offset = 0; offset < (nlinks.n_nlinks * 2); offset += 2)
	{
		if (labeling[nlinks.edge_nodes[offset + 0]] == s && labeling[nlinks.edge_nodes[offset + 1]] == t)
			pairwise_energy += nlinks.weights[offset + 0];
		else if (labeling[nlinks.edge_nodes[offset + 0]] == t && labeling[nlinks.edge_nodes[offset + 1]] == s)
			pairwise_energy += nlinks.weights[offset + 1];
	}
	//std::cout << "Computed flow: " << unary_energy + pairwise_energy << " =" << unary_energy << " + " << pairwise_energy << std::endl;
	return unary_energy + pairwise_energy;
}

template <class nweightType, class tweightType, class flowtype>
void MaxflowSolver<nweightType, tweightType, flowtype>::getRuntimes(double & construction_rt, double & maxflow_rt, double& total_rt)
{
	construction_rt = this->construction_runtime;
	maxflow_rt = this->maxflow_runtime;
	total_rt = this->total_runtime;
}
template <class nweightType, class tweightType, class flowtype>
typename MaxflowSolver<nweightType, tweightType, flowtype>::SolverName MaxflowSolver<nweightType, tweightType, flowtype>::getDefaultSolver()
{
	return IBFS;// QPBO_SLVR; //BK; IBFS;
}

template <class nweightType, class tweightType, class flowtype>
MaxflowSolver<nweightType, tweightType, flowtype>::MaxflowSolver(size_t in_n_nodes, size_t in_n_edges, typename MaxflowSolver<nweightType, tweightType, flowtype>::SolverName name)
{
	this->run_id = 1;
	this->construction_runtime = 0;
	this->maxflow_runtime = 0;
	this->total_runtime= 0;
	this->n_edges = in_n_edges;
	this->n_nodes = in_n_nodes;
	this->name = name;
	clock_t begin_time;
	switch (name)
	{
	case SolverName::BK:
		
		begin_time = clock();
		//ThreadSafeCout::Print("\tInitilizaing BK Solver...\n");
		bk_gc = new Graph<nweightType, tweightType, flowtype>(n_nodes, n_edges);
		//ThreadSafeCout::Print("\tInitilizaing BK Solver done in " + std::to_string(float(clock() - begin_time) / CLOCKS_PER_SEC)  + "\n");
		break;
	case SolverName::IBFS:
		begin_time = clock();
		//ThreadSafeCout::Print("\tInitilizaing IBFS Solver...\n");
		ibfs_gc = new IBFSGraph<nweightType, tweightType, flowtype>(IBFSGraph<nweightType, tweightType, flowtype>::IB_INIT_FAST);
		ibfs_gc->initSize(n_nodes, n_edges);
		//ThreadSafeCout::Print("\tInitilizaing IBFS Solver done in " + std::to_string(float(clock() - begin_time) / CLOCKS_PER_SEC) + "\n");
		break;
	case SolverName::QPBO_SLVR:
		begin_time = clock();
		//ThreadSafeCout::Print("\tInitilizaing QPBO Solver...\n");
		qpbo = new QPBO<flowtype>(n_nodes, n_edges);
		//ThreadSafeCout::Print("\tInitilizaing QPBO Solver done in " + std::to_string(float(clock() - begin_time) / CLOCKS_PER_SEC) + "\n");
		break;
	default:
		break;
	}
};

template <class nweightType, class tweightType, class flowtype>
MaxflowSolver<nweightType, tweightType, flowtype>::~MaxflowSolver()
{
	switch (name)
	{
	case SolverName::BK:
		if (bk_gc)
			delete bk_gc;
		break;
	case SolverName::IBFS:
		if (ibfs_gc) delete ibfs_gc;
		break;
	case SolverName::QPBO_SLVR:
		if (qpbo) 
			delete qpbo;
		break;
	default:
		throw exception("Unsupported method\n");
	}
};
template <class nweightType, class tweightType, class flowtype>
typename flowtype MaxflowSolver<nweightType, tweightType, flowtype>::solve(BasicGraph<nweightType, tweightType> * graph, char * labeling)
{
	switch (name)
	{
	case SolverName::BK:
		return BK_solver(graph, labeling);
		break;
	case SolverName::IBFS:
		return IBFS_solver(graph, labeling);
		break;
	case SolverName::QPBO_SLVR:
		return QPBO_solver(graph, labeling);
		break;
	default:
		throw exception("Unsupported method\n");
	}
	return 0.0;
};

template <class nweightType, class tweightType, class flowtype>
typename flowtype MaxflowSolver<nweightType, tweightType, flowtype>::BK_solver(BasicGraph<nweightType, tweightType> * graph, char * labeling)
{
	clock_t total_start = clock();

	
	BasicGraph<nweightType, tweightType>::t_links_struct tlinks = graph->get_tlinks();
	BasicGraph<nweightType, tweightType>::n_links_struct nlinks = graph->get_nlinks();
	size_t n_nodes = graph->getNumberOfNodes();

	clock_t const_start = clock();
	bk_gc->add_node(n_nodes);
	//set t_link weights 
	for (size_t link_id = 0; link_id < tlinks.n_tlinks; ++link_id)
		bk_gc->add_tweights(tlinks.node_ids[link_id], tlinks.weights[link_id * 2 + 0], tlinks.weights[link_id * 2 + 1]);
	//set n_link weights 
	for (size_t offset = 0; offset < (nlinks.n_nlinks * 2); offset += 2)
		bk_gc->add_edge(nlinks.edge_nodes[offset + 0], nlinks.edge_nodes[offset + 1], nlinks.weights[offset + 0], nlinks.weights[offset + 1]);
	clock_t const_end = clock();

	clock_t max_start = clock();
	flowtype maxflow = bk_gc->maxflow();
	clock_t max_end = clock();
	//cout << ((float)t)/CLOCKS_PER_SEC << endl;

	//get labeling from solver
	memset(labeling, 0, sizeof(char)*n_nodes);
	for (size_t node_id = 0; node_id < n_nodes; ++node_id)
		labeling[node_id] = bk_gc->what_segment(node_id);

	#ifdef MFS_DEBUG
	ssize_t dims[2]; dims[0] = n_nodes; dims[1] = 1;
	int *tmp_labeling = new int[n_nodes];
	for (size_t node_id = 0; node_id < n_nodes; ++node_id)
		tmp_labeling[node_id]=labeling[node_id];
	Utils::ArrayInOut<int> label_writer(tmp_labeling, dims, 2);
	label_writer.save("E:\\hossam\\temp\\bk\\labeling_" + std::to_string(run_id) + ".dat");
	run_id+=1;
	delete[] tmp_labeling;
	std::cout <<"Returned flow: "<< maxflow << std::endl;
	flowtype local_maxflow_compute = computEnergy(graph,labeling);
	if (local_maxflow_compute != maxflow) 
		std::cout << "flow does not match\n";
	#endif

	//always rest before solving a new maxflow
	bk_gc->reset();
	clock_t total_end = clock();

	this->construction_runtime += ((float)(const_end - const_start)) / CLOCKS_PER_SEC;
	this->maxflow_runtime += ((float)(max_end - max_start)) / CLOCKS_PER_SEC;
	this->total_runtime += ((float)(total_end - total_start)) / CLOCKS_PER_SEC;
	return maxflow;
};

template <class nweightType, class tweightType, class flowtype>
typename flowtype MaxflowSolver<nweightType, tweightType, flowtype>::IBFS_solver(BasicGraph<nweightType, tweightType> * graph, char * labeling)
{
	// Initialize here since it has to be re-initialized each time
	clock_t total_start = clock();

	BasicGraph<nweightType, tweightType>::t_links_struct tlinks = graph->get_tlinks();
	BasicGraph<nweightType, tweightType>::n_links_struct nlinks = graph->get_nlinks();
	size_t n_nodes = graph->getNumberOfNodes();

	clock_t const_start = clock();
	// create nodes and set t_link weights
	for (size_t link_id = 0; link_id < tlinks.n_tlinks; ++link_id)
		ibfs_gc->addNode(tlinks.node_ids[link_id], tlinks.weights[link_id * 2 + 0], tlinks.weights[link_id * 2 + 1]);

	// set n_link weights
	for (size_t offset = 0; offset < (nlinks.n_nlinks * 2); offset += 2)
		ibfs_gc->addEdge(nlinks.edge_nodes[offset + 0], nlinks.edge_nodes[offset + 1], nlinks.weights[offset + 0], nlinks.weights[offset + 1]);
	clock_t const_end = clock();

	clock_t max_start = clock();
	ibfs_gc->initGraph();
	flowtype maxflow = ibfs_gc->computeMaxFlow();
	clock_t max_end = clock();	

	// get labelling from solver
	memset(labeling, 0, sizeof(char)*n_nodes);
	
	for (size_t node_id = 0; node_id < n_nodes; ++node_id)
		labeling[node_id] = 1-ibfs_gc->isNodeOnSrcSide(node_id,1);
	long long  local_maxflow_compute = computEnergy(graph, labeling);
	//bgn_log << LogType::INFO_LEVEL1 << local_maxflow_compute << "\n" << end_log;

	if (local_maxflow_compute != maxflow)
		for (size_t node_id = 0; node_id < n_nodes; ++node_id)
			labeling[node_id] = 1-ibfs_gc->isNodeOnSrcSide(node_id,0);

	#ifdef MFS_DEBUG
	ssize_t dims[2]; dims[0] = n_nodes; dims[1] = 1;
	int *tmp_labeling = new int[n_nodes];
	for (size_t node_id = 0; node_id < n_nodes; ++node_id)
		tmp_labeling[node_id] = labeling[node_id];
	Utils::ArrayInOut<int> label_writer(tmp_labeling, dims, 2);
	label_writer.save("E:\\hossam\\temp\\ibfs\\labeling_" + std::to_string(run_id) + ".dat");
	run_id += 1;
	delete[] tmp_labeling;
	std::cout << "Returned flow: " << maxflow << std::endl;
	flowtype local_maxflow_compute = computEnergy(graph, labeling);
	if (local_maxflow_compute != maxflow)
		std::cout << "\tflows does not match: " << maxflow << " vs " << local_maxflow_compute << std::endl;
	#endif
	
	clock_t total_end = clock();
	//always rest before solving the new maxflow for the new basicgraph
	ibfs_gc->reset();

	this->construction_runtime += ((float)(const_end - const_start)) / CLOCKS_PER_SEC;
	this->maxflow_runtime += ((float)(max_end - max_start)) / CLOCKS_PER_SEC;
	this->total_runtime += ((float)(total_end - total_start)) / CLOCKS_PER_SEC;
	return maxflow;
};

template <class nweightType, class tweightType, class flowtype>
typename flowtype MaxflowSolver<nweightType, tweightType, flowtype>::QPBO_solver(BasicGraph<nweightType, tweightType> * graph, char * labeling)
{
	//always rest before solving the new maxflow for the new basicgraph
	clock_t total_start = clock();

	
	BasicGraph<nweightType, tweightType>::t_links_struct tlinks = graph->get_tlinks();
	BasicGraph<nweightType, tweightType>::n_links_struct nlinks = graph->get_nlinks();
	size_t n_nodes = graph->getNumberOfNodes();

	
	//s=0, t=1
	clock_t const_start = clock();
	qpbo->AddNode(n_nodes);
	//set t_link weights 
	for (size_t link_id = 0; link_id < tlinks.n_tlinks; ++link_id)
		qpbo->AddUnaryTerm(tlinks.node_ids[link_id], tlinks.weights[link_id * 2 + 1], tlinks.weights[link_id * 2 + 0]);
	//set n_link weights 
	for (size_t offset = 0; offset < (nlinks.n_nlinks * 2); offset += 2)
		qpbo->AddPairwiseTerm(nlinks.edge_nodes[offset + 0], nlinks.edge_nodes[offset + 1],0, nlinks.weights[offset + 0], nlinks.weights[offset + 1],0);
	clock_t const_end = clock();

	clock_t max_start = clock();
	qpbo->MergeParallelEdges();
	qpbo->Solve();
	qpbo->ComputeWeakPersistencies();
	flowtype maxflow = qpbo->ComputeTwiceEnergy() / 2;
	clock_t max_end = clock();
	//cout << ((float)t)/CLOCKS_PER_SEC << endl;

	//get labeling from solver
	memset(labeling, 0, sizeof(char)*n_nodes);
	for (size_t node_id = 0; node_id < n_nodes; ++node_id)
		if (qpbo->GetLabel(node_id) >= 0)
			labeling[node_id] = qpbo->GetLabel(node_id);
		else
			bool tft = 1;//ThreadSafeCout::Print("Unlabeled pixel\n");
	
	#ifdef MFS_DEBUG
	ssize_t dims[2]; dims[0] = n_nodes; dims[1] = 1;
	int *tmp_labeling = new int[n_nodes];
	for (size_t node_id = 0; node_id < n_nodes; ++node_id)
		tmp_labeling[node_id] = labeling[node_id];
	Utils::ArrayInOut<int> label_writer(tmp_labeling, dims, 2);
	label_writer.save("E:\\hossam\\temp\\qpbo\\labeling_" + std::to_string(run_id) + ".dat");
	run_id += 1;
	delete[] tmp_labeling;
	std::cout << "Returned flow: " << maxflow << std::endl;
	flowtype local_maxflow_compute = computEnergy(graph, labeling);
	if (local_maxflow_compute != maxflow)
		std::cout << "\tflows does not match: " << maxflow << " vs " << local_maxflow_compute << std::endl;
	#endif
	//always rest before solving a new maxflow
	qpbo->Reset();	
	clock_t total_end = clock();

	this->construction_runtime += ((float)(const_end - const_start)) / CLOCKS_PER_SEC;
	this->maxflow_runtime += ((float)(max_end - max_start)) / CLOCKS_PER_SEC;
	this->total_runtime += ((float)(total_end - total_start)) / CLOCKS_PER_SEC;
	return maxflow;
};

#endif