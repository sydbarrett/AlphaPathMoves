#ifndef __BASICGRAPH_H__
#define __BASICGRAPH_H__
#include <iostream>
#include <cstdint>
#include "Utilities.h"
#include "TS_Logger.h"

template <class nweightType, class tweightType> class BasicGraph
{
private:
	uint64_t n_nodes;
	void update_nnodes(uint64_t node_idx) { if (n_nodes < node_idx + 1) n_nodes = node_idx; };
public:
	struct t_links_struct{
		tweightType * weights; // source, then sink; double size node_ids; double n_tlinks
		uint64_t * node_ids;
		uint64_t n_tlinks;
		uint64_t maxcap;
	} t_links;

	struct n_links_struct{
		nweightType * weights; // n1->n2, then n1<-n2; same size edge_nodes; double n_nlinks
		uint64_t * edge_nodes;
		uint64_t n_nlinks;
		uint64_t maxcap;
	} n_links;
	

	BasicGraph();
	BasicGraph(uint64_t, uint64_t , uint64_t ); //set the n_nodes and n_edges maximum capcity, this cap could be excceded but this will be slow
	void reset(); 
	void add_nlink(uint64_t, uint64_t, nweightType, nweightType);
	void add_tlink(uint64_t, tweightType, tweightType);
	void get_tlink(uint64_t, tweightType&, tweightType&);
	void get_nlink(uint64_t, uint64_t &, uint64_t&, nweightType &, nweightType &);
	
	t_links_struct get_tlinks();
	n_links_struct get_nlinks();
	void saveToFile(std::string fname);
	void saveToFile_Binary(std::string fname);
	void saveToMaxflowFile(std::string fname);

	uint64_t getNumberOfNodes() { return this->n_nodes; }
	double getAllocatedMemSizeInGB();
	~BasicGraph();
};
template <class nweightType, class tweightType>
void BasicGraph<nweightType, tweightType>::saveToMaxflowFile(std::string output_fname)
{
	ofstream output_file(output_fname.c_str(), ios::out);
	uint64_t s = 1;
	uint64_t t = 2;

	output_file << "p max " << this->n_nodes << " " << this->t_links.n_tlinks + this->n_links.n_nlinks << std::endl;
	output_file << "n 1 s" << std::endl;
	output_file << "n 2 t" << std::endl;
	for (uint64_t link_id = 0; link_id < t_links.n_tlinks; ++link_id)
	{
		output_file <<"a "<< s << " " << t_links.node_ids[link_id]+3 << " " << t_links.weights[link_id * 2 + 0] << std::endl;
		output_file <<"a "<< t_links.node_ids[link_id]+3 << " " << t << " " << t_links.weights[link_id * 2 + 1] << std::endl;
	}
	for (uint64_t offset = 0; offset < (n_links.n_nlinks * 2); offset += 2)
	{
		output_file << "a " << n_links.edge_nodes[offset + 0] + 3 << " " << n_links.edge_nodes[offset + 1] + 3 << " " << n_links.weights[offset + 0] << std::endl;
		output_file << "a " << n_links.edge_nodes[offset + 1] + 3 << " " << n_links.edge_nodes[offset + 0] + 3 << " " << n_links.weights[offset + 1] << std::endl;
	}
	output_file.close();
}

template <class nweightType, class tweightType>
BasicGraph<nweightType, tweightType>::BasicGraph()
{
	this->t_links.n_tlinks = 0;
	this->t_links.maxcap = 0;
	this->t_links.weights = nullptr;
	this->t_links.node_ids = nullptr;

	this->n_links.n_nlinks = 0;
	this->n_links.maxcap = 0;
	this->n_links.weights = nullptr;
	this->n_links.edge_nodes = nullptr;
};

template <class nweightType, class tweightType>
BasicGraph<nweightType, tweightType>::BasicGraph(uint64_t n_nodes, uint64_t n_tlinks_maxcap, uint64_t n_nlinks_maxcap)
{
	this->n_nodes = n_nodes;
	this->t_links.n_tlinks = 0;
	this->t_links.maxcap = n_tlinks_maxcap;
	this->t_links.weights = new tweightType[this->t_links.maxcap*2];
	this->t_links.node_ids = new uint64_t[this->t_links.maxcap];

	this->n_links.n_nlinks = 0;
	this->n_links.maxcap = n_nlinks_maxcap;
	this->n_links.weights = new nweightType[this->n_links.maxcap * 2];
	this->n_links.edge_nodes = new uint64_t[this->n_links.maxcap * 2];
};
template <class nweightType, class tweightType>
double BasicGraph<nweightType, tweightType>::getAllocatedMemSizeInGB( void )
{
	double size_inbytes = sizeof(tweightType) *this->t_links.maxcap * 2 + sizeof(uint64_t)*this->t_links.maxcap + sizeof(nweightType) * this->n_links.maxcap * 2 + sizeof(uint64_t)* this->n_links.maxcap * 2;
	return (size_inbytes/1024/1024/1024);
}
template <class nweightType, class tweightType>
BasicGraph<nweightType, tweightType>::~BasicGraph()
{
	CLNDEL1D(this->t_links.weights);
	CLNDEL1D(this->t_links.node_ids);
	CLNDEL1D(this->n_links.weights);
	CLNDEL1D(this->n_links.edge_nodes);
}
template <class nweightType, class tweightType>
void BasicGraph<nweightType, tweightType>::reset()
{
	//memset(this->t_links.weights, 0, sizeof(tweightType)*this->t_links.maxcap*2);
	//memset(this->t_links.node_ids, 0, sizeof(uint64_t)*this->t_links.maxcap);
	//memset(this->n_links.weights, 0, sizeof(nweightType)*this->n_links.maxcap * 2);
	//memset(this->n_links.edge_nodes, 0, sizeof(uint64_t)*this->n_links.maxcap * 2);
	this->t_links.n_tlinks = 0;
	this->n_links.n_nlinks = 0;
}

template <class nweightType, class tweightType>
void BasicGraph<nweightType, tweightType>::add_nlink(uint64_t node0_idx, uint64_t node1_idx, nweightType w_0to1, nweightType w_1to0)
{
	if (this->n_links.n_nlinks == this->n_links.maxcap) //maxcap will be excceded
	{
		bgn_log << LogType::WARNING << "BG: Warning number of estimated n_links was exceeded, use a higher estimate to avoid slow runnning time\n" << end_log;
		nweightType * new_weights = new nweightType[n_links.maxcap * 4];
		memcpy(new_weights, this->n_links.weights, sizeof(nweightType)* this->n_links.n_nlinks * 2);
		memset(new_weights + this->n_links.n_nlinks * 2, 0, sizeof(nweightType)*(n_links.maxcap * 4 - this->n_links.n_nlinks * 2));
		CLNDEL1D(this->n_links.weights);
		this->n_links.weights = new_weights;

		uint64_t * new_edge_nodes = new uint64_t[n_links.maxcap * 4];
		memcpy(new_edge_nodes, this->n_links.edge_nodes, sizeof(uint64_t)* this->n_links.n_nlinks * 2);
		memset(new_edge_nodes + this->n_links.n_nlinks * 2, 0, sizeof(uint64_t)*(n_links.maxcap * 4 - this->n_links.n_nlinks * 2));
		CLNDEL1D(this->n_links.edge_nodes);
		this->n_links.edge_nodes = new_edge_nodes;
		n_links.maxcap *= 2;
	}
	uint64_t edge_id = this->n_links.n_nlinks * 2;
	this->n_links.edge_nodes[edge_id + 0] = node0_idx;
	this->n_links.edge_nodes[edge_id + 1] = node1_idx;
	this->n_links.weights[edge_id + 0] = w_0to1;
	this->n_links.weights[edge_id + 1] = w_1to0;
	this->n_links.n_nlinks++;
	update_nnodes(node0_idx);
	update_nnodes(node1_idx);
}
template <class nweightType, class tweightType>
void BasicGraph<nweightType, tweightType>::add_tlink(uint64_t node_idx, tweightType w_src2node, tweightType w_node2sink)
{
	if (this->t_links.maxcap == this->t_links.n_tlinks) //maxcap will be excceded
	{
		bgn_log << LogType::WARNING << "BG: Warning number of estimated t_links was exceeded, use a higher estimate to avoid slow runnning time\n" << end_log;
		tweightType * new_weights = new tweightType[this->t_links.maxcap * 4];
		memcpy(new_weights, this->t_links.weights, sizeof(tweightType)* this->t_links.maxcap*2);
		memset(new_weights + this->t_links.maxcap * 2, 0, sizeof(tweightType)*(t_links.maxcap * 4 - this->t_links.maxcap*2));
		CLNDEL1D(this->t_links.weights);
		this->t_links.weights = new_weights;

		uint64_t * new_node_ids = new uint64_t[this->t_links.maxcap * 2];
		memcpy(new_node_ids, this->t_links.node_ids, sizeof(uint64_t)* this->t_links.maxcap);
		memset(new_node_ids + this->t_links.maxcap, 0, sizeof(uint64_t)*(t_links.maxcap * 2 - this->t_links.maxcap));
		CLNDEL1D(this->t_links.node_ids);
		this->t_links.node_ids = new_node_ids;
		this->t_links.maxcap *= 2;
	}
	
	t_links.node_ids[this->t_links.n_tlinks] = node_idx;
	t_links.weights[this->t_links.n_tlinks * 2 + 0] = w_src2node;
	t_links.weights[this->t_links.n_tlinks * 2 + 1] = w_node2sink;
	this->t_links.n_tlinks++;
	update_nnodes(node_idx);
}
template <class nweightType, class tweightType>
void  BasicGraph<nweightType, tweightType>::get_tlink(uint64_t link_id, tweightType& w_src2node, tweightType&w_node2sink)
{
	if (link_id >= this->t_links.n_tlinks)
	{
		bgn_log << LogType::ERROR << "BG: node access violation\n" << end_log;
		throw exception("Access violation");
	}
		
	w_src2node = this->t_links.weights[link_id * 2 + 0];
	w_node2sink = this->t_links.weights[link_id * 2 + 1];
}

template <class nweightType, class tweightType>
void BasicGraph<nweightType, tweightType>::get_nlink(uint64_t edge_idx, uint64_t & node0_idx, uint64_t& node1_idx, nweightType& w_0to1, nweightType& w_1to0)
{
	if (edge_idx >= this->n_links.n_nlinks)
	{
		bgn_log << LogType::ERROR << "BG: edge access violation\n" << end_log;
		throw exception("Access violation");
	}
	uint64_t lcl_edge_idx = edge_idx * 2;
	node0_idx = this->n_links.edge_nodes[lcl_edge_idx + 0];
	node1_idx = this->n_links.edge_nodes[lcl_edge_idx + 0];
	w_0to1 = this->n_links.weights[lcl_edge_idx + 0];
	w_1to0 = this->n_links.weights[lcl_edge_idx + 1];
};

template <class nweightType, class tweightType>
typename BasicGraph<nweightType, tweightType>::t_links_struct BasicGraph<nweightType, tweightType>::get_tlinks()
{
	return this->t_links;
}

template <class nweightType, class tweightType>
typename BasicGraph<nweightType, tweightType>::n_links_struct BasicGraph<nweightType, tweightType>::get_nlinks()
{
	return this->n_links;
}

template <class nweightType, class tweightType>
void BasicGraph<nweightType, tweightType>::saveToFile(std::string fname)
{
	int64_t dim2[2];
	dim2[0] = 2;
	dim2[1] = this->t_links.n_tlinks;
	Utils::ArrayInOut<tweightType> tweights(this->t_links.weights, dim2, 2);
	tweights.save(fname + "_BG_tlink_weights.dat");
	dim2[0] = 1;
	Utils::ArrayInOut<uint64_t> nodes(this->t_links.node_ids, dim2, 2);
	tweights.save(fname + "_BG_tlink_nodes.dat");
	
	dim2[0] = 2;
	dim2[1] = this->n_links.n_nlinks;
	Utils::ArrayInOut<nweightType> nweights(this->n_links.weights, dim2, 2);
	nweights.save(fname + "_BG_nlink_weights.dat");
	Utils::ArrayInOut<uint64_t> edges(this->n_links.edge_nodes, dim2, 2);
	edges.save(fname + "_BG_nlink_nodes.dat");

}
template <class nweightType, class tweightType>
void BasicGraph<nweightType, tweightType>::saveToFile_Binary(std::string fname)
{
	
	ofstream output_file1((fname+"_tlinks_weights.dat").c_str(), ios::out | ios::binary);
	output_file1.write(reinterpret_cast<const char*>(this->t_links.weights), sizeof(tweightType)*this->t_links.n_tlinks * 2);
	output_file1.close();

	ofstream output_file2((fname + "_tlinks_nodes.dat").c_str(), ios::out | ios::binary);
	output_file2.write(reinterpret_cast<const char*>(this->t_links.node_ids), sizeof(uint64_t)*this->t_links.n_tlinks);
	output_file2.close();


	ofstream output_file3((fname + "_nlinks_weights.dat").c_str(), ios::out | ios::binary);
	output_file3.write(reinterpret_cast<const char*>(this->n_links.weights), sizeof(nweightType)* this->n_links.n_nlinks * 2);
	output_file3.close();

	ofstream output_file4((fname + "_nlinks_nodes.dat").c_str(), ios::out | ios::binary);
	output_file4.write(reinterpret_cast<const char*>(this->n_links.edge_nodes), sizeof(uint64_t)* this->n_links.n_nlinks * 2);
	output_file4.close();

}

#endif