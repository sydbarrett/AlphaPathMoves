/*
#################################################################################
#                                                                            	#
# 								AlphaPathMoves 								    #
# software for optimizing Hierarchically-structured Interacting Segments (HINTS)#
#			                     Beta Version                                   #
#                                                                            	#
#    				Copyright  Hossam Isack 									#
#						       isack.hossam@gmal.com							#
#							   http://www.hossamisack.com						#
#              			                                                        #
#################################################################################


  Minimum requirements 
	[1] 64-bit machine
	[2] C++11 
	[3] CUDA architecture sm_30 or higher

  To use this software, YOU MUST CITE the following in any resulting publication:

    [1] Efficient optimization for Hierarchically-structured Interacting Segments (HINTS)
		Hossam Isack, Olga Veksler, Ipek Oguz, Milan Sonka, and Yuri Boykov.
		@inproceedings{pathmovesIsack, title={Efficient optimization for Hierarchically-structured 
		Interacting Segments ({HINTS})}, author={Isack, Hossam and Veksler, Olga and Oguz, Ipek and 
		Sonka, Milan and Boykov, Yuri}, booktitle = {IEEE Conference on Computer Vision and Pattern 
		Recognition}, year = {2017}}

  Furthermore, if you using Hedgehog shape prior, you should cite

    [4] Hedgehog Shape Priors for Multi-object Segmentation
		Hossam Isack, Olga Veksler, Milan Sonka, and Yuri Boykov.		
		@inproceedings{hedgehogIsack, title={Hedgehog Shape Priors for Multi-object Segmentation},
		author={Isack, Hossam and Veksler, Olga and Sonka, Milan and Boykov, Yuri}, 
		booktitle = {IEEE Conference on Computer Vision and Pattern Recognition}, year = {2016}}
	
  This software can be used only for research purposes. For commercial purposes, 
  please contacnt Hossam Isack.

  ##################################################################

    License & disclaimer.

    Copyright Hossam Isack 	<isack.hossam@gmal.com>

    This software and its modifications can be used and distributed for 
    research purposes only. Publications resulting from use of this code
    must cite publications according to the rules given above. Only
    Hossam Isack has the right to redistribute this code, unless expressed
    permission is given otherwise. Commercial use of this code, any of 
    its parts, or its modifications is not permited. The copyright notices 
    must not be removed in case of any modifications. This Licence 
    commences on the date it is electronically or physically delivered 
    to you and continues in effect unless you fail to comply with any of 
    the terms of the License and fail to cure such breach within 30 days 
    of becoming aware of the breach, in which case the Licence automatically 
    terminates. This Licence is governed by the laws of Canada and all 
    disputes arising from or relating to this Licence must be brought 
    in Toronto, Ontario.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

##################################################################
*/
#ifndef __ALPHAPATHMOVES_H__
#define __ALPHAPATHMOVES_H__
#include "BK\graph.h"
#include "QPBO\QPBO.h"
#include "Utilities.h"
#include <assert.h>
#include <algorithm>
#include <vector>
#include <set>
#include <ctime>
#include <utility>
#include <fstream>
#include <iostream>
#include <limits>
#include <iostream>
#include <utility>
#include <thread>
#include <queue>
#include <cstdint>
#include "BasicGraph.h"
#include "MaxflowSolver.h"
#include "TS_Logger.h"
#include "HedgehogConstraints.cuh"
#include "MarginEdgesCuda.cuh"
#include "sobelFilter.cuh"
#include "GirdNeighboursGenerator.h"
#include "EuclideanDT.h"

#undef max
#undef min


#define getqidx(dx, dy, dz, r, twor1, twor1_sq, includeZ) (dx+r+twor1*(dy+r)+includeZ*twor1_sq*(dz+r))
#define getConvIdx(r,twor1,includeZ) (2 * r + 2 * r * twor1 + 2 * r* twor1*twor1 * includeZ)
#define INLINE_SETBIT(p,idx) \
	uint32_t bnk_id = (idx / 64); \
	minmarginBIT->edges.data[p*minmarginBIT->n_banks + bnk_id] |= (((uint64_t)1) << (idx - bnk_id * 64));
#define INLINE_LOOKUPSETBIT(p,idx) \
	minmarginBIT->edges.data[p*minmarginBIT->n_banks + minmarginBIT->gbitid2bnkid[idx]] |= (((uint64_t)1) << minmarginBIT->gbitid2bitid[idx]);


using namespace Utilities;

double default_getWpq_Euclidean(int64_t, int64_t, const void *) { return 1; }


namespace GC{
	template <class captype, class tcaptype, class flowtype> class AlphaPathMoves
	{
	public:
		enum class MinMarginMethod { IDX_BASED, BIT_BASED, CUDA_BASED };
	private:
		
		uint32_t nlabels;
		int64_t dims[3];
		int32_t * child2parent; //This MUST BE signed because the root has no parent, and it is identified by having -1 as its parent
		Array3D<int32_t> *paths; 
		Array2D<int32_t> *paths_lengths; //resereved for future use
		Array2D<tcaptype> *dataterms;

		

		//Constructed Graph
		int64_t ndspl, gc_nds;
		uint32_t max_nlayers;
		BasicGraph<captype, tcaptype> *graph;
		uint32_t *labeling;
		std::vector<std::thread> labeling_writingthreads;
		MaxflowSolver<captype, tcaptype, flowtype> * maxflow_solver;


		//Smoothness Prior
		double lambda;
		Array2D<int32_t>   * sm_nh_shifts;
		Array2D<double> * treemetric;
		std::function<double(int64_t, int64_t, const void *)> func_getWpq; // function called to compute wpq
		const void * wpq_extra_data; //extra data sent to func_getWpq along with p and q index

		//Hedgehog Priors
		uint32_t hhog_radius;
		bool hhogprior_status;
		std::vector<bool> hhoglabel_status;
		Array2D<int32_t> * hhog_nh_shifts;
		Array2D<char> ** hhogconstraints;
		double hhog_theta;

		//Min-Margin Priors
		MinMarginMethod mm_method;
		uint32_t max_n_mm_nhs, max_mm_r;
		Array2D<int32_t>   ** mm_nh_shifts;
		struct MinMarginIDX{
			MinMarginIDX(int32_t max_n_mm_nhs, uint64_t ndspl, int32_t nlabels, int32_t * child2parent)
			{
				edges.allocate(max_n_mm_nhs,ndspl);
				std::fill_n(edges.data,ndspl*max_n_mm_nhs, -1);

				InSubTree = new Array2D<int32_t>(); // InSubTree(x,y) if 1 then x in Tree(y) 0 otherwise
				InSubTree->allocate(nlabels,nlabels);
				InSubTree->fill(0);
				for (int32_t x = 0; x < nlabels; x++)
					for (int32_t parent_id = x; parent_id != -1; parent_id = child2parent[parent_id])
						InSubTree->data[x + parent_id* InSubTree->constX] = 1;

				NotInSubTreeOrParent = new Array2D<int32_t>(); // NotInSubTreeOrParent(x,y) if 1 then x not (in Tree(y) or Parent[y]) 0 otherwise
				NotInSubTreeOrParent->allocate(nlabels, nlabels);
				NotInSubTreeOrParent->fill(0);
				for (int32_t x = 0; x < nlabels; x++)
					for (int32_t y = 0; y < nlabels; y++)
						if (!(InSubTree->data[x + y*NotInSubTreeOrParent->constX] || child2parent[y] == x))
							NotInSubTreeOrParent->data[x + y*NotInSubTreeOrParent->constX] = 1;
			};
			~MinMarginIDX()
			{
				CLNDEL0D(InSubTree);
				CLNDEL0D(NotInSubTreeOrParent);
			}

			Array2D<int32_t> * InSubTree, *NotInSubTreeOrParent;
			Array2D<int64_t>  edges;
		} * minmarginIDX; // helper data structure for the IDX encoding;
		struct MinMarginBIT
		{
			MinMarginBIT(int32_t max_n_mm_nhs, uint64_t ndspl, int32_t max_mm_r, Array2D<int32_t> * mm_shifts)
			{
				max_mm_ws = 2 * max_mm_r + 1;

				gbitid2shift = new int32_t[pow3(max_mm_ws) * 3];
				std::fill_n(gbitid2shift, pow3(max_mm_ws) * 3, -1);

				shift2gbitid = new int32_t **[max_mm_ws];
				for (int32_t x = 0; x < max_mm_ws; ++x)
				{
					shift2gbitid[x] = new int32_t *[max_mm_ws];
					for (int32_t y = 0; y < max_mm_ws; ++y)
					{
						shift2gbitid[x][y] = new int32_t[max_mm_ws];
						std::fill_n(shift2gbitid[x][y], max_mm_ws, -1);
					}

				}
				int32_t dx, dy, dz;
				for (int32_t shift_id = 0, bitid = 0; shift_id < mm_shifts->Y; shift_id++, bitid++)
				{
					dx = mm_shifts->data[shift_id * 3 + 0];
					dy = mm_shifts->data[shift_id * 3 + 1];
					dz = mm_shifts->data[shift_id * 3 + 2];

					gbitid2shift[bitid * 3 + 0] = dx;
					gbitid2shift[bitid * 3 + 1] = dy;
					gbitid2shift[bitid * 3 + 2] = dz;
					shift2gbitid[dx + max_mm_r][dy + max_mm_r][dz + max_mm_r] = bitid;
				}
				//shift2gbitid to avoid adding max_mm_r all the time 
				for (int32_t x = 0; x < max_mm_ws; ++x)
				{
					for (int32_t y = 0; y < max_mm_ws; ++y)
						shift2gbitid[x][y] += max_mm_r;
					shift2gbitid[x] += max_mm_r;
				}
				shift2gbitid += max_mm_r;
				n_banks = (uint64_t)ceil((double)max_n_mm_nhs / 64.0);
				edges.allocate(n_banks, ndspl);
				n_usedbits = max_n_mm_nhs;
				memset(edges.data, 0, sizeof(uint64_t)*n_banks*ndspl);

				gbitid2bnkid = new int32_t[n_banks * 64];
				gbitid2bitid = new int32_t[n_banks * 64];
				for (uint32_t bnk_id = 0, g_bitid = 0; bnk_id < n_banks; ++bnk_id)
				{
					std::fill_n(gbitid2bnkid + bnk_id * 64, 64, bnk_id);
					for (uint32_t itr = 0; itr < 64; itr++, ++g_bitid)
						gbitid2bitid[g_bitid] = itr;
				}
			}
			~MinMarginBIT()
			{
				//Before deleting shift2gbitid, revert it to its original indexing base
				//Reminder:shift2gbitid was modified to make it point to the center of the allocated memory (so we can used dx,dy,dz without adding offest)
				int32_t max_mm_r = (max_mm_ws - 1) / 2;
				if (shift2gbitid != nullptr)
				{
					shift2gbitid -= max_mm_r;
					for (int32_t x = 0; x < max_mm_ws; ++x)
					{
						shift2gbitid[x] -= max_mm_r;
						for (int32_t y = 0; y < max_mm_ws; ++y)
							shift2gbitid[x][y] -= max_mm_r;
					}
					for (int32_t x = 0; x < max_mm_ws; ++x)
						CLNDEL2D(shift2gbitid[x], max_mm_ws);
					delete[](shift2gbitid);
					shift2gbitid = nullptr;
				}
				CLNDEL1D(gbitid2shift);
				CLNDEL1D(gbitid2bnkid);
				CLNDEL1D(gbitid2bitid);
			}
			Array2D<uint64_t> edges;
			int32_t max_mm_ws;        // holds larget min-margin window size = 2*max_mm_r+1
			uint64_t n_banks;       // the number of  n uint64_t allocated for each voxel
			uint64_t n_usedbits;    // the total number bits used out of the n_banks*sizeof(uint64_t) bits to encode minmargin edges.
			int32_t ***shift2gbitid;  // a 3D lookup table from minmargin shift [dx] [dy] [dz] to its corresponding global bit_id (w.r.t. n_usedbits) 
			int32_t *gbitid2shift;    // a 1D lookup table from global bit_id (w.r.t. n_usedbits) to its corresponding mm shift (dx,dy,dz).
			int32_t * gbitid2bnkid;   // a 1D lookup table from a global bit id to its corresponding bank (out of the n_banks)
			int32_t * gbitid2bitid;   // a 1D lookup table from a global bit id to bit id w.r.t. to its corresponding bank
		} * minmarginBIT; // helper data structure for the BIT encoding;
		MarginEdgesCuda * marginEdgesCuda;
		
		//expansion_ordering
		struct ExpansionOrdering
		{
			enum class ExpansionOrderingType { CHAIN, VECTOR, SEED };
			int32_t chain_str_idx, chain_end_idx; //in case of a chain
			int32_t randomization_seed; 
			Array2D<uint32_t> label_ordering;
			ExpansionOrderingType type;
		} expansionOrdering;
		struct CalledFunctions
		{
			bool dataterms;
			bool sm_wsize;
			bool sm_tree_lambda;
			bool init_labeling;
			bool sm_wpq_func;
			bool mm_set;
			bool hhog_set;
			bool default_mode;
		} called_functions;

		void setPairwiseSmoothnessGCArcs(Array3D<char> *, flowtype *, captype  *, int64_t*, tcaptype *, int64_t*, uint32_t, uint32_t);
		void setMinMarginGCArcs(uint32_t, uint32_t);
		void setHedgehogGCArcs(Array3D<char> *);

		void fillMarginConstraints_IDX_BASED(uint32_t, uint32_t);
		void fillMarginConstraints_BIT_BASED(uint32_t, uint32_t);
		void fillPairewisePathSmoothnessArcs(int64_t, int64_t, uint32_t, uint32_t, uint32_t, captype *, int64_t*, int64_t &, tcaptype *,int64_t*, int64_t &);
		void fillPathMoveHedgehogConstraints(Array3D<char> *, uint32_t, uint32_t);
		void setupMaxflowSolver(typename MaxflowSolver<captype, tcaptype, flowtype>::SolverName);
		
		
		flowtype computeEnergyForLabeling(uint32_t *);
		flowtype computeEnergyForLabeling(uint32_t *, flowtype &, flowtype &);
		void saveLabelingToFile(uint32_t *, std::string);
		void convertGCBinLabeling2MultiLabeling(uint32_t *, uint32_t, char *);
		captype LRG_PENALTY;
		bool ValidateLabeling(char * , BasicGraph<captype, tcaptype> * ); //asserts that no inifity cost edges are severed
	
	public:
		AlphaPathMoves(int64_t dims[3], uint32_t in_nlabels);
		~AlphaPathMoves(void);
		static captype getLargePenalty( int32_t n_dims) 
		{ 
			return std::min<captype>(10000000000, std::numeric_limits<captype>::max() / 100);
		}
		void getRuntime(double & , double &, double &);
		void setWpqFunction(double(*in_getWpq)(int64_t, int64_t, const void * ), const void *);
		void setDataTerms(const Array2D<tcaptype> * );
		void setSmoothnessNeighbourhoodWindowSize(uint32_t);
		void setHedgehogAttributes(uint32_t, const Array3D<int32_t> *, double);
		void setTreeWeights(double, const Array2D<double> *  );
		void setMinimumMargins(const Array2D<uint32_t> *in_minmargins_radii, MinMarginMethod in_mm_method = MinMarginMethod::CUDA_BASED);
		void setInitialLabeling(const Array3D<uint32_t> *);
		void setInitialLabeling(uint32_t);
		void setExpansionOrdering(int32_t str_id = -1, int32_t exp_id = -1);
		void setExpansionOrdering(int32_t seed);
		void setExpansionOrdering(const Array2D<uint32_t> * label_ordering);
		Array3D<uint32_t> * runPathMoves(typename MaxflowSolver<captype, tcaptype, flowtype>::SolverName, flowtype & final_energy);
	};
	//The constructor accepts the following
	//a- dims of the volume (it deals with 2D images as 3D volumes consisting of one slice)
	//b- the number of labels
	template <class captype, class tcaptype, class flowtype>
	AlphaPathMoves<captype, tcaptype, flowtype>::AlphaPathMoves(int64_t dims[3],  uint32_t in_nlabels)
	{
		called_functions.default_mode = true;
		if (dims[2]==1)
			LRG_PENALTY = this->getLargePenalty(2);
		else
			LRG_PENALTY = this->getLargePenalty(3);

		this->nlabels = in_nlabels;
		this->dims[0] = dims[0];
		this->dims[1] = dims[1];
		this->dims[2] = dims[2];

		this->dataterms = nullptr;
		this->child2parent = nullptr;
		this->paths = nullptr;
		this->paths_lengths = nullptr;

		//Constructed graph
		this->ndspl = dims[0]* dims[1]* dims[2];
		this->max_nlayers = 0;
		this->gc_nds = 0;
		this->labeling = new uint32_t[this->ndspl];
		std::memset(this->labeling, 0, this->ndspl*sizeof(uint32_t));
		this->labeling_writingthreads.clear();
		this->graph = nullptr;
		this->maxflow_solver = nullptr;

		//smoothness
		this->sm_nh_shifts = nullptr;
		this->treemetric = nullptr;
		this->func_getWpq = default_getWpq_Euclidean;
		this->wpq_extra_data = nullptr;
		
		//hedgehogs
		this->hhogprior_status = false;
		this->hhog_theta = 90;
		this->hhog_radius = 0;
		this->hhoglabel_status.clear();
		this->hhog_nh_shifts = nullptr;
		this->hhogconstraints = new Array2D<char>*[this->nlabels];
		for (uint32_t i = 0; i < this->nlabels; ++i) 
			this->hhogconstraints[i] = nullptr;

		//min-margins
		this->mm_method = MinMarginMethod::CUDA_BASED;
		this->max_n_mm_nhs = 0;
		this->max_mm_r = 0;
		this->mm_nh_shifts = nullptr;
		this->minmarginIDX = nullptr;
		this->minmarginBIT = nullptr;
		this->marginEdgesCuda = nullptr;
		Array2D<uint32_t> in_minmargins_radii;
		in_minmargins_radii.allocate(this->nlabels, 1);
		in_minmargins_radii.fill(0);
		setMinimumMargins(&in_minmargins_radii, this->mm_method);

		//expansion ordering 
		expansionOrdering.type = ExpansionOrdering::ExpansionOrderingType::SEED;
		expansionOrdering.randomization_seed = 101;
		expansionOrdering.chain_str_idx = -1;
		expansionOrdering.chain_end_idx = -1;
		expansionOrdering.label_ordering.allocate(0, 0);

		//called_functions status 
		called_functions.dataterms=false;
		called_functions.sm_wsize=false;
		called_functions.sm_tree_lambda=false;
		called_functions.init_labeling=false;
		called_functions.sm_wpq_func = false;
		called_functions.mm_set = false;
		called_functions.hhog_set = false;
		called_functions.default_mode = false;
	}
	template <class captype, class tcaptype, class flowtype>
	AlphaPathMoves<captype, tcaptype, flowtype>::~AlphaPathMoves(void)
	{
		CLNDEL0D(dataterms);
		CLNDEL1D(child2parent);
		CLNDEL0D(paths);
		CLNDEL0D(paths_lengths);

		//Constructed Graph
		CLNDEL1D(this->labeling);
		this->labeling_writingthreads.clear();
		CLNDEL0D(graph);
		CLNDEL0D(maxflow_solver);

		//Smoothness Prior
		CLNDEL0D(sm_nh_shifts);
		CLNDEL0D(treemetric);

		//Hedgehog Prior
		CLNDEL0D(hhog_nh_shifts);
		for (uint32_t i = 0; this->hhogconstraints != nullptr && i < this->nlabels; ++i)
			CLNDEL0D(this->hhogconstraints[i]);
		CLNDEL1D(this->hhogconstraints);


		//Min-margin prior
		for (uint32_t l = 0; this->mm_nh_shifts != nullptr && l < this->nlabels; ++l)
			CLNDEL0D(this->mm_nh_shifts[l]);
		CLNDEL1D(this->mm_nh_shifts);
		CLNDEL0D(minmarginIDX);
		CLNDEL0D(minmarginBIT);
		CLNDEL0D(marginEdgesCuda);
	}
	//This function sets the W_pq cost function, it accepts a pointer to funtion
	//the W_pq will be sent a- the volume as a double array, b- canny edges (if originaly set otherwise nullptr will be sent), c- p index, and d- q index
	template <class captype, class tcaptype, class flowtype>
	void AlphaPathMoves<captype, tcaptype, flowtype>::setWpqFunction(double(*in_getWpq)(int64_t, int64_t, const void * ), const void * wpq_extra_data)
	{
		if (called_functions.sm_wpq_func)
		{
			bgn_log << LogType::ERROR << "APM: setting w_pq function could be done only once\n" << end_log;
			throw("APM: setting w_pq function could be done only once");
		}
		this->wpq_extra_data = wpq_extra_data;
		this->func_getWpq = in_getWpq;
		called_functions.sm_wpq_func = true;
	}
	//This function set the minimum margin constrains for each label (which could be different). The function accepts the following
	//a- an array consiting of the minmum margins for each label (even if it is 0) 
	//b- the used method to compute/store minimum margins 
	template <class captype, class tcaptype, class flowtype>
	void AlphaPathMoves<captype, tcaptype, flowtype>::setMinimumMargins(const Array2D<uint32_t> * in_minmargins_radii, MinMarginMethod in_mm_method)
	{
		if (!called_functions.default_mode && called_functions.mm_set)
		{
			bgn_log << LogType::ERROR << "APM: min-margins could be called only once\n" << end_log;
			throw("APM: min-margins could be called only once");
		}
		for (uint32_t l = 0; this->mm_nh_shifts != nullptr && l < this->nlabels; ++l)
			CLNDEL0D(this->mm_nh_shifts[l]);
		CLNDEL1D(this->mm_nh_shifts);
		CLNDEL0D(minmarginIDX);
		CLNDEL0D(minmarginBIT);
		CLNDEL0D(marginEdgesCuda);

		mm_method = in_mm_method;
		mm_nh_shifts = new Array2D<int32_t>*[this->nlabels];
		double * tmpdbl = nullptr;
		uint32_t n_nhs, idx_largest_mm = -1; // label with the largest minmargin 
		uint32_t current_mm = 0;
		this->max_mm_r = 0;
		for (uint32_t t = 0; t < this->nlabels; ++t)
		{
			current_mm = 0;
			if (in_minmargins_radii != nullptr)
			{
				current_mm = in_minmargins_radii->data[t];
				if (in_minmargins_radii->data[t] > max_mm_r)
					max_mm_r = in_minmargins_radii->data[t];
			}
			if (dims[2] > 1)
			{
				mm_nh_shifts[t] = GirdNeighboursGenerator::getShifts(3, current_mm * 2 + 1, false, true, current_mm, n_nhs, tmpdbl);
				CLNDEL1D(tmpdbl);
				continue;
			}
			Array2D<int32_t> *lcl_shifts = GirdNeighboursGenerator::getShifts(2, current_mm * 2 + 1, false, true, current_mm, n_nhs, tmpdbl);
			CLNDEL1D(tmpdbl);
			mm_nh_shifts[t] = new Array2D<int32_t>();
			mm_nh_shifts[t]->allocate(3, lcl_shifts->Y);
			for (int32_t i = 0; i < lcl_shifts->Y; ++i)
			{
				mm_nh_shifts[t]->data[0 + i * 3] = lcl_shifts->data[0 + i * 2];
				mm_nh_shifts[t]->data[1 + i * 3] = lcl_shifts->data[1 + i * 2];
				mm_nh_shifts[t]->data[2 + i * 3] = 0;
			}
			CLNDEL0D(lcl_shifts);
		}

		if (mm_method == AlphaPathMoves::MinMarginMethod::IDX_BASED  && max_mm_r != 0)
		{
			this->max_n_mm_nhs = (2 * max_mm_r + 1)*(2 * max_mm_r + 1);
			if (this->dims[2] != 1) //no need to over estimate if the volume is 2D
				this->max_n_mm_nhs *= (2 * max_mm_r + 1);
			if (!child2parent)
			{
				bgn_log << LogType::ERROR << "APM: Tree structure must be set first before setting the min-margins.\n" << end_log;
				throw exception("APM: Tree structure must be set first before setting the min-margins.");
			}
				
			this->minmarginIDX = new MinMarginIDX(max_n_mm_nhs, ndspl, nlabels, child2parent);
		}
		else if ((mm_method == AlphaPathMoves::MinMarginMethod::BIT_BASED || this->mm_method == AlphaPathMoves::MinMarginMethod::CUDA_BASED) && max_mm_r != 0)
		{
			this->max_n_mm_nhs = (uint32_t)mm_nh_shifts[0]->Y;
			idx_largest_mm = 0;
			for (uint32_t t = 1; t < this->nlabels; ++t)
				if (this->max_n_mm_nhs < mm_nh_shifts[t]->Y)
				{
					this->max_n_mm_nhs = (uint32_t) mm_nh_shifts[t]->Y;
					idx_largest_mm = t;
				}

			this->minmarginBIT = new MinMarginBIT(max_n_mm_nhs, this->ndspl, this->max_mm_r, this->mm_nh_shifts[idx_largest_mm]);
		}

		if (this->mm_method == AlphaPathMoves::MinMarginMethod::CUDA_BASED && max_mm_r != 0)
		{
			marginEdgesCuda = new MarginEdgesCuda(ndspl, dims, child2parent, nlabels
				, paths, mm_nh_shifts, minmarginBIT->shift2gbitid, max_mm_r, this->minmarginBIT->edges.data
				, this->minmarginBIT->n_banks, this->minmarginBIT->n_usedbits);
		}
		if (!called_functions.default_mode)
			called_functions.mm_set = true;
	}

	//This function sets the data terms (-logl + infinity cost seeds+ forbidden labels), it accepts a 2D array of size n pixels x m labels
	template <class captype, class tcaptype, class flowtype>
	void AlphaPathMoves<captype, tcaptype, flowtype>::setDataTerms(const Array2D<tcaptype> * in_dataterms) // pixels x n_labels 
	{
		if (called_functions.dataterms)
		{
			bgn_log << LogType::ERROR << "APM: The dataterms could be set only once before a PathMove run\n" << end_log;
			throw("APM: The dataterms could be set only once before a PathMove run");
		}
		assert(in_dataterms != nullptr);
		assert(in_dataterms->X == dims[0] * dims[1] * dims[2]);
		assert(in_dataterms->Y == this->nlabels);
		if (dataterms)
			delete dataterms;
		dataterms = new Array2D<tcaptype>();
		dataterms->allocate(in_dataterms->X, in_dataterms->Y);
		memcpy(dataterms->data, in_dataterms->data, sizeof(tcaptype)*in_dataterms->totalsize);
		called_functions.dataterms = true;
	}
	template <class captype, class tcaptype, class flowtype>
	flowtype AlphaPathMoves<captype, tcaptype, flowtype>::computeEnergyForLabeling(uint32_t * t_labeling)
	{
		flowtype t1, t2;
		return computeEnergyForLabeling(t_labeling, t1, t2);
	}
	template <class captype, class tcaptype, class flowtype>
	flowtype AlphaPathMoves<captype, tcaptype, flowtype>::computeEnergyForLabeling(uint32_t * t_labeling, flowtype & d_tenergy, flowtype &s_tenergy)
	{
		flowtype tenergy = 0;
		d_tenergy = 0, s_tenergy = 0;
		for (int64_t tp = 0; tp < this->ndspl; ++tp)
			d_tenergy += this->dataterms->data[tp + t_labeling[tp] * this->ndspl];

		int64_t qx, qy, qz;
		int64_t p = 0, q, dims01 = dims[0] * dims[1];
		for (int64_t pz = 0; pz < this->dims[2]; ++pz)
			for (int64_t py = 0; py < this->dims[1]; ++py)
				for (int64_t px = 0; px < this->dims[0]; ++px)
				{
					for (uint32_t shift_id = 0; shift_id < this->sm_nh_shifts->totalsize; shift_id += 3)
					{
						qx = px + this->sm_nh_shifts->data[shift_id + 0]; if (qx < 0 || qx >= this->dims[0]) continue;
						qy = py + this->sm_nh_shifts->data[shift_id + 1]; if (qy < 0 || qy >= this->dims[1]) continue;
						qz = pz + this->sm_nh_shifts->data[shift_id + 2]; if (qz < 0 || qz >= this->dims[2]) continue;
						q = qx + qy*dims[0] + qz*dims01;
						if (t_labeling[p] == t_labeling[q])
							continue;
						s_tenergy += this->treemetric->data[t_labeling[p] + t_labeling[q] * this->nlabels] * this->func_getWpq(p, q,this->wpq_extra_data) * lambda;
					}
					p++;
				}
		tenergy = d_tenergy + s_tenergy;
		return tenergy;
	}

	template<class captype, class tcaptype, class flowtype>
	inline void AlphaPathMoves<captype, tcaptype, flowtype>::setExpansionOrdering(int32_t str_id, int32_t exp_id)
	{
		if (str_id < 0 || str_id >= this->nlabels || \
			exp_id < 0 || exp_id >= this->nlabels)
		{
			bgn_log << LogType::ERROR << "The following condition was violated: 0 <= label_id < n_labels\n" << end_log;
			throw("The following condition was violated: 0 <= label_id < n_labels");
		}
		this->expansionOrdering.chain_str_idx = str_id;
		this->expansionOrdering.chain_end_idx = exp_id;
		this->expansionOrdering.randomization_seed= 101;
		this->expansionOrdering.label_ordering.allocate(0, 0);
		expansionOrdering.type = ExpansionOrdering::ExpansionOrderingType::CHAIN;
	}

	template<class captype, class tcaptype, class flowtype>
	inline void AlphaPathMoves<captype, tcaptype, flowtype>::setExpansionOrdering(int32_t seed)
	{
		if (seed<0)
		{
			bgn_log << LogType::ERROR << "The seed should be a postive integer\n" << end_log;
			throw("The seed should be a postive integer");
		}
		this->expansionOrdering.chain_str_idx = -1;
		this->expansionOrdering.chain_end_idx = -1;
		this->expansionOrdering.randomization_seed = seed;
		this->expansionOrdering.label_ordering.allocate(0,0);
		expansionOrdering.type = ExpansionOrdering::ExpansionOrderingType::SEED;
	}

	template<class captype, class tcaptype, class flowtype>
	inline void AlphaPathMoves<captype, tcaptype, flowtype>::setExpansionOrdering(const Array2D<uint32_t> * label_ordering)
	{
		if (label_ordering->totalsize != this->nlabels)
		{
			bgn_log << LogType::ERROR << "The label ordering vector should be the same size as the number of labels\n" << end_log;
			throw("The label ordering vector should be the same size as the number of labels");
		}
		for (int32_t i = 0; i < this->nlabels; ++i)
		{
			bool found = false;
			for (int32_t j=0 ; j < this->nlabels;++j)
				if (label_ordering->data[j] == i)
				{
					found = true;
					break;
				}
			if (!found)
			{
				bgn_log << LogType::ERROR << "Label "<< i<< " is missing from expansion ordering\n" << end_log;
				throw("Label " + std::to_string(i) +" is missing from expansion ordering\n");
			}
		}

		this->expansionOrdering.chain_str_idx = -1;
		this->expansionOrdering.chain_end_idx = -1;
		this->expansionOrdering.randomization_seed = 101;
		this->expansionOrdering.label_ordering.allocate(this->nlabels, 1);
		std::memcpy(this->expansionOrdering.label_ordering.data, label_ordering->data, sizeof(uint32_t)*this->nlabels);
		expansionOrdering.type = ExpansionOrdering::ExpansionOrderingType::VECTOR;
	}

	//The function sets the initial labeling (if any)
	template <class captype, class tcaptype, class flowtype>
	void AlphaPathMoves<captype, tcaptype, flowtype>::setInitialLabeling(const Array3D<uint32_t> * in_labeling)
	{
		if (called_functions.init_labeling)
		{
			bgn_log << LogType::ERROR << "APM: The initial labeling could be called only once before a PathMove run\n" << end_log;
			throw("APM: The initial labeling could be called only once before a PathMove run");
		}
		memcpy(this->labeling, in_labeling->data, sizeof(uint32_t)*this->ndspl);
		this->called_functions.init_labeling = true;
	}
	
	//The function designates a specific label to be used for initial labeling (in precense of multiple labels with seeds avoid using this function)
	template <class captype, class tcaptype, class flowtype>
	void AlphaPathMoves<captype, tcaptype, flowtype>::setInitialLabeling(uint32_t lbl_id)
	{
		if (called_functions.init_labeling)
		{
			bgn_log << LogType::ERROR << "APM: The initial labeling could be called only once before a PathMove run\n" << end_log;
			throw("APM: The initial labeling could be called only once before a PathMove run");
		}
		std::fill_n(this->labeling, this->ndspl, lbl_id);
		this->called_functions.init_labeling = true;
	}

	//This function sets the neighbourhood window size where W=2*r+1 and r is the window radius.
	//For a pixel p, all the pixels in a WxW window centered around p will be considered its neighbours.
	template <class captype, class tcaptype, class flowtype>
	void AlphaPathMoves<captype, tcaptype, flowtype>::setSmoothnessNeighbourhoodWindowSize(uint32_t in_windowsize)
	{
		if (called_functions.sm_wsize)
		{
			bgn_log << LogType::ERROR << "APM: The smoothness neighbourhood window size could be called only once\n" << end_log;
			throw("APM: The smoothness neighbourhood window size could be called only once");
		}

		if (in_windowsize % 2 == 0)
		{
			bgn_log << LogType::ERROR << "Neighbourhood window size must be an odd number\n" << end_log;
			throw("Neighbourhood window size must be an odd number\n");
		}
		double * nh_shift_mag = nullptr;
		uint32_t n_shifts;
		sm_nh_shifts = GirdNeighboursGenerator::getShifts(3, in_windowsize, true, false, -1, n_shifts, nh_shift_mag);
		CLNDEL1D(nh_shift_mag);
		called_functions.sm_wsize = true;
	}

	//This function sets the Hedgehog Attributes
	//a- Hedgehog neighbourhood window size. Note that all the lables with active hedgehog prior will use the same window size.
	//b- a mask which indicates the seeds that will be used to generate the distance map and the vector field for each label with an active hedgehog prior.
	//   The passed mask is label map, i.e. mask[p]=i means that pixel p is part of label_i seed. mask[p]=-1 means that it is not part of any label's seed.
	//theta is in degrees and it always best to stay out of extreem values, like 0 and 90. For best perfomance theta should be 45+/-20.
	//c- theta : which is a parameter that controls how tight the hedgehog prior is. theta =0 means segmetnation must aligne with the level sets of the distance map.
	//   when theta = 90 and the seed is a single pixel then hedgeho prior reduces to star-shape prior.
	template <class captype, class tcaptype, class flowtype>
	void AlphaPathMoves<captype, tcaptype, flowtype>::setHedgehogAttributes(uint32_t hhog_windowsize \
		, const Array3D<int32_t> * mask \
		, double theta)
	{
		if (called_functions.hhog_set)
		{
			bgn_log << LogType::ERROR << "APM: Hedgehogs attributes could be called only once\n" << end_log;
			throw("APM: Hedgehogs attributes could be called only once");
		}

		if (hhog_windowsize != 0 && hhog_windowsize % 2 == 0)
		{
			bgn_log << LogType::ERROR << "Neighbourhood window size must be an odd number\n" << end_log;
			throw("Neighbourhood window size must be an odd number\n");
		}
		this->hhog_theta = theta;
		bgn_log << LogType::INFO_LEVEL1 << "APM: Computing Hedghog Constraints...\n" << end_log;
		clock_t begin_time = clock();
		this->hhog_radius = 0;
		hhogprior_status = false;
		if (hhog_windowsize == 0 || mask == nullptr)
		{
			bgn_log << LogType::INFO_LEVEL1 << "APM: Computing Hedghog Constraints done in " << std::to_string(float(clock() - begin_time) / CLOCKS_PER_SEC) << " secs\n" << end_log;
			return;
		}

		std::vector<int32_t> hhog_active_list;
		for (int64_t p = 0; p < mask->totalsize; p++)
		{
			if (mask->data[p] == -1)
				continue;
			if (find(hhog_active_list.begin(), hhog_active_list.end(), mask->data[p]) == hhog_active_list.end())
				hhog_active_list.push_back(mask->data[p]);
		}
		if (hhog_active_list.size() == 0)
		{
			bgn_log << LogType::INFO_LEVEL1 << "APM: Computing Hedghog Constraints done in " << std::to_string(float(clock() - begin_time) / CLOCKS_PER_SEC) << " secs\n" << end_log;
			return;
		}

		this->hhog_radius = (hhog_windowsize-1)/2;
		hhoglabel_status = std::vector<bool>(this->nlabels, false);

		double * tmp_shifts_mag = nullptr;
		uint32_t tmp_n_shifts;
		if (dims[2] == 1)
		{
			Array2D<int32_t>   *tmp_shifts;
			tmp_shifts = GirdNeighboursGenerator::getShifts(2, hhog_radius * 2 + 1, false, false, -1, tmp_n_shifts, tmp_shifts_mag);
			hhog_nh_shifts = new Array2D<int32_t>();
			hhog_nh_shifts->allocate(3, tmp_shifts->Y);
			for (uint32_t s = 0; s < tmp_shifts->Y; s++)
			{
				hhog_nh_shifts->data[s * 3 + 0] = tmp_shifts->data[s * 2 + 0];
				hhog_nh_shifts->data[s * 3 + 1] = tmp_shifts->data[s * 2 + 1];
				hhog_nh_shifts->data[s * 3 + 2] = 0;
			}
			delete tmp_shifts;
		}
		else  {
			hhog_nh_shifts = GirdNeighboursGenerator::getShifts(3, hhog_radius * 2 + 1, false, false, -1, tmp_n_shifts, tmp_shifts_mag);
		}
		CLNDEL1D(tmp_shifts_mag);
		
		for (uint32_t i = 0; i < hhog_active_list.size(); ++i)
		{
			Array3D<float> *  outside3D = DistanceTransform::EDT<int32_t, float>(mask, hhog_active_list[i], DistanceTransform::OUTSIDE | DistanceTransform::SQUARED);
			NDField<float>* revNormals = SobelFilter<float, float>(outside3D, true, true);
			hhogconstraints[hhog_active_list[i]] = getHedgehogConstraints <float, int32_t>(revNormals, hhog_nh_shifts, hhog_radius, theta);
			delete revNormals;
			delete outside3D;
			hhoglabel_status[hhog_active_list[i]] = true;
			hhogprior_status = true;
		}
		bgn_log << LogType::INFO_LEVEL1 << "APM: Computing Hedghog Constraints done in " << std::to_string(float(clock() - begin_time) / CLOCKS_PER_SEC) << " secs\n" << end_log;
		called_functions.hhog_set = true;
	}

	//This function sets the following:
	//a-lambda (normalization factor between unary and pairwise potentials) parameters
	//b-the structural directed tree. For n labels in_partialTreemetric will be an nxn matrix 
	//  in_partialTreemetric[i][j]=k means that j has i for a parent, and the cost of assiging two neary by pixels to i and j is k.
	//TODO:rename
	template <class captype, class tcaptype, class flowtype>
	void AlphaPathMoves<captype, tcaptype, flowtype>::setTreeWeights(double lambda, const Array2D<double> * in_partialTreemetric)
	{
		if (called_functions.sm_tree_lambda)
		{
			bgn_log << LogType::ERROR << "APM: Tree structure could set only once\n" << end_log;
			throw("APM: Tree structure could set only once");
		}

		this->lambda = lambda;

		assert(in_partialTreemetric->X == this->nlabels);
		assert(in_partialTreemetric->Y == this->nlabels);
		//populate child2parent 
		this->child2parent = new int32_t[this->nlabels];
		fill_n(this->child2parent, this->nlabels, -1);
		for (auto row_id = 0u; row_id < this->nlabels; ++row_id)
			for (auto col_id = 0u; col_id < this->nlabels; ++col_id)
				if (in_partialTreemetric->data[col_id + row_id*in_partialTreemetric->constX]>=0)
					child2parent[col_id] = row_id;

		//generate all possible paths between tree nodes
		max_nlayers = 0;
		paths = new Array3D<int32_t>();
		paths->allocate(this->nlabels + 2,this->nlabels, this->nlabels);
		paths->fill(-1);
		paths_lengths = new Array2D<int32_t>();
		paths_lengths->allocate(this->nlabels, this->nlabels);
		paths_lengths->fill(0);

		std::vector<uint32_t> path;
		std::queue<uint32_t> q;
		uint32_t path_idx;
		int32_t *pred = new int32_t[this->nlabels], current_node;
		bool * flag = new bool[this->nlabels];
		for (uint32_t s = 0; s <this->nlabels; s++)
		{
			fill_n(pred, this->nlabels, -1);
			fill_n(flag, this->nlabels, false);
			q.push(s);
			flag[s] = true;
			while (!q.empty())
			{
				int32_t v = q.front(); q.pop();
				for (uint32_t k = 0; k <this->nlabels; ++k)
					if ((in_partialTreemetric->data[k + v*in_partialTreemetric->constX]>=0 || in_partialTreemetric->data[v + k*in_partialTreemetric->constX]>=0) && flag[k] == false)
					{
						flag[k] = true;
						pred[k] = v;
						q.push(k);
					}
			}
			for (uint32_t e = 0; e < this->nlabels; ++e)
			{
				current_node = e;
				path_idx = 0;
				do
				{
					paths->data[path_idx + s*paths->constX + e*paths->constXY] = current_node;
					path_idx++;
					current_node = pred[current_node];
				} while (current_node != -1);
				if (max_nlayers < path_idx)
					max_nlayers = path_idx;
				paths_lengths->data[s + paths_lengths->constX*e] = path_idx;
			}
		}
		delete[] pred;
		delete[] flag;

		/* populate full tree metric */
		treemetric = new Array2D<double>();
		treemetric->allocate(this->nlabels, this->nlabels);

		int32_t curr, nxt,c,rc,ntail,nhead;
		for (uint32_t s = 0; s <this->nlabels; s++)
		{
			for (uint32_t e = 0; e < this->nlabels; ++e)
			{
				curr = 0; nxt = 1;
				treemetric->data[e + s*treemetric->constX] = 0;
				while (paths->data[nxt + s*paths->constX + e*paths->constXY] != -1)
				{
					ntail = (paths->data[curr + e*paths->constX + s*paths->constXY]);
					nhead = (paths->data[nxt + e*paths->constX + s*paths->constXY]);
					c = in_partialTreemetric->data[ntail + nhead*in_partialTreemetric->constX];
					rc = in_partialTreemetric->data[nhead + ntail*in_partialTreemetric->constX];
					if (c == -1 && rc >= 0) 
					{
						treemetric->data[e + s*treemetric->constX] += rc;
					}
					else if (c >= 0 && rc == -1) 
					{
						treemetric->data[e + s*treemetric->constX] += c;
					} 
					else 
					{
						bgn_log << LogType::ERROR << "APM: A cycle was detected in the structural tree\n" << end_log;
						throw exception("APM: A cycle was detected in the structural tree");

					}
					curr++;	nxt++;
				}
			}
		}
		called_functions.sm_tree_lambda = true;
	}
	template <class captype, class tcaptype, class flowtype>
	void AlphaPathMoves<captype, tcaptype, flowtype>::setupMaxflowSolver(typename MaxflowSolver<captype, tcaptype, flowtype>::SolverName  maxflow_solvername)
	{
		int32_t n_active_hhogs=0;
		for (int32_t i = 0; i < this->hhoglabel_status.size(); ++i)
			if (this->hhoglabel_status[i])
				n_active_hhogs++;
		
		//find ns in hhog but not in smoothness 
		int32_t hhog_exp_smoothnes_nhs = 0;
		if (this->hhogprior_status)
		{
			for (uint32_t hhog_shift_id = 0; hhog_shift_id < this->hhog_nh_shifts->totalsize; hhog_shift_id += 3)
			{
				bool count = true;
				for (uint32_t sm_shift_id = 0; sm_shift_id < this->sm_nh_shifts->totalsize; sm_shift_id += 3)
					if ((this->hhog_nh_shifts->data[hhog_shift_id + 0] == this->sm_nh_shifts->data[sm_shift_id + 0]) &&
						(this->hhog_nh_shifts->data[hhog_shift_id + 1] == this->sm_nh_shifts->data[sm_shift_id + 1]) &&
						(this->hhog_nh_shifts->data[hhog_shift_id + 2] == this->sm_nh_shifts->data[sm_shift_id + 2]))
					{
						count = false;
						break;
					}
				if (count)
					hhog_exp_smoothnes_nhs++;
			}
		}
		gc_nds = this->ndspl*this->max_nlayers;
		uint64_t max_n_smoothness_edges = this->sm_nh_shifts->Y*gc_nds;
		uint64_t max_n_hedgehog_edges = 0;
		if (this->hhogprior_status)
			max_n_hedgehog_edges = hhog_exp_smoothnes_nhs*this->ndspl*n_active_hhogs;
		int64_t n_pariwise_edges = max_n_smoothness_edges+max_n_hedgehog_edges;

		uint64_t max_n_ishkawa_edges = this->ndspl*(this->max_nlayers - 1);
		uint64_t max_n_mm_edges = this->ndspl*this->max_n_mm_nhs*(this->max_nlayers - 1);
		clock_t begin_time;
		if (!maxflow_solver)
		{
			begin_time = clock();
			bgn_log << LogType::INFO_LEVEL2 << "APM: Initializing Maxflow_solver Memory ...\n" << end_log;
			maxflow_solver = new MaxflowSolver<captype, tcaptype, flowtype>(gc_nds, n_pariwise_edges + max_n_ishkawa_edges + max_n_mm_edges, maxflow_solvername);
			bgn_log << LogType::INFO_LEVEL2 << "APM: Initializing Maxflow_solver Memory done in " << float(clock() - begin_time) / CLOCKS_PER_SEC << " secs\n" << end_log;
		}
			
		if (!graph)
		{
			begin_time = clock();
			bgn_log << LogType::INFO_LEVEL2 << "APM: Allocating Proxy Graph Memory ...\n" << end_log;
			graph = new BasicGraph<captype, tcaptype>(gc_nds, gc_nds * 2, n_pariwise_edges + max_n_ishkawa_edges + max_n_mm_edges);
			bgn_log << LogType::INFO_LEVEL2 << "APM: Allocating Proxy Graph Memory " << graph->getAllocatedMemSizeInGB() << "Gb done in " << float(clock() - begin_time) / CLOCKS_PER_SEC << " secs\n" << end_log;
		}
		else
		{
			begin_time = clock();
			bgn_log << LogType::INFO_LEVEL2 << "APM: Resetting Proxy Graph Memory ...\n" << end_log;
			graph->reset();
			bgn_log << LogType::INFO_LEVEL2 << "APM: Resetting Proxy Graph Memory done in " << float(clock() - begin_time) / CLOCKS_PER_SEC<< " secs\n" << end_log;
		}
			
	}
	template <class captype, class tcaptype, class flowtype>
	void AlphaPathMoves<captype, tcaptype, flowtype>::setHedgehogGCArcs(Array3D<char> *pm_hhogconstraints)
	{
		if (!this->hhogprior_status) return;

		int64_t p = 0, q, dims01 = dims[0] * dims[1], qidx, pidx;
		int64_t hhog_2r1 = (2 * this->hhog_radius + 1), hhog_2r1_sq = hhog_2r1*hhog_2r1;
		int64_t countZ = dims[2]>1 ? 1 : 0;
		int64_t conv_idx = getConvIdx(this->hhog_radius, hhog_2r1, countZ);
		flowtype wpq, wqp;
		int64_t qx, qy, qz; //must remain unsigned to check for boundries
		for (int64_t pz = 0; pz < this->dims[2]; ++pz)
			for (int64_t py = 0; py < this->dims[1]; ++py)
				for (int64_t px = 0; px < this->dims[0]; ++px)
				{
					for (uint32_t shift_id = 0; shift_id < this->hhog_nh_shifts->totalsize; shift_id += 3)
					{
						qx = px + this->hhog_nh_shifts->data[shift_id + 0]; if (qx < 0 || qx >= this->dims[0]) continue;
						qy = py + this->hhog_nh_shifts->data[shift_id + 1]; if (qy < 0 || qy >= this->dims[1]) continue;
						qz = pz + this->hhog_nh_shifts->data[shift_id + 2]; if (qz < 0 || qz >= this->dims[2]) continue;
						q = qx + qy*dims[0] + qz*dims01;
						qidx=getqidx(this->hhog_nh_shifts->data[shift_id + 0], this->hhog_nh_shifts->data[shift_id + 1], this->hhog_nh_shifts->data[shift_id + 2], this->hhog_radius, hhog_2r1, hhog_2r1_sq, countZ);
						pidx = conv_idx - qidx;

						for (uint32_t layer_id = 0; layer_id < this->max_nlayers; layer_id++)
						{
							wpq = 0;
							wqp = 0;
							if (pm_hhogconstraints->data[layer_id*pm_hhogconstraints->constXY + p*pm_hhogconstraints->constX + qidx] == 1)
							{
								wpq = LRG_PENALTY;
								pm_hhogconstraints->data[layer_id*pm_hhogconstraints->constXY + p*pm_hhogconstraints->constX + qidx] = 0;
							}
							if (pm_hhogconstraints->data[layer_id*pm_hhogconstraints->constXY + q*pm_hhogconstraints->constX + pidx] == 1)
							{
								wqp = LRG_PENALTY;
								pm_hhogconstraints->data[layer_id*pm_hhogconstraints->constXY + q*pm_hhogconstraints->constX + pidx] = 0;
							}
							if (wpq == 0 && wqp == 0) continue;
							graph->add_nlink(p + this->ndspl*layer_id, q + this->ndspl*layer_id, (captype)wpq, (captype)wqp);
						}
					}
					p++;
				}
	}
	template <class captype, class tcaptype, class flowtype>
	void AlphaPathMoves<captype, tcaptype, flowtype>::setPairwiseSmoothnessGCArcs(Array3D<char> * pm_hhogconstraints, flowtype *exp_dataterms, captype  *lcl_pairwise, int64_t * nlinks_ids, 
		tcaptype *lcl_unary, int64_t * tlinks_ids, uint32_t explbl, uint32_t current_cycle)
	{
		int64_t pcounter, ucounter, layer_id;

		//Compute Smoohtness Edges [passed unit testing]
		int64_t p = 0, q, dims01 = dims[0] * dims[1], qidx, pidx;
		int64_t hhog_2r1 = (2 * this->hhog_radius + 1), hhog_2r1_sq = hhog_2r1*hhog_2r1;
		int64_t countZ = dims[2]>1 ? 1 : 0;
		int64_t conv_idx = getConvIdx(this->hhog_radius, hhog_2r1, countZ);
		int32_t * P, *Q;
		

		double wpq, wqp;
		int64_t qx, qy, qz; //must remain unsigned to check for boundries
		for (int64_t pz = 0; pz < this->dims[2]; ++pz)
			for (int64_t py = 0; py < this->dims[1]; ++py)
				for (int64_t px = 0; px < this->dims[0]; ++px)
				{
					for (uint32_t shift_id = 0; shift_id < this->sm_nh_shifts->totalsize; shift_id += 3)
					{
						qx = px + this->sm_nh_shifts->data[shift_id + 0]; if (qx < 0 || qx >= this->dims[0]) continue;
						qy = py + this->sm_nh_shifts->data[shift_id + 1]; if (qy < 0 || qy >= this->dims[1]) continue;
						qz = pz + this->sm_nh_shifts->data[shift_id + 2]; if (qz < 0 || qz >= this->dims[2]) continue;
						q = qx + qy*dims[0] + qz*dims01;
						fillPairewisePathSmoothnessArcs(p, q, this->labeling[p], this->labeling[q], explbl, lcl_pairwise, nlinks_ids, pcounter, lcl_unary,tlinks_ids, ucounter);

						P = this->paths->data + this->labeling[p] * paths->constX + explbl*paths->constXY;
						Q = this->paths->data + this->labeling[q] * paths->constX + explbl*paths->constXY;

						//set the pariwise edges directly into GC and accumulate the smoothness2unary potentials
						for (uint32_t tk = 0, path_idx=0; tk < pcounter * 2; tk += 2, path_idx++)
						{
							//we know if where are visitng P[path_idx] and Q[path_idx] in this loop then they are equal other wise a pair-wise edge would not have been created
							if (!this->hhogprior_status)
							{
								graph->add_nlink(nlinks_ids[tk], nlinks_ids[tk+1], lcl_pairwise[tk], lcl_pairwise[tk + 1]);
								continue;
							}
							layer_id = nlinks_ids[tk] / this->ndspl;
							wpq = lcl_pairwise[tk];
							wqp = lcl_pairwise[tk + 1];
							qidx=getqidx(this->sm_nh_shifts->data[shift_id + 0], this->sm_nh_shifts->data[shift_id + 1], this->sm_nh_shifts->data[shift_id + 2], this->hhog_radius, hhog_2r1, hhog_2r1_sq, countZ);
							pidx = conv_idx - qidx;
							if (pm_hhogconstraints->data[layer_id*pm_hhogconstraints->constXY + p*pm_hhogconstraints->constX + qidx] == 1)
							{
								wpq = LRG_PENALTY;
								pm_hhogconstraints->data[layer_id*pm_hhogconstraints->constXY + p*pm_hhogconstraints->constX + qidx] = 0;
							}
							if (pm_hhogconstraints->data[layer_id*pm_hhogconstraints->constXY + q*pm_hhogconstraints->constX + pidx] == 1)
							{
								wqp = LRG_PENALTY;
								pm_hhogconstraints->data[layer_id*pm_hhogconstraints->constXY + q*pm_hhogconstraints->constX + pidx] = 0;
							}
							graph->add_nlink(nlinks_ids[tk], nlinks_ids[tk + 1], wpq, wqp);

						}

						//add auxillary smoothness2unary terms to the the dataterms 
						for (uint32_t i = 0; i < ucounter; i++)
							exp_dataterms[tlinks_ids[i]] += lcl_unary[i];

					}
					p++;
				}
	}
	template <class captype, class tcaptype, class flowtype>
	void AlphaPathMoves<captype, tcaptype, flowtype>::fillPathMoveHedgehogConstraints(Array3D<char> * pm_hhogconstraints, uint32_t explbl, uint32_t current_cycle)
	{
		if (!this->hhogprior_status)
			return;
		memset(pm_hhogconstraints->data, 0, sizeof(char)*pm_hhogconstraints->totalsize);
		
		//initial implementation of populateGCHedgehogConstraints is going to be C++ (initially)
		//Compute Smoohtness Edges [passed unit testing]
		int64_t p = 0, q, dims01 = dims[0] * dims[1], qidx, pidx;
		int64_t hhog_2r1 = (2 * this->hhog_radius + 1), hhog_2r1_sq = hhog_2r1*hhog_2r1;
		int64_t countZ = dims[2]>1 ? 1 : 0;
		int64_t conv_idx = 2 * this->hhog_radius + 2 * this->hhog_radius * hhog_2r1 + 2 * this->hhog_radius* pow2(hhog_2r1) * countZ;
		int32_t * P, *Q;
		int32_t A, B, X, Y;
		int64_t wpq, wqp;
		int64_t qx, qy, qz; //must remain unsigned to check for boundries
		for (int64_t pz = 0; pz < this->dims[2]; ++pz)
			for (int64_t py = 0; py < this->dims[1]; ++py)
				for (int64_t px = 0; px < this->dims[0]; ++px)
				{
					for (uint32_t shift_id = 0; shift_id < this->hhog_nh_shifts->totalsize; shift_id += 3)
					{
						qx = px + this->hhog_nh_shifts->data[shift_id + 0]; if (qx < 0 || qx >= this->dims[0]) continue;
						qy = py + this->hhog_nh_shifts->data[shift_id + 1]; if (qy < 0 || qy >= this->dims[1]) continue;
						qz = pz + this->hhog_nh_shifts->data[shift_id + 2]; if (qz < 0 || qz >= this->dims[2]) continue;
						q = qx + qy*dims[0] + qz*dims01;
						P = this->paths->data + this->labeling[p] * paths->constX + explbl*paths->constXY;
						Q = this->paths->data + this->labeling[q] * paths->constX + explbl*paths->constXY;
						qidx = getqidx(this->hhog_nh_shifts->data[shift_id + 0], this->hhog_nh_shifts->data[shift_id + 1], this->hhog_nh_shifts->data[shift_id + 2], this->hhog_radius, hhog_2r1, hhog_2r1_sq, countZ);
						pidx = conv_idx - qidx;
						for (uint32_t layer_id = 0; layer_id<= this->max_nlayers; ++layer_id)
						{
							wpq = 0;
							wqp = 0;
							if (layer_id + 1 <= this->max_nlayers && Q[layer_id + 1] == -1 && P[layer_id + 1] == -1)
								break;
							if (layer_id == 0 && Q[layer_id + 1] != -1 && P[layer_id + 1] != -1)
							{
								//scenario 8
								A = P[layer_id];
								X = P[layer_id+1];
								Y = Q[layer_id + 1];
								if (this->hhoglabel_status[A] && X == Y  && this->hhogconstraints[A]->data[p*hhogconstraints[A]->X + qidx] == 1) //senario 8, case 2
									pm_hhogconstraints->data[layer_id*pm_hhogconstraints->constXY + p*pm_hhogconstraints->constX + qidx] = 1;
								
							}
							if (P[layer_id] != -1 && P[layer_id] == Q[layer_id] &&
								(layer_id + 1) <= this->max_nlayers && P[layer_id + 1] != -1 && P[layer_id + 1] == Q[layer_id + 1] &&
								(layer_id + 2) <= this->max_nlayers && (P[layer_id + 2] == -1 || Q[layer_id + 2] == -1))
							{
								if (P[layer_id + 2] == -1 && Q[layer_id + 2] == -1) //senario 5
								{
									B = P[layer_id];
									A = P[layer_id + 1];
									if (this->hhoglabel_status[A] && this->hhogconstraints[A]->data[p*hhogconstraints[A]->X + qidx] == 1)
									{
										if (this->child2parent[A] == B) //senario 5, case 1 (A in B)
											pm_hhogconstraints->data[layer_id*pm_hhogconstraints->constXY + q*pm_hhogconstraints->constX + pidx] = 1;
										else if (this->child2parent[B] == A) //senario 5, case 2 (B in A)
											; //nothing to be done in this case
										else
											bgn_log << LogType::ERROR << "APM: Path with invalid parent-child relationship encountered\n" << end_log;
											
									}

								} else { //senario 2 and (2 sym)
									B = P[layer_id];
									A = P[layer_id + 1];
									Y = P[layer_id + 2];
									X = Q[layer_id + 2];
									if (this->hhoglabel_status[A] && this->hhogconstraints[A]->data[p*hhogconstraints[A]->X + qidx] == 1)
									{
										if (Y == -1) {  //scenario 2 case 1
											if (this->child2parent[X] == A && this->child2parent[A] == B)
												pm_hhogconstraints->data[layer_id*pm_hhogconstraints->constXY + q*pm_hhogconstraints->constX + pidx] = 1;
										}
										else if (X == -1) { //scenario 2 (sym) case 1
											if (this->child2parent[Y] == A && this->child2parent[A] == B)
												pm_hhogconstraints->data[layer_id*pm_hhogconstraints->constXY + q*pm_hhogconstraints->constX + pidx] = 1;
										}
										else {
											bgn_log << LogType::ERROR << "APM: Path with invalid hedgehog scenario reached\n"<<end_log;
										}
									}
								}
							}
							if (P[layer_id] != -1 && P[layer_id] == Q[layer_id] &&
								(layer_id + 1) <= this->max_nlayers && P[layer_id + 1] != -1 && P[layer_id + 1] == Q[layer_id + 1] &&
								(layer_id + 2) <= this->max_nlayers && P[layer_id + 2] != -1 && Q[layer_id + 2] != -1)
							{
								//senario 1
								B = P[layer_id];
								A = P[layer_id + 1];
								Y = P[layer_id + 2];
								X = Q[layer_id + 2];
								if (this->hhoglabel_status[A] && this->hhogconstraints[A]->data[p*hhogconstraints[A]->X + qidx] == 1)
								{
									if (this->child2parent[X] == A && this->child2parent[A] == B && this->child2parent[Y] == A) // scenario 1, case 1
										pm_hhogconstraints->data[layer_id*pm_hhogconstraints->constXY + q*pm_hhogconstraints->constX + pidx] = 1;
									else if (this->child2parent[B] == A && this->child2parent[A] == X && X == Y) // scenario 1, case 2
										pm_hhogconstraints->data[(layer_id+1)*pm_hhogconstraints->constXY + p*pm_hhogconstraints->constX + qidx] = 1;
								}
							}
						}
					}
					p++;
				}
	}
	
	template <class captype, class tcaptype, class flowtype>
	bool AlphaPathMoves<captype, tcaptype, flowtype>::ValidateLabeling(char * labeling, BasicGraph<captype, tcaptype> * graph)
	{
		BasicGraph<captype, tcaptype>::t_links_struct tlinks = graph->get_tlinks();
		BasicGraph<captype, tcaptype>::n_links_struct nlinks = graph->get_nlinks();
		uint64_t n_nodes = graph->getNumberOfNodes();
		int64_t dim0 = this->dims[0];
		int64_t dim01 = this->dims[0] * this->dims[1];
		int32_t s = 0, t = 1;
		int32_t p_layer_id = 0, q_layer_id = 0;
		bool infcost_severed = false;
		int64_t p, q, px, py, pz, qx, qy, qz;
		for (uint64_t link_id = 0; link_id < tlinks.n_tlinks; ++link_id)
		{
			p = tlinks.node_ids[link_id] % this->ndspl;
			p_layer_id = (tlinks.node_ids[link_id]-p) / this->ndspl;
					
			if (this->dims[2] == 1)
			{
				//p = px + py*dims0;
				pz = 1;
				py = p / dim0;
				px = p - py*dim0;
			}
			else
			{
				//p = px + py*dims0 + pz*dim01;
				pz = p / dim01;
				py = (p - pz*dim01) / dim0;
				px = p - py*dim0 - pz*dim01;
				
			}

			if (labeling[tlinks.node_ids[link_id]] == t && tlinks.weights[link_id * 2 + 0] >= LRG_PENALTY)
			{
				infcost_severed = true;
				bgn_log << LogType::WARNING << "ARM: Infinity cost s-link was severedn\n" <<end_log;

			}
			if (labeling[tlinks.node_ids[link_id]] != t && tlinks.weights[link_id * 2 + 1] >= LRG_PENALTY)
			{
				infcost_severed = true;
				bgn_log << LogType::WARNING << "ARM: Infinity cost s-link was severedn\n" << end_log;
			}
		}
		for (uint64_t offset = 0; offset < (nlinks.n_nlinks * 2); offset += 2)
		{

			p = nlinks.edge_nodes[offset + 0] % this->ndspl;
			p_layer_id = (nlinks.edge_nodes[offset + 0] - p) / this->ndspl;

			q = nlinks.edge_nodes[offset + 1] % this->ndspl;
			q_layer_id = (nlinks.edge_nodes[offset + 1] - q) / this->ndspl;

			if (this->dims[2] == 1)
			{
				//p = px + py*dims0;
				pz = 1;
				py = p / dim0;
				px = p - py*dim0;

				qz = 1;
				qy = q / dim0;
				qx = q - qy*dim0;
			}
			else
			{
				//p = px + py*dims0 + pz*dim01;
				pz = p / dim01;
				py = (p - pz*dim01) / dim0;
				px = p - py*dim0 - pz*dim01;
				
				qz = q / dim01;
				qy = (q - qz*dim01) / dim0;
				qx = q - qy*dim0 - qz*dim01;

			}

			if (labeling[nlinks.edge_nodes[offset + 0]] == s && labeling[nlinks.edge_nodes[offset + 1]] == t && nlinks.weights[offset + 0] >= LRG_PENALTY)
			{
				infcost_severed = true;
				bgn_log << LogType::WARNING << "ARM: Infinity cost n-link was severedn\n" << end_log;
			}
				
			if (labeling[nlinks.edge_nodes[offset + 0]] == t && labeling[nlinks.edge_nodes[offset + 1]] == s && nlinks.weights[offset + 1] >= LRG_PENALTY)
			{
				infcost_severed = true;
				bgn_log << LogType::WARNING << "ARM: Infinity cost n-link was severedn\n" << end_log;
			}
		}
		return infcost_severed;
	}
	//Call this function to run path-moves. The function accepts the following pameters 
	//a-  the maxflow solver to be used (IBFS is the fastest so far)
	//b-  (a return value) final energy which is the enegy at convergence
template <class captype, class tcaptype, class flowtype>
Array3D<uint32_t> * AlphaPathMoves<captype, tcaptype, flowtype>::runPathMoves(typename MaxflowSolver<captype, tcaptype, flowtype>::SolverName maxflow_solvername, flowtype & final_energy)
{
	struct RNG {
		int32_t operator() (int32_t n) {
			return std::rand() / (1.0 + RAND_MAX) * n;
		}
	};

	bgn_log << LogType::INFO_LEVEL1 << "ARM: Validating Status\n" << end_log;
	if (!called_functions.dataterms)
	{
		bgn_log << LogType::ERROR << "ARM: The dataterms must be set (or reset) before running PathMoves\n" << end_log;
		throw("ARM: The dataterms must before running PathMoves\n");
	}
	if (!called_functions.sm_wsize)
	{
		bgn_log << LogType::ERROR << "ARM: The smoohtness window size must be set before running PathMoves\n" << end_log;
		throw("ARM: The smoohtness window size must be set before running PathMoves");
	}
	if (!called_functions.sm_tree_lambda)
	{
		bgn_log << LogType::ERROR << "ARM: The hierarchal tree must be set before running PathMoves\n" << end_log;
		throw("ARM: The hierarchal tree must be set before running PathMoves");
	}
	if (!called_functions.init_labeling)
	{
		bgn_log << LogType::ERROR << "ARM: The initlal labling must be set before running PathMoves\n" << end_log;
		throw("ARM: The initlal labling must be set before running PathMoves");
	}




	bgn_log << LogType::INFO_LEVEL1 << "ARM: Allocating Memory\n" << end_log; 
	clock_t begin_time = clock();
	setupMaxflowSolver(maxflow_solvername);
	bgn_log << LogType::INFO_LEVEL1 << "APM: Allocating Memory done in " << double(clock() - begin_time) / CLOCKS_PER_SEC << " secs\n" << end_log;
	

	
	char * solver_labeling = new char[this->gc_nds];
	flowtype *exp_dataterms = new flowtype[this->max_nlayers*this->ndspl];
	int64_t * nlinks_ids = new int64_t[this->max_nlayers * 3 * 2];//p q 
	captype  *lcl_pairwise = new captype[this->max_nlayers * 3 * 2];  // wpq wqp		
	int64_t * tlinks_ids = new int64_t[this->max_nlayers * 3];//p 
	tcaptype *lcl_unary = new tcaptype[this->max_nlayers * 3];  //wsp

	flowtype tmp_energy = 0, const_term = 0;;
	uint32_t * tmp_labeling = new uint32_t[this->ndspl];


	int32_t *P, Pi, expansion_id = 1, cycleid = 1, best_expansionlabel;
	int64_t dims01 = dims[0] * dims[1];
	
	bool change_occured;
	bool * dirty_list = new bool[this->nlabels];
	std::fill_n(dirty_list, this->nlabels, true);
	auto IsAnyTrue_Lambda = [](bool *state_array, uint32_t & size) ->bool { for (uint32_t i = 0; i < size; ++i)  if (state_array[i]) return true;  return false; };
	Array3D<char> * pm_hhogconstraints = nullptr;
	if (this->hhogprior_status)
	{
		pm_hhogconstraints = new Array3D<char>();
		if (dims[2] == 1)
			pm_hhogconstraints->allocate(pow2(2 * this->hhog_radius + 1), this->ndspl, this->max_nlayers);
		else
			pm_hhogconstraints->allocate(pow3(2 * this->hhog_radius + 1), this->ndspl, this->max_nlayers);
	}

	flowtype min_value = this->LRG_PENALTY*1000;
	for (int64_t p = 0; p < this->dataterms->totalsize; ++p)
		if (min_value > this->dataterms->data[p])
			min_value = this->dataterms->data[p];

	flowtype offset = 0;
	if (min_value < 0)
	{
		offset = min_value * this->dataterms->X;
		for (int64_t p = 0; p < this->dataterms->totalsize; ++p)
			this->dataterms->data[p] -= min_value;
	}
	flowtype energy = this->computeEnergyForLabeling(labeling) + offset;
	final_energy = energy;

	//expanding on the leafs only is not as good as expanding on all the labels
	std::vector<uint32_t> label_ordering;
	if (expansionOrdering.type == ExpansionOrdering::ExpansionOrderingType::SEED)
	{
		std::srand(expansionOrdering.randomization_seed);
		for (uint32_t i = 0; i < this->nlabels; ++i)
			label_ordering.push_back(i);
		std::random_shuffle(label_ordering.begin(), label_ordering.end(), RNG());
	}
	else if  (expansionOrdering.type == ExpansionOrdering::ExpansionOrderingType::CHAIN)
	{
		this->setInitialLabeling((uint32_t)expansionOrdering.chain_str_idx);
		label_ordering.clear();
		label_ordering.push_back(expansionOrdering.chain_end_idx);
	}
	else if (expansionOrdering.type == ExpansionOrdering::ExpansionOrderingType::VECTOR)
	{
		for (uint32_t i = 0; i < expansionOrdering.label_ordering.totalsize; ++i)
			label_ordering.push_back(expansionOrdering.label_ordering.data[i]);
	}
	else
	{
		bgn_log << LogType::ERROR << "Unsupported expansion ordering\n" << end_log;
		throw("Unsupported expansion ordering\n");
	}
	

	do
	{
		
		change_occured = false;
		bgn_log << LogType::INFO_LEVEL1 << "APM: Cycle #" << cycleid << " current energy = " << energy << "\n" << end_log;
		for (uint32_t explidx = 0; explidx < label_ordering.size(); ++explidx)//this->nlabels
		{
			if (!IsAnyTrue_Lambda(dirty_list, this->nlabels)) break;
			bgn_log << LogType::INFO_LEVEL1 << "APM: \tExpansion #" << (int64_t)expansion_id << " on L" << label_ordering[explidx] << "\n" << end_log;

			//Compute Dataterms via telescopic sum
			const_term = 0; // actual energy = maxflow+ const_dataterm (s->t edges remaining from the telescopic sum of dataterms)
			std::fill_n(exp_dataterms, this->gc_nds, 0);
			//Set dataterms and ishakaw backward inf edges
			for (int64_t p = 0; p < this->ndspl; ++p) 
			{
				P = this->paths->data + labeling[p] * paths->constX + label_ordering[explidx] * paths->constXY;
				if (P[1] == -1) // 
				{
					graph->add_tlink(p, (captype)LRG_PENALTY, this->dataterms->data[p + this->ndspl*P[0]]);
					continue;
				}
				if (P[2] == -1)
				{
					graph->add_tlink(p, this->dataterms->data[p + this->ndspl*P[1]], this->dataterms->data[p + this->ndspl*P[0]]);
					continue;
				}
				graph->add_tlink(p, 0, this->dataterms->data[p + this->ndspl*P[0]]);
				for (Pi = 1; P[Pi+1] != -1; Pi++) 
					graph->add_nlink(p + this->ndspl*(Pi - 1), p + this->ndspl*Pi, (captype)LRG_PENALTY, this->dataterms->data[p + this->ndspl*P[Pi]]);
				graph->add_tlink(p + this->ndspl*(Pi - 1), this->dataterms->data[p + this->ndspl*P[Pi]], 0);
			}

			//Compute Smoohtness Edges [passed unit testing]
			clock_t btime = clock();
			bgn_log << LogType::INFO_LEVEL1 << "APM: \t\tPopulating HHogs constraints...\n" << end_log;
			if (this->hhogprior_status)
				fillPathMoveHedgehogConstraints(pm_hhogconstraints, label_ordering[explidx], expansion_id);
			bgn_log << LogType::INFO_LEVEL1 << "APM: \t\tPopulating HHogs constraints done in " <<double(clock() - btime) / CLOCKS_PER_SEC << " secs\n" << end_log;
			

			btime = clock();
			bgn_log << LogType::INFO_LEVEL1 << "APM: \t\tSetting pairwise smoothness arcs...\n" << end_log;
			setPairwiseSmoothnessGCArcs(pm_hhogconstraints,exp_dataterms, lcl_pairwise, nlinks_ids,lcl_unary,tlinks_ids,label_ordering[explidx], expansion_id);
			bgn_log << LogType::INFO_LEVEL1 << "APM: \t\tSetting pairwise smoothness arcs done in " <<  double(clock() - btime) / CLOCKS_PER_SEC << " secs\n" << end_log;

			
			bgn_log << LogType::INFO_LEVEL1 << "APM: \t\tSetting hedgehog arcs...\n" << end_log;
			setHedgehogGCArcs(pm_hhogconstraints);
			bgn_log << LogType::INFO_LEVEL1 << "APM: \t\tSetting hedgehog arcs done in " << double(clock() - btime) / CLOCKS_PER_SEC << " secs\n" << end_log;

			//Set the t-links
			for (auto p = this->ndspl*0; p < this->ndspl; ++p)
			{
				P = this->paths->data + labeling[p] * paths->constX + label_ordering[explidx] * paths->constXY;
				for( Pi=1; (P[Pi] != -1 || ((Pi == 1) && P[0] == label_ordering[explidx])) ; Pi++)
					this->graph->add_tlink(p + this->ndspl*(Pi - 1), exp_dataterms[p + this->ndspl*(Pi - 1)],0);
			}

			////Set min margin constraints
			bgn_log << LogType::INFO_LEVEL1 << "APM: \t\tSetting margin constraints...\n" << end_log;
			clock_t begin_time = clock();
			if (max_mm_r != 0)
				this->setMinMarginGCArcs(label_ordering[explidx], expansion_id);
			bgn_log << LogType::INFO_LEVEL1 << "APM: \t\tSetting margin constraints done in " << double(clock() - begin_time) / CLOCKS_PER_SEC << " secs\n" << end_log;
			

			//compute max-flow
			bgn_log << LogType::INFO_LEVEL1 << "APM: \t\tRunning maxflow...\n" << end_log;
			begin_time = clock();

			flowtype r_energy = maxflow_solver->solve(graph, solver_labeling);
			if (ValidateLabeling(solver_labeling, graph))
				bgn_log << LogType::WARNING << "APM:WARNING: infinity cost edge was SEVERED!!!!\n" << end_log;

			tmp_energy = r_energy + offset; // +const_term;
			bgn_log << LogType::INFO_LEVEL1 << "APM: \t\t\tmaxflow energy = " << tmp_energy << " found in " <<float(clock() - begin_time) / CLOCKS_PER_SEC << " secs \n" << end_log;
			bgn_log << LogType::INFO_LEVEL1 << "APM: \t\t\tenergy / infinity_cost_edge = " << tmp_energy / LRG_PENALTY << "\n" << end_log;


			if (tmp_energy < energy ) {
				this->convertGCBinLabeling2MultiLabeling(this->labeling, label_ordering[explidx], solver_labeling);
				best_expansionlabel = label_ordering[explidx];
				energy = tmp_energy;
				final_energy = energy;
				change_occured = true;
				bgn_log << LogType::INFO_LEVEL1 << "APM: \t\t\t***Updating best solution to energy " << energy << "\n" << end_log;

				std::fill_n(dirty_list, this->nlabels,true);
				dirty_list[label_ordering[explidx]] = false;
			
			} else  { dirty_list[label_ordering[explidx]] = false; }
			expansion_id++;
			graph->reset();
		}
		cycleid++;
		if (expansionOrdering.type == ExpansionOrdering::ExpansionOrderingType::CHAIN)
			break;
	} while (change_occured || IsAnyTrue_Lambda(dirty_list, this->nlabels));

	
	CLNDEL0D(pm_hhogconstraints);
	CLNDEL1D(dirty_list);
	CLNDEL1D(lcl_pairwise);
	CLNDEL1D(lcl_unary);
	CLNDEL1D(tmp_labeling);
	CLNDEL1D(exp_dataterms);
	CLNDEL1D(solver_labeling);
	CLNDEL1D(nlinks_ids);
	CLNDEL1D(tlinks_ids);
	CLNDEL0D(graph);

	//Do not return to the calling fuction before making sure that all files were written
	// otherwise the calling function might exit, thus terminating the current process 
	// and kill all the writing threads before they properly finish.
	for (uint32_t t = 0; t < labeling_writingthreads.size(); ++t) 
		labeling_writingthreads[t].join(); 

	Array3D<uint32_t> *labelling_array = new Array3D<uint32_t>();
	labelling_array->allocate(dims[0], dims[1], dims[2]);
	memcpy(labelling_array->data, labeling, sizeof(uint32_t)*this->ndspl);

	called_functions.dataterms = false;
	called_functions.init_labeling = false;
	return labelling_array;
}

template <class captype, class tcaptype, class flowtype>
void AlphaPathMoves<captype, tcaptype, flowtype>::getRuntime(double & const_rt, double &maxflow_rt, double &total_rt)
{
	this->maxflow_solver->getRuntimes(const_rt, maxflow_rt, total_rt);
}
template <class captype, class tcaptype, class flowtype>
void AlphaPathMoves<captype, tcaptype, flowtype>::setMinMarginGCArcs(uint32_t expl, uint32_t expansion_id)
{
	if (max_mm_r == 0) return ;
	//Set the inclusion min margin constraints
	int64_t idx = 0, gc_p, gc_q, pConstX;// , p;
	int64_t q, qx, qy, qz;
	
	for (uint32_t layer_id = 0; layer_id < (this->max_nlayers - 1); ++layer_id) 
	{
		/* p [ q1 q2 -1 q3 -1 -1 ... q4 ..-1] fast but uses a lot of memory*/
		if (this->mm_method == MinMarginMethod::IDX_BASED)
		{
			std::fill_n(minmarginIDX->edges.data, this->ndspl*this->max_n_mm_nhs, -1);
			this->fillMarginConstraints_IDX_BASED(layer_id, expl);

			for (int64_t p = 0; p < this->ndspl; ++p)
			{
				pConstX = p*this->minmarginIDX->edges.constX;
				for (idx = 0; idx < this->max_n_mm_nhs; ++idx)
				{
					if (this->minmarginIDX->edges.data[idx + pConstX] != -1)
					{
						q = this->minmarginIDX->edges.data[idx + pConstX];
						gc_p = p + layer_id*ndspl;
						gc_q = q + (layer_id + 1)*ndspl;
						this->graph->add_nlink(gc_p, gc_q, LRG_PENALTY, 0);
					}

				}
			}
		}
		else if (this->mm_method == MinMarginMethod::BIT_BASED || this->mm_method == MinMarginMethod::CUDA_BASED) {
			/*uses less memory but slow, need to speed it up via one-time computed look-up tables*/
			//* p [shift vectors bit enconding] memeroy effiecent but slow*/
			int64_t dims01 = dims[0] * dims[1];
			
			if (this->mm_method == MinMarginMethod::BIT_BASED)
				this->fillMarginConstraints_BIT_BASED(layer_id, expl); //the following bank reading is distructive (i.e. no need to memset before the next call) 
			else
				marginEdgesCuda->runKernel(layer_id, expl,this->labeling);
			

			const int64_t one = 1;
			int64_t bitid, g_bank_id = 0;
			for (int64_t pz = 0, p = 0; pz < this->dims[2]; ++pz)
			for (int64_t py = 0; py < this->dims[1]; ++py)
			for (int64_t px = 0; px < this->dims[0]; ++px, p++)
			{
				bitid = 0;
				for (auto bank_id = 0; bank_id < this->minmarginBIT->n_banks; ++bank_id, ++g_bank_id)
				{
					for (auto bit = 0; bit < 64; ++bit)
					{
						if (this->minmarginBIT->edges.data[g_bank_id] == 0)
						{
							bitid += 64 - bit;
							break;
						}
						if (this->minmarginBIT->edges.data[g_bank_id] & one)
						{
							qx = px + this->minmarginBIT->gbitid2shift[bitid * 3 + 0];
							qy = py + this->minmarginBIT->gbitid2shift[bitid * 3 + 1];
							qz = pz + this->minmarginBIT->gbitid2shift[bitid * 3 + 2];
							q = qx + qy*dims[0] + qz*dims01;

							gc_p = p + layer_id*ndspl;
							gc_q = q + (layer_id + 1)*ndspl;
							this->graph->add_nlink(gc_p, gc_q, LRG_PENALTY, 0);
						}
						this->minmarginBIT->edges.data[g_bank_id] = this->minmarginBIT->edges.data[g_bank_id]>>one;
						bitid++;
					}
				}
			}
		}
	}
}

template <class captype, class tcaptype, class flowtype>
void AlphaPathMoves<captype, tcaptype, flowtype>::convertGCBinLabeling2MultiLabeling(uint32_t * trgt_labeling, uint32_t exp_label, char * solver_labeling)
{
	int32_t * P, Pi;
	for (int64_t p = 0; p < this->ndspl; ++p)
	{
		P = this->paths->data + labeling[p] * paths->constX + exp_label*paths->constXY;
		trgt_labeling[p] = P[0];
		Pi = 1;
		while (P[Pi] != -1)
		{
			if (solver_labeling[p + this->ndspl*(Pi - 1)] == 1)
				trgt_labeling[p] = P[Pi++];
			else
				break;
		}
	}
};

template <class captype, class tcaptype, class flowtype>
void AlphaPathMoves<captype, tcaptype, flowtype>::saveLabelingToFile(uint32_t * in_labeling, std::string filename)
{
	int64_t n_dims = 3;
	if (this->dims[2] == 1)
		n_dims = 2;
	ArrayInOut<uint32_t> *tmp_array = new ArrayInOut<uint32_t>(this->dims, n_dims);
	uint32_t * tmp_labeling = tmp_array->getArrayCopyShallow();
	memcpy(tmp_labeling, in_labeling, sizeof(uint32_t)*this->ndspl);
	tmp_array->save(filename);
	delete tmp_array;
};
template <class captype, class tcaptype, class flowtype>
void AlphaPathMoves<captype, tcaptype, flowtype>::fillMarginConstraints_BIT_BASED(uint32_t in_Pi, uint32_t explbl)
{
	int64_t dims01 = dims[0] * dims[1], idx, p = 0, q, qx, qy, qz;
	int32_t *P, *Q;
	uint32_t Pi;
	//yoffset and zoffset are used the map a shift vector to a unique idx
	int64_t yoffset = this->max_mm_r * 2 + 1;
	int64_t zoffset = yoffset*yoffset;
	if (dims[2] == 1)
		zoffset = 0;
	for (int64_t pz = 0; pz < this->dims[2]; ++pz)
	for (int64_t py = 0; py < this->dims[1]; ++py)
	for (int64_t px = 0; px < this->dims[0]; ++px, p++)
	{
		P = this->paths->data + labeling[p] * paths->constX + explbl*paths->constXY;
		Pi = in_Pi;
		if (P[Pi] != -1 && Pi + 1 < this->max_nlayers && P[Pi + 1] != -1) //change // node gc_p is active  (p.s. Pi+1<this->max_nlayers is redundant because Pi is never the final layer)
		{//bottom-2-top  
			for (uint32_t shift_id = 0; shift_id < this->mm_nh_shifts[P[Pi]]->totalsize; shift_id += 3)
			{
				qx = px + this->mm_nh_shifts[P[Pi]]->data[shift_id + 0]; if (qx < 0 || qx >= this->dims[0]) continue;
				qy = py + this->mm_nh_shifts[P[Pi]]->data[shift_id + 1]; if (qy < 0 || qy >= this->dims[1]) continue;
				qz = pz + this->mm_nh_shifts[P[Pi]]->data[shift_id + 2]; if (qz < 0 || qz >= this->dims[2]) continue;
				q = qx + qy*dims[0] + qz*dims01;
				Q = this->paths->data + labeling[q] * paths->constX + explbl*paths->constXY;
				// The constraints are enforced between L(p) and Parent(L(p)) in the Metric-Tree, P[Pi+1]==Q[Pi+1] is not enough
				if (Q[Pi + 1] != -1 && Pi + 2 < this->max_nlayers && Q[Pi + 2] != -1 &&//change // node gc_q is active
					Q[Pi + 1] == this->child2parent[P[Pi]]) //there is a constraint if Q[Pi+1] is the parent of P[Pi] in the original tree
				{
					idx = this->minmarginBIT->shift2gbitid[this->mm_nh_shifts[P[Pi]]->data[shift_id]][this->mm_nh_shifts[P[Pi]]->data[shift_id + 1]][this->mm_nh_shifts[P[Pi]]->data[shift_id + 2]];
					INLINE_LOOKUPSETBIT(p, idx);
				}
			}
		}
		Pi += 2;
		if (Pi<this->max_nlayers && P[Pi] != -1 && //change
			Pi - 1>0 && Pi - 1 < this->max_nlayers && P[Pi - 1] != -1) //node gc_p is active 
		{//top-2-bottom 
			for (uint32_t shift_id = 0; shift_id < this->mm_nh_shifts[P[Pi]]->totalsize; shift_id += 3)
			{
				qx = px + this->mm_nh_shifts[P[Pi]]->data[shift_id + 0]; if (qx < 0 || qx >= this->dims[0]) continue;
				qy = py + this->mm_nh_shifts[P[Pi]]->data[shift_id + 1]; if (qy < 0 || qy >= this->dims[1]) continue;
				qz = pz + this->mm_nh_shifts[P[Pi]]->data[shift_id + 2]; if (qz < 0 || qz >= this->dims[2]) continue;
				q = qx + qy*dims[0] + qz*dims01;
				Q = this->paths->data + labeling[q] * paths->constX + explbl*paths->constXY;
				if (Q[Pi - 1] != -1 && // gc_q is active 
					Q[Pi - 1] == this->child2parent[P[Pi]])  //there is a constraint if Q[Pi-1] is the parent of P[Pi] in the original tree
				{
					idx = this->minmarginBIT->shift2gbitid[-this->mm_nh_shifts[P[Pi]]->data[shift_id]][-this->mm_nh_shifts[P[Pi]]->data[shift_id + 1]][-this->mm_nh_shifts[P[Pi]]->data[shift_id + 2]];
					INLINE_LOOKUPSETBIT(q, idx);
				}
			}
		}
	}
};
template <class captype, class tcaptype, class flowtype>
void AlphaPathMoves<captype, tcaptype, flowtype>::fillMarginConstraints_IDX_BASED(uint32_t in_Li, uint32_t explbl)
{
	int64_t dims01 = dims[0] * dims[1], idx, p = 0, q;
	int32_t *P, *Q;
	uint32_t Li;
	int64_t qx, qy, qz;
	//yoffset and zoffset are used the map a shift vector to a unique idx
	int64_t yoffset = this->max_mm_r * 2 + 1;
	int64_t zoffset = yoffset*yoffset;
	if (dims[2] == 1)
		zoffset = 0;
	for (int64_t pz = 0; pz < this->dims[2]; ++pz)
	for (int64_t py = 0; py < this->dims[1]; ++py)
	for (int64_t px = 0; px < this->dims[0]; ++px)
	{
		Li = in_Li;
		P = this->paths->data + labeling[p] * paths->constX + explbl*paths->constXY;
		if (P[Li] != -1 && Li + 1 < this->max_nlayers && P[Li + 1] != -1 && this->minmarginIDX->InSubTree->data[explbl + P[Li] * this->minmarginIDX->InSubTree->constX])
		{
			for (uint32_t shift_id = 0; shift_id < this->mm_nh_shifts[P[Li]]->totalsize; shift_id += 3)
			{
				qx = px + this->mm_nh_shifts[P[Li]]->data[shift_id + 0]; if (qx < 0 || qx >= this->dims[0]) continue;
				qy = py + this->mm_nh_shifts[P[Li]]->data[shift_id + 1]; if (qy < 0 || qy >= this->dims[1]) continue;
				qz = pz + this->mm_nh_shifts[P[Li]]->data[shift_id + 2]; if (qz < 0 || qz >= this->dims[2]) continue;
				q = qx + qy*dims[0] + qz*dims01;
				Q = this->paths->data + labeling[q] * paths->constX + explbl*paths->constXY;
				if ( Q[Li + 1] != -1 && Li + 2 < this->max_nlayers && Q[Li + 2] != -1 
					&& this->minmarginIDX->NotInSubTreeOrParent->data[labeling[q] + this->minmarginIDX->NotInSubTreeOrParent->constX*P[Li]])
				{
					idx = this->mm_nh_shifts[P[Li]]->data[shift_id] + this->max_mm_r + yoffset*(this->mm_nh_shifts[P[Li]]->data[shift_id + 1] + this->max_mm_r) + zoffset*(this->mm_nh_shifts[P[Li]]->data[shift_id + 2] + this->max_mm_r);
					if (this->minmarginIDX->edges.data[idx + p*this->minmarginIDX->edges.constX] == -1)
						this->minmarginIDX->edges.data[idx + p*this->minmarginIDX->edges.constX] = q;
				}
			}
		}
		Li += 2;
		if (Li < this->max_nlayers && P[Li] != -1 && Li - 1> 0 && P[Li - 1] != -1 && this->minmarginIDX->InSubTree->data[labeling[p] + P[Li] * this->minmarginIDX->InSubTree->constX]) //node gc_p is active 
		{
			for (uint32_t shift_id = 0; shift_id < this->mm_nh_shifts[P[Li]]->totalsize; shift_id += 3)
			{
				qx = px + this->mm_nh_shifts[P[Li]]->data[shift_id + 0]; if (qx < 0 || qx >= this->dims[0]) continue;
				qy = py + this->mm_nh_shifts[P[Li]]->data[shift_id + 1]; if (qy < 0 || qy >= this->dims[1]) continue;
				qz = pz + this->mm_nh_shifts[P[Li]]->data[shift_id + 2]; if (qz < 0 || qz >= this->dims[2]) continue;
				q = qx + qy*dims[0] + qz*dims01;
				Q = this->paths->data + labeling[q] * paths->constX + explbl*paths->constXY;
				if (Q[Li - 1] != -1 && Li - 2 >= 0 && Q[Li - 1] == this->child2parent[P[Li]] 
					&& this->minmarginIDX->NotInSubTreeOrParent->data[explbl + this->minmarginIDX->NotInSubTreeOrParent->constX*P[Li]])
				{
					idx = -this->mm_nh_shifts[P[Li]]->data[shift_id] + this->max_mm_r + yoffset*(-this->mm_nh_shifts[P[Li]]->data[shift_id + 1] + this->max_mm_r) + zoffset*(-this->mm_nh_shifts[P[Li]]->data[shift_id + 2] + this->max_mm_r);
					if (this->minmarginIDX->edges.data[idx + q*this->minmarginIDX->edges.constX] == -1)
						this->minmarginIDX->edges.data[idx + q*this->minmarginIDX->edges.constX] = p;
				}
			}
		}
		p++;
	}
	
};

template <class captype, class tcaptype, class flowtype>
void AlphaPathMoves<captype, tcaptype, flowtype>::fillPairewisePathSmoothnessArcs(int64_t p, int64_t q, uint32_t lp, uint32_t lq, uint32_t explbl, \
	captype * pairwise, int64_t * nlinks_ids,int64_t & pcounter, tcaptype * unary, \
	int64_t* tlinks_ids, int64_t &ucounter)
{   
	// int64_t nlinks_ids [max_nlayers*3*2]  each row is p,q
	// captype pairwise   [max_nlayers*3*2]  each row is wpq,wqp
	// int64_t tlinks_ids [max_nlayers*3]    each element is p         
	// tcaptype unary     [max_nlayers*3]    each element is weight from s to p

	int32_t * P = this->paths->data + lp*paths->constX + explbl*paths->constXY;
	int32_t * Q = this->paths->data + lq*paths->constX + explbl*paths->constXY;
	int32_t pi = 0, qi = 0, Pl1, Pl2, Ql1, Ql2;
	double lambda_wpq;
	pcounter = 0;
	ucounter = 0;
	for (uint32_t plength = 0; plength < (this->max_nlayers - 1); ++plength) 
	{
		if (P[pi] == -1 && Q[qi] == -1) //case 10
			break;
		Pl1 = P[pi]; Pl2 = P[pi + 1];
		Ql1 = Q[qi]; Ql2 = Q[qi + 1];

		lambda_wpq = this->lambda *this->func_getWpq(p, q, this->wpq_extra_data);
		if (Pl2 != -1 && Ql2 != -1) //case 1
		{		
			
			if (Pl1 == Ql1 && Pl2 == Ql2)
			{
				nlinks_ids[pcounter + 0] = p + pi*ndspl;
				nlinks_ids[pcounter + 1] = q + qi*ndspl;
				pairwise[pcounter + 0] = this->treemetric->data[Pl1 + this->treemetric->constX*Ql2] * lambda_wpq;
				pairwise[pcounter + 1] = this->treemetric->data[Pl2 + this->treemetric->constX*Ql1] * lambda_wpq; pcounter += 2;
			}
			else {
				tlinks_ids[ucounter] = p + pi*ndspl;
				unary[ucounter] = this->treemetric->data[Pl2 + this->treemetric->constX*Pl1] * lambda_wpq; ucounter += 1;

				tlinks_ids[ucounter] = q + qi*ndspl;
				unary[ucounter] = this->treemetric->data[Ql2 + this->treemetric->constX*Ql1] * lambda_wpq; ucounter += 1;
			}

		}
		else if (Pl2 != -1) { //case 2
			tlinks_ids[ucounter] = p + pi*ndspl;
			unary[ucounter] = this->treemetric->data[Pl2 + this->treemetric->constX*Pl1] * lambda_wpq; ucounter += 1;
		}
		else if (Ql2 != -1){ //case 3
			tlinks_ids[ucounter] = q + qi*ndspl;
			unary[ucounter] = this->treemetric->data[Ql1 + this->treemetric->constX*Ql2] * lambda_wpq; ucounter += 1;
		}else if (Pl1 == Ql1 &&  Pl1 == explbl) {
			//case 4 (need it to be there but their unary potentials will forbid severing them)
			nlinks_ids[pcounter + 0] = p + pi*ndspl;
			nlinks_ids[pcounter + 1] = q + qi*ndspl;
			pairwise[pcounter + 0] = this->treemetric->data[explbl + this->treemetric->constX*explbl] * lambda_wpq;
			pairwise[pcounter + 1] = this->treemetric->data[explbl + this->treemetric->constX*explbl] * lambda_wpq; pcounter += 2;
		}
		pi++;
		qi++;
	}
	pcounter /= 2;
}
}
#endif