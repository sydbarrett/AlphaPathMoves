#include "graph.h"

#ifdef _MSC_VER
#pragma warning(disable: 4661)
#endif

// Instantiations: <captype, tcaptype, flowtype>
// IMPORTANT: 
//    flowtype should be 'larger' than tcaptype 
//    tcaptype should be 'larger' than captype

//template class Graph<int,int,int>;
//template class Graph<short,int,int>;
template class Graph<long long,long long,long long>;
//template class Graph<float,float,float>;
//template class Graph<double,double,double>;
//template class Graph<float,double,double>;
//template class Graph<__int16,__int32,__int32>;
//template class Graph<__int32,__int64,__int64>;
template class Graph<__int64,__int64,__int64>;

