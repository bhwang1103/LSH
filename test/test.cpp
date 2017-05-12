#include <io/hdf5.h>
#include <util/flsh_table.h>
#include <util/random.h>
#include <util/params.h>
#include <algorithms/flsh_index.h>
#include <algorithms/dist.h>
using namespace flann;

int main(int argc, char** argv)
{
	int nn = 100; //the number of nearest neighbors
	Matrix<float> dataset;
	Matrix<float> query;
	load_from_file(dataset, "siftdata.hdf5", "base");
	load_from_file(query, "siftdata.hdf5", "query");
	Matrix<size_t> indices(new size_t[query.rows*nn], query.rows, nn);
	Matrix<float> dists(new float[query.rows*nn], query.rows, nn);

	//根据实验需要进行调整
	unsigned int table_ = 16;        // The number of hash tables to use
	unsigned int key_size = 20;      //The length of the key in the hash tables
	unsigned int multi_p = 8;        //The MAX-Number of levels to use in multi-probe
	unsigned int pool_size = 5000;   // The pool size
	
	FlshIndex<L2<float>>index(dataset, FlshIndexParams(table_, key_size, multi_p));
	index.buildIndex();
	
	index.pnnSearch(query, indices, dists, nn, pool_size, flann::SearchParams(-1));
	save_to_file(indices, "flshindices.hdf5", "indices");

	
	delete[] dataset.ptr();
	delete[] query.ptr();
	delete[] indices.ptr();
	delete[] dists.ptr();

	
	std::system("pause");
	return 0;
}




