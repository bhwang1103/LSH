#include <time.h>
#include <fstream>
#include <sstream>

#include <flann/io/hdf5.h>
#include <flann/algorithms/flsh_table.h>
#include <flann/algorithms/flsh_index.h>
#include <flann/util/random.h>
#include <flann/util/params.h>
#include <flann/algorithms/dist.h>
using namespace flann;
using namespace std;

int main(int argc, char** argv)
{

	int nn = 100;
	Matrix<float> dataset;
	Matrix<float> query;
	load_from_file(dataset, "siftdata.hdf5", "base");
	load_from_file(query, "siftdata.hdf5", "query");
	Matrix<size_t> indices(new size_t[query.rows*nn], query.rows, nn);
	Matrix<float> dists(new float[query.rows*nn], query.rows, nn);


	double time_Start = (double)clock();

	FlshIndex<L2<float>>index(dataset, FlshIndexParams(16, 20, 8));
	index.buildIndex();
	double time_End = (double)clock();
	std::cout << "build index time is " << (time_End - time_Start) / 1000.0 << "s" << std::endl;

	index.pnnSearch(query, indices, dists, nn, 100, flann::SearchParams(-1));
	save_to_file(indices, "flshindices.hdf5", "indices");

	
	delete[] dataset.ptr();
	delete[] query.ptr();
	delete[] indices.ptr();
	delete[] dists.ptr();

	
	std::system("pause");
	return 0;
}




