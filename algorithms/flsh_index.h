#ifndef FLANN_FLSH_INDEX_H_
#define FLANN_FLSH_INDEX_H_

#include <algorithm>
#include <cassert>
#include <cstring>
#include <map>
#include <vector>

#include "general.h"
#include "util/matrix.h"
#include "util/result_set.h"
#include "util/params.h"
#include "util/heap.h"
#include "util/allocator.h"
#include "util/random.h"
#include "util/saving.h"
#include "util/flsh_table.h"

namespace flann
{

	struct FlshIndexParams : public IndexParams
	{
		FlshIndexParams(unsigned int table_number = 12, unsigned int key_size = 20, unsigned int multi_probe_level = 2)
		{
			(*this)["algorithm"] = FLANN_INDEX_FLSH;
			// The number of hash tables to use
			(*this)["table_number"] = table_number;
			// The length of the key in the hash tables
			(*this)["key_size"] = key_size;
			// Number of levels to use in multi-probe (0 for standard LSH)
			(*this)["multi_probe_level"] = multi_probe_level;
		}
	};

	
	template<typename Distance>
	class FlshIndex
	{
	public:
		typedef typename Distance::ElementType ElementType;
		typedef typename Distance::ResultType DistanceType;

		
		/** Constructor
		* @param params parameters passed to the LSH algorithm
		* @param d the distance used
		*/
		FlshIndex(const IndexParams& params = FlshIndexParams(), Distance d = Distance()) :
			distance_(d), last_id_(0), size_(0), size_at_build_(0), veclen_(0),
			index_params_(params), removed_(false), removed_count_(0), data_ptr_(NULL)
		{
			table_number_ = get_param<unsigned int>(index_params_, "table_number", 12);
			key_size_ = get_param<unsigned int>(index_params_, "key_size", 20);
			multi_probe_level_ = get_param<unsigned int>(index_params_, "multi_probe_level", 2);

			fill_xor_mask(0, key_size_, multi_probe_level_, xor_masks_);
		}


		/** Constructor
		* @param input_data dataset with the input features
		* @param params parameters passed to the LSH algorithm
		* @param d the distance used
		*/
		FlshIndex(const Matrix<ElementType>& input_data, const IndexParams& params = FlshIndexParams(), Distance d = Distance()) :
			distance_(d), last_id_(0), size_(0), size_at_build_(0), veclen_(0),
			index_params_(params), removed_(false), removed_count_(0), data_ptr_(NULL)
		{
			table_number_ = get_param<unsigned int>(index_params_, "table_number", 12);
			key_size_ = get_param<unsigned int>(index_params_, "key_size", 20);
			multi_probe_level_ = get_param<unsigned int>(index_params_, "multi_probe_level", 2);

			fill_xor_mask(0, key_size_, multi_probe_level_, xor_masks_);

			setDataset(input_data);
		}

		FlshIndex(const FlshIndex& other) : BaseClass(other),
			tables_(other.tables_),
			table_number_(other.table_number_),
			key_size_(other.key_size_),
			multi_probe_level_(other.multi_probe_level_),
			xor_masks_(other.xor_masks_)
		{
		}

		FlshIndex& operator=(FlshIndex other)
		{
			this->swap(other);
			return *this;
		}

		virtual ~FlshIndex()
		{
			freeIndex();
		}


		void buildIndex(const Matrix<ElementType>& dataset)
		{
			setDataset(dataset);
			this->buildIndex();
		}

		/**
		* Builds the index
		*/
		void buildIndex()
		{
			tables_.resize(table_number_);
			std::vector<std::pair<size_t, ElementType*> > features;
			features.reserve(points_.size());
			for (size_t i = 0; i<points_.size(); ++i) {
				features.push_back(std::make_pair(i, points_[i]));   //i是索引 ，points_[i]是点的信息
			}
			for (unsigned int i = 0; i < table_number_; ++i) {
				flsh::FlshTable<ElementType>& table = tables_[i];
				table = flsh::FlshTable<ElementType>(veclen_, key_size_);
				//std::cout <<"这时的speedlevel 是"<< table.getspeedlevel();
				// Add the features to the table
				table.add(features);     //存在问题
			}

			size_at_build_ = size_;

		}

		void addPoints(const Matrix<ElementType>& points, float rebuild_threshold = 2)
		{
			assert(points.cols == veclen_);
			size_t old_size = size_;

			extendDataset(points);

			if (rebuild_threshold>1 && size_at_build_*rebuild_threshold<size_) {
				buildIndex();
			}
			else {
				for (unsigned int i = 0; i < table_number_; ++i) {
					flsh::FlshTable<ElementType>& table = tables_[i];
					for (size_t i = old_size; i<size_; ++i) {
						table.add(i, points_[i]);
					}
				}
			}
		}


		flann_algorithm_t getType() const
		{
			return FLANN_INDEX_FLSH;
		}


		template<typename Archive>
		void serialize(Archive& ar)
		{
			ar.setObject(this);

			ar & *static_cast<NNIndex<Distance>*>(this);

			ar & table_number_;
			ar & key_size_;
			ar & multi_probe_level_;

			ar & xor_masks_;
			ar & tables_;

			if (Archive::is_loading::value) {
				index_params_["algorithm"] = getType();
				index_params_["table_number"] = table_number_;
				index_params_["key_size"] = key_size_;
				index_params_["multi_probe_level"] = multi_probe_level_;
			}
		}

		void saveIndex(FILE* stream)
		{
			serialization::SaveArchive sa(stream);
			sa & *this;
		}

		void loadIndex(FILE* stream)
		{
			serialization::LoadArchive la(stream);
			la & *this;
		}

		/**
		* Computes the index memory usage
		* Returns: memory used by the index
		*/
		int usedMemory() const
		{
			return size_ * sizeof(int);
		}





		int pnnSearch(const Matrix<ElementType>& queries,
			Matrix<size_t>& indices,
			Matrix<DistanceType>& dists,
			size_t knn,
			size_t p,
			const SearchParams& params) const
		{
			assert(queries.cols == veclen_);
			assert(indices.rows >= queries.rows);
			assert(dists.rows >= queries.rows);
			assert(indices.cols >= knn);
			assert(dists.cols >= knn);

			int count = 0;
			if (params.use_heap == FLANN_True) {
#pragma omp parallel num_threads(params.cores)
			{
				KNNUniqueResultSet<DistanceType> resultSet(knn);
#pragma omp for schedule(static) reduction(+:count)
				for (int i = 0; i < (int)queries.rows; i++) {
					resultSet.clear();
					pfindNeighbors(resultSet, queries[i], p,params);
					size_t n = std::min(resultSet.size(), knn);
					resultSet.copy(indices[i], dists[i], n, params.sorted);
					indices_to_ids(indices[i], indices[i], n);
					count += n;
				}
			}
			}
			else {
#pragma omp parallel num_threads(params.cores)
			{
				KNNResultSet<DistanceType> resultSet(knn);
#pragma omp for schedule(static) reduction(+:count)
				for (int i = 0; i < (int)queries.rows; i++) {
					resultSet.clear();
					//debug
					//std::cout << "\n进的是Knnresult这里面\n";
					pfindNeighbors(resultSet, queries[i],p, params);
					size_t n = std::min(resultSet.size(), knn);
					resultSet.copy(indices[i], dists[i], n, params.sorted);
					indices_to_ids(indices[i], indices[i], n);
					count += n;
				}
			}
			}

			return count;
		}





		void pfindNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, size_t p,const SearchParams& /*searchParams*/) const
		{
			pgetNeighbors(vec, result,p);
		}

	protected:

		
	

		void freeIndex()
		{
			/* nothing to do here */
		}


	private:
		/** Defines the comparator on score and index
		*/
		typedef std::pair<float, unsigned int> ScoreIndexPair;
		struct SortScoreIndexPairOnSecond
		{
			bool operator()(const ScoreIndexPair& left, const ScoreIndexPair& right) const
			{
				return left.second < right.second;
			}
		};

		/** Fills the different xor masks to use when getting the neighbors in multi-probe LSH
		* @param key the key we build neighbors from
		* @param lowest_index the lowest index of the bit set
		* @param level the multi-probe level we are at
		* @param xor_masks all the xor mask
		*/

		// 新的 fill_xor_mask 方法
		void fill_xor_mask(flsh::BucketKey key, int lowest_index, unsigned int level,
			std::vector<flsh::BucketKey>& xor_masks)
		{
			std::vector<flsh::BucketKey> base;
			//初始化只有一位变化的base
			for (int index = 0; index < lowest_index; ++index) {
				// Create a new key
				size_t _ma = 1;
				flsh::BucketKey new_key = key | (_ma << index);
				//debug
				//std::cout<<"Key = :" << hex << (key) << std::endl;
				//std::cout << "NewKey = :" << hex << (new_key) << std::endl;
				base.push_back(new_key);
			}
			//将原来的key 0 放入
			xor_masks.push_back(key);
			
			if (level > 0){
				//一位变化放入
				flsh::BucketKey new_key;
				for (int i = 0; i < base.size(); i++){			
						new_key = base.at(i);
						xor_masks.push_back(new_key);
					}
			}
			if (level > 1){
				//2位变化放入
				flsh::BucketKey new_key;
				for (int i = 0; i < base.size(); i++){
					for (int j = i + 1; j < base.size(); j++){
						new_key = base.at(i) | base.at(j);
						xor_masks.push_back(new_key);
					}
				}
			}
			if (level > 2){
				//3位变化放入
				flsh::BucketKey new_key;
				for (int i = 0; i < base.size(); i++){
					for (int j = i + 1; j < base.size(); j++){
						for (int k = j + 1; k < base.size(); k++){
							new_key = base.at(i) | base.at(j) | base.at(k);
							xor_masks.push_back(new_key);
						}
					}
				}
			}
			if (level > 3){
				//4位变化放入
				flsh::BucketKey new_key;
				for (int i = 0; i < base.size(); i++){
					for (int j = i + 1; j < base.size(); j++){
						for (int k = j + 1; k < base.size(); k++){
							for (int _i = k + 1; _i < base.size(); _i++){
								new_key = base.at(i) | base.at(j) | base.at(k)|base.at(_i);
								xor_masks.push_back(new_key);
							}
						}
					}
				}
			}
			if (level > 4){
				//5位变化放入
				flsh::BucketKey new_key;
				for (int i = 0; i < base.size(); i++){
					for (int j = i + 1; j < base.size(); j++){
						for (int k = j + 1; k < base.size(); k++){
							for (int _i = k + 1; _i < base.size(); _i++){
								for (int _j = _i + 1; _j < base.size(); _j++){
									new_key = base.at(i) | base.at(j) | base.at(k) | base.at(_i)|base.at(_j);
									xor_masks.push_back(new_key);
								}
								
							}
						}
					}
				}
			}
			if (level > 5){
				//6位变化放入
				flsh::BucketKey new_key;
				for (int i = 0; i < base.size(); i++){
					for (int j = i + 1; j < base.size(); j++){
						for (int k = j + 1; k < base.size(); k++){
							for (int _i = k + 1; _i < base.size(); _i++){
								for (int _j = _i + 1; _j < base.size(); _j++){
									for (int _k = _j + 1; _k < base.size(); _k++){
										new_key = base.at(i) | base.at(j) | base.at(k) | base.at(_i) | base.at(_j)|base.at(_k);
										xor_masks.push_back(new_key);
									}
								}

							}
						}
					}
				}
			}
			if (level > 6){
				//7位变化放入
				flsh::BucketKey new_key;
				for (int i = 0; i < base.size(); i++){
					for (int j = i + 1; j < base.size(); j++){
						for (int k = j + 1; k < base.size(); k++){
							for (int _i = k + 1; _i < base.size(); _i++){
								for (int _j = _i + 1; _j < base.size(); _j++){
									for (int _k = _j + 1; _k < base.size(); _k++){
										for (int __i = _k + 1; __i < base.size(); __i++){
											new_key = base.at(i) | base.at(j) | base.at(k) | base.at(_i) | base.at(_j) | base.at(_k)|base.at(__i);
											xor_masks.push_back(new_key);
										}
									}
								}

							}
						}
					}
				}
			}
			if (level > 7){
				//8位变化放入,最多8位
				flsh::BucketKey new_key;
				for (int i = 0; i < base.size(); i++){
					for (int j = i + 1; j < base.size(); j++){
						for (int k = j + 1; k < base.size(); k++){
							for (int _i = k + 1; _i < base.size(); _i++){
								for (int _j = _i + 1; _j < base.size(); _j++){
									for (int _k = _j + 1; _k < base.size(); _k++){
										for (int __i = _k + 1; __i < base.size(); __i++){
											for (int __j = __i + 1; __j < base.size(); __j++){
												new_key = base.at(i) | base.at(j) | base.at(k) | base.at(_i) | base.at(_j) | base.at(_k) | base.at(__i)|base.at(__j);
												xor_masks.push_back(new_key);
											}
										}
									}
								}

							}
						}
					}
				}
			}



		}


		/** Performs the approximate nearest-neighbor search.
		* @param vec the feature to analyze
		* @param do_radius flag indicating if we check the radius too
		* @param radius the radius if it is a radius search
		* @param do_k flag indicating if we limit the number of nn
		* @param k_nn the number of nearest neighbors
		* @param checked_average used for debugging
		*/

		void pgetNeighbors(const ElementType* vec, ResultSet<DistanceType>& result,size_t p) const
		{

			typename std::vector<flsh::FlshTable<ElementType> >::const_iterator it_table;
			typename std::vector<flsh::FlshTable<ElementType> >::const_iterator table_begin = tables_.begin();
			typename std::vector<flsh::FlshTable<ElementType> >::const_iterator table_end = tables_.end();
			std::vector<flsh::BucketKey> _keys;
			for (it_table = table_begin; it_table != table_end; ++it_table) {
				size_t tempkey = it_table->getKey(vec);
				_keys.push_back(tempkey);
			}

			std::vector<flsh::BucketKey>::const_iterator xor_mask = xor_masks_.begin();
			std::vector<flsh::BucketKey>::const_iterator xor_mask_end = xor_masks_.end();
			std::vector<flsh::BucketKey>::const_iterator key_begin = _keys.begin();
			std::vector<flsh::BucketKey>::const_iterator key_end = _keys.end();
			std::vector<flsh::BucketKey>::const_iterator it_key;

			size_t _C = 0;
			//bool isfull_flag = false;
			for (; xor_mask != xor_mask_end; ++xor_mask) {
				for (it_key = key_begin,it_table = table_begin; it_key != key_end; ++it_key,++it_table) {
					size_t key = (*it_key);
			
					size_t sub_key = key ^ (*xor_mask);
					const flsh::Bucket* bucket = it_table->getBucketFromKey(sub_key);
					
					if (bucket == 0) continue;

					// Go over each descriptor index
					std::vector<flsh::FeatureIndex>::const_iterator training_index = bucket->begin();
					std::vector<flsh::FeatureIndex>::const_iterator last_training_index = bucket->end();
					DistanceType euclidean_distance;

					// Process the rest of the candidates
					for (; training_index < last_training_index; ++training_index) {
						if (removed_ && removed_points_.test(*training_index)) continue;
						// Compute the Hamming distance
						euclidean_distance = distance_(vec, points_[*training_index], veclen_);
						result.addPoint(euclidean_distance, *training_index);
						//将当前pool的size值+1
						++_C;
						
					
					}
					
				}
				//当前的C大于p的值时，退出
				if (_C >= p){
					break;
				}
			
			}
		}

		size_t id_to_index(size_t id)
		{
			if (ids_.size() == 0) {
				return id;
			}
			size_t point_index = size_t(-1);
			if (ids_[id] == id) {
				return id;
			}
			else {
				// binary search
				size_t start = 0;
				size_t end = ids_.size();

				while (start<end) {
					size_t mid = (start + end) / 2;
					if (ids_[mid] == id) {
						point_index = mid;
						break;
					}
					else if (ids_[mid]<id) {
						start = mid + 1;
					}
					else {
						end = mid;
					}
				}
			}
			return point_index;
		}


		void indices_to_ids(const size_t* in, size_t* out, size_t size) const
		{
			if (removed_) {
				for (size_t i = 0; i<size; ++i) {
					out[i] = ids_[in[i]];
				}
			}
		}

		void setDataset(const Matrix<ElementType>& dataset)
		{
			size_ = dataset.rows;
			veclen_ = dataset.cols;
			last_id_ = 0;

			ids_.clear();
			removed_points_.clear();
			removed_ = false;
			removed_count_ = 0;

			points_.resize(size_);
			for (size_t i = 0; i<size_; ++i) {
				points_[i] = dataset[i];
			}
		}

		void extendDataset(const Matrix<ElementType>& new_points)
		{
			size_t new_size = size_ + new_points.rows;
			if (removed_) {
				removed_points_.resize(new_size);
				ids_.resize(new_size);
			}
			points_.resize(new_size);
			for (size_t i = size_; i<new_size; ++i) {
				points_[i] = new_points[i - size_];
				if (removed_) {
					ids_[i] = last_id_++;
					removed_points_.reset(i);
				}
			}
			size_ = new_size;
		}

		void swap(FlshIndex& other)
		{
			BaseClass::swap(other);
			std::swap(tables_, other.tables_);
			std::swap(size_at_build_, other.size_at_build_);
			std::swap(table_number_, other.table_number_);
			std::swap(key_size_, other.key_size_);
			std::swap(multi_probe_level_, other.multi_probe_level_);
			std::swap(xor_masks_, other.xor_masks_);
		}



		/** Indices if the index was loaded from a file */
		bool loaded_;
		/** Parameters passed to the index */
		IndexParams index_params_;
		/**
		* The distance functor
		*/
		Distance distance_;
		/**
		* Each index point has an associated ID. IDs are assigned sequentially in
		* increasing order. This indicates the ID assigned to the last point added to the
		* index.
		*/
		size_t last_id_;

		/**
		* Number of points in the index (and database)
		*/
		size_t size_;

		/**
		* Number of features in the dataset when the index was last built.
		*/
		size_t size_at_build_;

		/**
		* Size of one point in the index (and database)
		*/
		size_t veclen_;

		/**
		* Parameters of the index.
		*/
		FlshIndexParams params;

		/**
		* Flag indicating if at least a point was removed from the index
		*/
		bool removed_;

		/**
		* Array used to mark points removed from the index
		*/
		DynamicBitset removed_points_;

		/**
		* Number of points removed from the index
		*/
		size_t removed_count_;

		/**
		* Array of point IDs, returned by nearest-neighbour operations
		*/
		std::vector<size_t> ids_;

		/**
		* Point data
		*/
		std::vector<ElementType*> points_;

		/**
		* Pointer to dataset memory if allocated by this index, otherwise NULL
		*/
		ElementType* data_ptr_;


		/** The different hash tables */
		std::vector<flsh::FlshTable<ElementType> > tables_;

		/** table number */
		unsigned int table_number_;
		/** key size */
		unsigned int key_size_;
		/** How far should we look for neighbors in multi-probe LSH */
		unsigned int multi_probe_level_;

		/** The XOR masks to apply to a key to get the neighboring buckets */
		std::vector<flsh::BucketKey> xor_masks_;


	};
}

#endif //FLANN_LSH_INDEX_H_
