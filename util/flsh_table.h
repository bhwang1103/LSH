#ifndef FLANN_FLSH_TABLE_H_
#define FLANN_FLSH_TABLE_H_

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits.h>
// TODO as soon as we use C++0x, use the code in USE_UNORDERED_MAP
#if USE_UNORDERED_MAP
#include <unordered_map>
#else
#include <map>
#endif
#include <math.h>
#include <stddef.h>
#include <util/random.h>
#include <util/dynamic_bitset.h>
#include <util/matrix.h>

namespace flann
{

	namespace flsh
	{
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		/** What is stored in an FLSH bucket
		*/
		typedef size_t FeatureIndex;
		/** The id from which we can get a bucket back in an LSH table
		*/
		typedef size_t BucketKey;

		/** A bucket in an LSH table
		*/
		typedef std::vector<FeatureIndex> Bucket;

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		/** POD for stats about an LSH table
		*/
		struct FlshStats
		{
			std::vector<unsigned int> bucket_sizes_;
			size_t n_buckets_;
			size_t bucket_size_mean_;
			size_t bucket_size_median_;
			size_t bucket_size_min_;
			size_t bucket_size_max_;
			size_t bucket_size_std_dev;
			/** Each contained vector contains three value: beginning/end for interval, number of elements in the bin
			*/
			std::vector<std::vector<unsigned int> > size_histogram_;
		};



		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		/** Flsh hash table. As its key is a sub-feature, and as usually
		* the size of it is pretty small, we keep it as a continuous memory array.
		* The value is an index in the corpus of features (we keep it as an unsigned
		* int for pure memory reasons, it could be a size_t)
		*/
		template<typename ElementType>
		class FlshTable
		{
		public:
			/** A container of all the feature indices. Optimized for space
			*/
#if USE_UNORDERED_MAP
			typedef std::unordered_map<BucketKey, Bucket> BucketsSpace;
#else
			typedef std::map<BucketKey, Bucket> BucketsSpace;
#endif

			/** A container of all the feature indices. Optimized for speed
			*/
			typedef std::vector<Bucket> BucketsSpeed;

			/** Default constructor
			*/
			FlshTable()
			{
			}

			/** Default constructor
			* Create the mask and allocate the memory
			* @param feature_size is the size of the feature (considered as a ElementType[])
			* @param key_size is the number of bits that are turned on in the feature
			*/
			FlshTable(unsigned int /*feature_size*/, unsigned int /*key_size*/)
			{
				std::cerr << "FlSH is not implemented for that type" << std::endl;
				throw;
			}

			/** Add a feature to the table
			* @param value the value to store for that feature  在这里我猜value应该是索引
			* @param feature the feature itself
			*/
			void add(unsigned int value, const ElementType* feature)
			{
				// Add the value to the corresponding bucket
				BucketKey key = getKey(feature);
				//debug
				//std::cout << "当前的speedlevel :" << speed_level_ << std::endl;


				switch (speed_level_) {
				case kArray:
					// That means we get the buckets from an array
					buckets_speed_[key].push_back(value);
					break;
				case kBitsetHash:
					// That means we can check the bitset for the presence of a key
					key_bitset_.set(key);
					buckets_space_[key].push_back(value);
					break;
				case kHash:
				{
					// That means we have to check for the hash table for the presence of a key
					buckets_space_[key].push_back(value);
					break;
				}
				}
			}

			/** Add a set of features to the table
			* @param dataset the values to store
			*/
			void add(const std::vector< std::pair<size_t, ElementType*> >& features)
			{
#if USE_UNORDERED_MAP
				buckets_space_.rehash((buckets_space_.size() + features.size()) * 1.2);
#endif
				// Add the features to the table
				for (size_t i = 0; i < features.size(); ++i) {
					add(features[i].first, features[i].second);

				}
				// Now that the table is full, optimize it for speed/space
				optimize();

			}

			/** Get a bucket given the key
			* @param key
			* @return
			*/
			inline const Bucket* getBucketFromKey(BucketKey key) const
			{
				// Generate other buckets
				switch (speed_level_) {
				case kArray:
					// That means we get the buckets from an array
					return &buckets_speed_[key];
					break;
				case kBitsetHash:
					// That means we can check the bitset for the presence of a key
					if (key_bitset_.test(key)) return &buckets_space_.find(key)->second;
					else return 0;
					break;
				case kHash:
				{
					// That means we have to check for the hash table for the presence of a key
					BucketsSpace::const_iterator bucket_it, bucket_end = buckets_space_.end();
					bucket_it = buckets_space_.find(key);
					// Stop here if that bucket does not exist
					if (bucket_it == bucket_end) return 0;
					else return &bucket_it->second;
					break;
				}
				}
				return 0;
			}

			/** Compute the sub-signature of a feature
			*/
			size_t getKey(const ElementType* /*feature*/) const
			{
				std::cerr << "LSH is not implemented for that type" << std::endl;
				throw;
				return 1;
			}

			/** Get statistics about the table
			* @return
			*/
			FlshStats getStats() const;

			int getspeedlevel(){
				return this->speed_level_;
			}

		private:
			/** defines the speed fo the implementation
			* kArray uses a vector for storing data
			* kBitsetHash uses a hash map but checks for the validity of a key with a bitset
			* kHash uses a hash map only
			*/
			enum SpeedLevel
			{
				kArray, kBitsetHash, kHash
			};

			/** Initialize some variables
			*/
			void initialize(size_t key_size)
			{
				speed_level_ = kHash;
				key_size_ = key_size;
			}

			/** Optimize the table for speed/space
			*/
			void optimize()
			{

				// If we are already using the fast storage, no need to do anything
				if (speed_level_ == kArray) return;

				if ((buckets_space_.size() > ((size_t)1 << (key_size_ - 1))) ){
					speed_level_ = kArray;
					// Fill the array version of it
					buckets_speed_.resize(1 << key_size_);

					for (BucketsSpace::const_iterator key_bucket = buckets_space_.begin(); key_bucket != buckets_space_.end(); ++key_bucket){
						buckets_speed_[key_bucket->first] = key_bucket->second;
					}
					// Empty the hash table
					buckets_space_.clear();
					return;
				}

				// If the bitset is going to use less than 10% of the RAM of the hash map (at least 1 size_t for the key and two
				// for the vector) or less than 512MB (key_size_ <= 30)

				if (((std::max(buckets_space_.size(), buckets_speed_.size()) * CHAR_BIT * 3 * sizeof(BucketKey)) / 10
					>= size_t(1 << key_size_)) && (key_size_ <= 30) ) {

					speed_level_ = kBitsetHash;
					key_bitset_.resize(1 << key_size_);

					key_bitset_.reset();
					// Try with the BucketsSpace
					for (BucketsSpace::const_iterator key_bucket = buckets_space_.begin(); key_bucket != buckets_space_.end(); ++key_bucket) key_bitset_.set(key_bucket->first);
				}
				else {
					speed_level_ = kHash;
					key_bitset_.clear();
				}
			}

			template<typename Archive>
			void serialize(Archive& ar)
			{
				int val;
				if (Archive::is_saving::value) {
					val = (int)speed_level_;
				}
				ar & val;
				if (Archive::is_loading::value) {
					speed_level_ = (SpeedLevel)val;
				}

				ar & key_size_;
				ar & fmask_;

				if (speed_level_ == kArray) {
					ar & buckets_speed_;
				}
				if (speed_level_ == kBitsetHash || speed_level_ == kHash) {
					ar & buckets_space_;
				}
				if (speed_level_ == kBitsetHash) {
					ar & key_bitset_;
				}
			}
			friend struct serialization::access;

			/** The vector of all the buckets if they are held for speed
			*/
			BucketsSpeed buckets_speed_;

			/** The hash table of all the buckets in case we cannot use the speed version
			*/
			BucketsSpace buckets_space_;

			/** What is used to store the data */
			SpeedLevel speed_level_;

			/** If the subkey is small enough, it will keep track of which subkeys are set through that bitset
			* That is just a speedup so that we don't look in the hash table (which can be mush slower that checking a bitset)
			*/
			DynamicBitset key_bitset_;

			/** The size of the sub-signature in bits
			*/
			unsigned int key_size_;

			
			/** The mask to apply to a feature to get the hash key
			* 二维矩阵,大小为feature_size * sub_size(矩阵相乘：列数为数据的维数，行数为哈希key的长度)
			*/
			std::vector< std::vector<float> > fmask_;  
		};
		
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// 这里修改为支持float类型,将fmask_初始化,存的是[0,1]

		template<>
		inline FlshTable<float>::FlshTable(unsigned int feature_size, unsigned int subsignature_size)
		{
			int __num = feature_size*subsignature_size;
			initialize(subsignature_size);
			// Allocate the mask
			fmask_ = std::vector<std::vector<float> >(subsignature_size,std::vector<float>(feature_size,0));
			srand(time(NULL));
			for (int i = 0; i < subsignature_size; i++){
				for (int j = 0; j < feature_size; j++){
					fmask_[i][j] = GaussRand();
			
				}				
			}
	
		}

		/** Return the Subsignature of a feature
		* @param feature the feature to analyze
		*/
		template<>
		inline size_t FlshTable<float>::getKey(const float* feature) const
		{
			const float* feature_block_ptr = reinterpret_cast<const float*> (feature);
			//存的是每一个哈希函数当前状态的和
			std::vector<float> __sum(fmask_.size(), 0);
			int _i = fmask_[0].size();
			for (int i = 0; i < _i; i++)
			{
			// 计算当前的哈希值
				float feature_block = *feature_block_ptr;
			//	std::cout << feature_block << "*. "<<fmask_[i].size();
				for (int j = 0; j < fmask_.size(); j++) {
					__sum[j] += feature_block * fmask_[j][i];				
				}
				++feature_block_ptr;
			}
			//将__sum转变hashkey
			size_t subsignature = 0;
			size_t bit_index = 1;
			for (int k = 0; k < fmask_.size(); k++){
				subsignature += (__sum[k] > 0) ? bit_index : 0;
				bit_index <<= 1;
			}

			return subsignature;
		}

		template<>
		inline FlshStats FlshTable<float>::getStats() const
		{
			FlshStats stats;
			stats.bucket_size_mean_ = 0;
			if ((buckets_speed_.empty()) && (buckets_space_.empty())) {
				stats.n_buckets_ = 0;
				stats.bucket_size_median_ = 0;
				stats.bucket_size_min_ = 0;
				stats.bucket_size_max_ = 0;
				return stats;
			}

			if (!buckets_speed_.empty()) {
				for (BucketsSpeed::const_iterator pbucket = buckets_speed_.begin(); pbucket != buckets_speed_.end(); ++pbucket) {
					stats.bucket_sizes_.push_back(pbucket->size());
					stats.bucket_size_mean_ += pbucket->size();
				}
				stats.bucket_size_mean_ /= buckets_speed_.size();
				stats.n_buckets_ = buckets_speed_.size();
			}
			else {
				for (BucketsSpace::const_iterator x = buckets_space_.begin(); x != buckets_space_.end(); ++x) {
					stats.bucket_sizes_.push_back(x->second.size());
					stats.bucket_size_mean_ += x->second.size();
				}
				stats.bucket_size_mean_ /= buckets_space_.size();
				stats.n_buckets_ = buckets_space_.size();
			}

			std::sort(stats.bucket_sizes_.begin(), stats.bucket_sizes_.end());

			//  BOOST_FOREACH(int size, stats.bucket_sizes_)
	
			stats.bucket_size_median_ = stats.bucket_sizes_[stats.bucket_sizes_.size() / 2];
			stats.bucket_size_min_ = stats.bucket_sizes_.front();
			stats.bucket_size_max_ = stats.bucket_sizes_.back();

			// TODO compute mean and std
			/*float mean, stddev;
			stats.bucket_size_mean_ = mean;
			stats.bucket_size_std_dev = stddev;*/

			// Include a histogram of the buckets
			unsigned int bin_start = 0;
			unsigned int bin_end = 20;
			bool is_new_bin = true;
			for (std::vector<unsigned int>::iterator iterator = stats.bucket_sizes_.begin(), end = stats.bucket_sizes_.end(); iterator
				!= end;)
				if (*iterator < bin_end) {
					if (is_new_bin) {
						stats.size_histogram_.push_back(std::vector<unsigned int>(3, 0));
						stats.size_histogram_.back()[0] = bin_start;
						stats.size_histogram_.back()[1] = bin_end - 1;
						is_new_bin = false;
					}
					++stats.size_histogram_.back()[2];
					++iterator;
				}
				else {
					bin_start += 20;
					bin_end += 20;
					is_new_bin = true;
				}

				return stats;
		}

		// End the two namespaces
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* FLANN_LSH_TABLE_H_ */
