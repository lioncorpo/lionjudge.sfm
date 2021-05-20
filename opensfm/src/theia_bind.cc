#include <Eigen/Core>
#include <map>
#include <opencv2/core/core.hpp>
#include <vector>
#include <pybind11/pybind11.h>

#include "theia/matching/cascade_hasher.h"
#include "theia/matching/indexed_feature_match.h"

#include "types.h"


namespace csfm {

void MatchUsingCascadeHashing(std::vector<Eigen::VectorXf> f1,
                              std::vector<Eigen::VectorXf> f2,
                              int feature_size,
                              float lowes_ratio,
															std::vector<theia::IndexedFeatureMatch> *matches) {
  theia::CascadeHasher cascade_hasher;
  cascade_hasher.Initialize(feature_size);

	theia::HashedImage h1 = cascade_hasher.CreateHashedSiftDescriptors(f1);
	theia::HashedImage h2 = cascade_hasher.CreateHashedSiftDescriptors(f2);

	cascade_hasher.MatchImages(h1, f1, h2, f2, lowes_ratio, matches);
}

py::object match_using_cascade_hashing(pyarray_f features1,
                                 pyarray_f features2,
                                 float lowes_ratio) {
  if (features1.ndim() < 2 || features2.ndim() < 2) {
		py::list retn;
    return retn;
  }

  if (features1.ndim() != 2 || features2.ndim() != 2) {
    throw std::invalid_argument("feature vectors must be two dimensional");
  }

  if (features1.shape(1) != features2.shape(1)) {
    throw std::invalid_argument("length of feature vectors must be equal");
  }

  int feature_size(features1.shape(1));

  std::vector<Eigen::VectorXf> f1;
  for (ssize_t i = 0; i < features1.shape(0); i++) {
    Eigen::Map<Eigen::VectorXf> v(features1.mutable_data(i), features1.shape(1));
    f1.push_back(v);
  }

  std::vector<Eigen::VectorXf> f2;
  for (ssize_t i = 0; i < features2.shape(0); i++) {
    Eigen::Map<Eigen::VectorXf> v(features2.mutable_data(i), features2.shape(1));
    f2.push_back(v);
  }

	std::vector<theia::IndexedFeatureMatch> indexed_matches;
  MatchUsingCascadeHashing(f1, f2, feature_size, lowes_ratio, &indexed_matches);

	cv::Mat matches = cv::Mat(0, 2, CV_32S);
  for(std::vector<int>::size_type i = 0; i != indexed_matches.size(); i++) {
		cv::Mat tmp_match(1, 2, CV_32S);
		tmp_match.at<int>(0, 0) = indexed_matches[i].feature1_ind;
    tmp_match.at<int>(0, 1) = indexed_matches[i].feature2_ind;
    matches.push_back(tmp_match);
  }

  return py_array_from_cvmat<int>(matches);
}

}
