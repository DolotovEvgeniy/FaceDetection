// Copyright 2017 Dolotov Evgeniy

#ifndef SVM_H
#define SVM_H

#include <vector>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <caffe/caffe.hpp>
#include <caffe/common.hpp>

#include <DeepPyramid/feature_map.hpp>

enum ObjectType {
    OBJECT,
    NOT_OBJECT
};

class FeatureMapSVM
{
private:
    cv::Ptr<cv::ml::SVM> svm;
    cv::Size mapSize;

public:
    FeatureMapSVM(cv::Size size);
    void save(const std::string& filename) const;
    void load(const std::string& filename);
    ObjectType predictObjectType(const FeatureMap& sample) const;
    double predictConfidence(const FeatureMap& sample) const;
    void train(const std::vector<FeatureMap>& positive,
               const std::vector<FeatureMap>& negative);
    float printAccuracy(const std::vector<FeatureMap>& positive,
                        const std::vector<FeatureMap>& negative) const;
    cv::Size  getMapSize() const;
};
#endif
