// Copyright 2017 Dolotov Evgeniy

#ifndef DEEP_PYRAMID_INCLUDE_BOUNDING_BOX_H_
#define DEEP_PYRAMID_INCLUDE_BOUNDING_BOX_H_

#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class BoxFeature {
public:
    virtual void reshapeToVector(cv::Mat&  feature) const = 0;
};

class BoundingBox {
public:
    int level;
    cv::Rect norm5Box;
    double confidence;
    cv::Rect rect;
    std::shared_ptr<BoxFeature> feature;
    bool operator<(const BoundingBox &object) {
        return confidence < object.confidence;
    }
};

#endif  // DEEP_PYRAMID_INCLUDE_BOUNDING_BOX_H_
