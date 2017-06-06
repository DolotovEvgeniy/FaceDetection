// Copyright 2017 Dolotov Evgeniy

#ifndef CONFIDENCE_MAP_H
#define CONFIDENCE_MAP_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <vector>

class ConfidenceMap {
public:
    ConfidenceMap() = default;
    ConfidenceMap(int width, int height);
    ConfidenceMap(int width, int height, const float* data);
    ConfidenceMap(const ConfidenceMap& map);
    ConfidenceMap& operator=(const ConfidenceMap& map);
    cv::Size size() const;
    float& at(int x, int y);
    float at(int x, int y) const;
    void show(std::string windowName) const;
    void findComponentCenters(float thresholdValue,
                              std::vector<cv::Point>& centers) const;
private:
    cv::Mat map;
};

#endif // CONFIDENCE_MAP_H
