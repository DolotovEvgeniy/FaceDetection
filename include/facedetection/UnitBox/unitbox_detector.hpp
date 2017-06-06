// Copyright 2017 Dolotov Evgeniy

#ifndef UNITBOX_H
#define UNITBOX_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <vector>

#include <face_detector.hpp>
#include <neural_network.hpp>
#include <UnitBox/boundingbox_map.hpp>
#include <UnitBox/confidence_map.hpp>

const float SAMPLE_INTERSECTION_PERCENT = 0.3;

enum MergeType {
    VERTICAL,
    HORIZONT
};

class UnitboxDetector : public FaceDetector {
public:
    UnitboxDetector(cv::FileStorage config);
    void detect(const cv::Mat& img, std::vector<cv::Rect>& objects,
                    std::vector<float>& confidence);
private:
    void resizeToNetInputSize(const cv::Mat& image, int sideSize,
                              cv::Mat& resizedImage, float& scale);
};

#endif // UNITBOX_H
