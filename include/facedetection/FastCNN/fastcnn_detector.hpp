// Copyright 2017 Dolotov Evgeniy

#ifndef FASTCNN_H
#define FASTCNN_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <vector>
#include <cmath>

#include <face_detector.hpp>
#include <neural_network.hpp>
#include <FastCNN/heatmap.hpp>

const int DETECTOR_SIZE = 32;
const float SCALE = sqrt(2);
const int LEVEL_COUNT = 10;
class FastCNNDetector : public FaceDetector {
public:
    FastCNNDetector(cv::FileStorage config);
    void detect(const cv::Mat& image, std::vector<cv::Rect>& objects,
                std::vector<float>& confidence);
};

#endif // FASTCNN_H
