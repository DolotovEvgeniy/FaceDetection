// Copyright 2017 Dolotov Evgeniy

#ifndef FACE_DETECTOR_HPP
#define FACE_DETECTOR_HPP

#include <vector>
#include <string>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fddb_container.hpp>
#include <neural_network.hpp>

class FaceDetector {
public:
    virtual void detect(const cv::Mat& img, std::vector<cv::Rect>& objects,
                std::vector<float>& confidence) = 0;
protected:
    std::shared_ptr<NeuralNetwork> net;
};

#endif // FACE_DETECTOR_HPP
