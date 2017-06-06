// Copyright 2017 Dolotov Evgeniy

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <caffe/caffe.hpp>
#include <caffe/common.hpp>

class NeuralNetwork
{
public:
    NeuralNetwork() {}
    NeuralNetwork(std::string configFile, std::string trainedModel);
    void processImage(const cv::Mat& img,
                      std::vector< std::vector<float> > &data);
    cv::Size inputLayerSize();
    cv::Size outputLayerSize();
private:
    void fillNeuralNetInput(const cv::Mat& img);
    void getNeuralNetOutput(std::vector< std::vector<float> > &data);
    void calculate();

    caffe::shared_ptr<caffe::Net<float> > net;
};

#endif
