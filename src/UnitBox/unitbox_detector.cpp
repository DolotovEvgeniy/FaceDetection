// Copyright 2017 Dolotov Evgeniy

#include <UnitBox/unitbox_detector.hpp>

using namespace cv;
using namespace std;

UnitboxDetector::UnitboxDetector(FileStorage config) {
    string model_file;
    string trained_net_file;
    config["net"] >> model_file;
    config["weights"] >> trained_net_file;
    net = shared_ptr<NeuralNetwork>(new NeuralNetwork(model_file,
                                                      trained_net_file));
}

void UnitboxDetector::detect(const Mat& image,
                             vector<Rect>& objects,
                             vector<float>& confidence) {
    CV_Assert(image.channels() == 3);
    //resize(image, image, Size(320,320));
    Size inputSize = net->inputLayerSize();
    Mat resizedImage;
    float scale;
    resizeToNetInputSize(image, inputSize.width, resizedImage, scale);

    ConfidenceMap confidenceMap;
    BoundingboxMap boundingboxMap;
    
    vector< vector<float> > data;
    net->processImage(image, data);

    vector<Point> centers;
    confidenceMap.findComponentCenters(0.6, centers);

    objects.clear();
    for(size_t i = 0; i < centers.size(); i++) {
        Point p = centers[i];
        objects.push_back(boundingboxMap.getRect(p.x, p.y));
        cout << "rect: " << boundingboxMap.getRect(p.x, p.y) << endl;
    }
}

void UnitboxDetector::resizeToNetInputSize(const Mat& image, int sideSize,
                                           Mat& resizedImage, float& scale) {

    Size newSize;
    if (image.cols > image.rows) {
        scale = sideSize/(float)image.rows;
        newSize.height = sideSize;
        newSize.width = image.cols*scale;
    } else {
        scale = sideSize/(float)image.cols;
        newSize.width = sideSize;
        newSize.height = image.rows*scale;
    }

    resize(image, resizedImage, newSize);
}

