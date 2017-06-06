// Copyright 2017 Dolotov Evgeniy

#include <FasterRCNN/fasterrcnn_detector.hpp>
#include <nms.hpp>
#include <bounding_box.hpp>

using namespace cv;
using namespace std;


FasterRCNNDetector::FasterRCNNDetector(FileStorage config) {
    string model_file;
    string trained_net_file;
    config["net"] >> model_file;
    config["weights"] >> trained_net_file;
    net = shared_ptr<NeuralNetwork>(new NeuralNetwork(model_file,
                                                      trained_net_file));
}

void FasterRCNNDetector::detect(const Mat& image,
                                vector<Rect>& objects,
                                vector<float>& confidence) {
    CV_Assert(image.channels() == 3);
    vector<BoundingBox> detectedObjects;
    vector< vector<float> > data;
    net->processImage(image, data);
   
    for(int i = 0; i < data[0].size(); i++) {
        BoundingBox box;
        box.confidence = data[0][i];
        box.rect = Rect(Point(data[1][4*i], data[1][4*i+1]),
                Size(data[1][4*i+2], data[1][4*i+3]));
        detectedObjects.push_back(box);
    }

    NMSweightedAvg nms;
    nms.processBondingBox(detectedObjects, 0.2, 0.7);
    for (unsigned i = 0; i < detectedObjects.size(); i++) {

        objects.push_back(detectedObjects[i].rect);
        confidence.push_back(detectedObjects[i].confidence);
        cout << detectedObjects[i].rect << endl;
        cout << detectedObjects[i].confidence << endl;
    }
}
