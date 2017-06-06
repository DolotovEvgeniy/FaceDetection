// Copyright 2017 Dolotov Evgeniy

#include <FastCNN/fastcnn_detector.hpp>
#include <nms.hpp>
#include <bounding_box.hpp>

using namespace cv;
using namespace std;


FastCNNDetector::FastCNNDetector(FileStorage config) {
    string model_file;
    string trained_net_file;
    config["net"] >> model_file;
    config["weights"] >> trained_net_file;
    net = shared_ptr<NeuralNetwork>(new NeuralNetwork(model_file,
                                                      trained_net_file));
}

void FastCNNDetector::detect(const Mat& image,
                             vector<Rect>& objects,
                             vector<float>& confidence) {
    CV_Assert(image.channels() == 3);
    int initialWidth;
    int initialHeight;
    if ( image.cols > image.rows ) {
        initialWidth = DETECTOR_SIZE*image.cols/float(image.rows);
        initialHeight = DETECTOR_SIZE;
    } else {
        initialWidth = DETECTOR_SIZE;
        initialHeight = DETECTOR_SIZE*image.rows/float(image.cols);
    }
    vector<BoundingBox> detectedObjects;
    for (int i = 0; i < LEVEL_COUNT; i++) {
        Mat resizedImage;
        Size size(initialWidth*pow(SCALE, i), initialHeight*pow(SCALE, i));
        cout << "LEVEL" << i << ": "<< size << endl;
        resize(image, resizedImage, size);
        Heatmap map;
        vector< vector<float> > data;
        net->processImage(resizedImage, data);
        float levelScale = image.cols/float(resizedImage.cols);
        int count = 0;
        for (int x = 0; x < map.size().width; x++) {
            for ( int y = 0; y < map.size().height; y++) {
                if(map.at(x, y) > 0.4) {
                    Rect rect(Point(2*x*levelScale, 2*y*levelScale),
                              Size(DETECTOR_SIZE*levelScale,
                                   DETECTOR_SIZE*levelScale));
                    BoundingBox box;
                    box.rect = rect;
                    box.confidence = map.at(x, y);
                    detectedObjects.push_back(box);
                    count++;
                }
            }
        }
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
