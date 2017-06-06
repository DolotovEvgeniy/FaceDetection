// Copyright 2017 Dolotov Evgeniy

#include <UnitBox/confidence_map.hpp>
#include <iostream>

using namespace std;
using namespace cv;

ConfidenceMap::ConfidenceMap(int width, int height, const float* data) {
    float *tmpData = new float[width*height];
    for(int i = 0; i<width*height; i++) {
        tmpData[i] = data[i];
    }
    map = Mat(height, width, CV_32FC1, tmpData);
}

ConfidenceMap::ConfidenceMap(const ConfidenceMap& map) {
    map.map.copyTo(this->map);
}

ConfidenceMap& ConfidenceMap::operator=(const ConfidenceMap& map) {
    map.map.copyTo(this->map);
    return *this;
}

float& ConfidenceMap::at(int x, int y) {
    return map.at<float>(Point(x, y));
}

float ConfidenceMap::at(int x, int y) const {
    return map.at<float>(Point(x, y));
}

void ConfidenceMap::show(std::string windowName) const {
    Mat picture;
    cv::normalize(map, picture, 0, 255, cv::NORM_MINMAX);
  //  threshold(picture, picture, 0.5, 255, THRESH_BINARY);
  //  picture.convertTo(picture, CV_8UC1);
    imshow(windowName, picture);
    waitKey(0);
    imwrite("op.jpg", picture);
}

void ConfidenceMap::findComponentCenters(float thresholdValue,
                                         vector<Point>& centers) const {
    Mat binaryMask;
    imshow("map", map);
    threshold(map, binaryMask, thresholdValue, 1, THRESH_BINARY);
    imshow("bin", binaryMask);
    waitKey(0);

    binaryMask.convertTo(binaryMask, CV_8UC1);
    cout << "type: " << binaryMask.type() << endl;
    Mat centroids, labels, states;

    connectedComponentsWithStats(binaryMask, labels, states, centroids);
    show("lol");
    for(int i = 1; i < centroids.rows; i++) {
        int x = centroids.at<double>(i, 0);
        int y = centroids.at<double>(i, 1);
        centers.push_back(Point(x, y));
        cout << Point(x, y) << endl;
    }
}

Size ConfidenceMap::size() const {
    return Size(map.cols, map.rows);
}

ConfidenceMap::ConfidenceMap(int width, int height) {
    map = Mat::zeros(height, width, CV_8UC1);
}
