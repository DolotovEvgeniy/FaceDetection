// Copyright 2017 Dolotov Evgeniy

#include <nms.hpp>
#include <rectangle_transform.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

#include <opencv2/objdetect/objdetect.hpp>

using namespace std;
using namespace cv;

void NMS::divideIntoClusters(vector<BoundingBox>& objects,
                             const double &box_threshold,
                             vector<BoundingBoxCluster>& clusters) {
    while (!objects.empty()) {
        BoundingBox boxWithMaxConfidence = *max_element(objects.begin(),
                                                        objects.end());
        BoundingBoxCluster cluster;
        vector<BoundingBox> newObjects;
        for (size_t i = 0; i < objects.size(); i++) {
            if (IOU(boxWithMaxConfidence, objects[i]) <= box_threshold) {
                newObjects.push_back(objects[i]);
            } else {
                cluster.push_back(objects[i]);
            }
        }
        clusters.push_back(cluster);
        objects = newObjects;
    }
}

void NMS::processBondingBox(vector<BoundingBox> &objects,
                            const double &box_threshold,
                            const double &confidence_threshold) {
    vector<BoundingBox> detectedObjects;
    vector<BoundingBoxCluster> clusters;

    divideIntoClusters(objects, box_threshold, clusters);
    for (auto cluster = clusters.begin(); cluster != clusters.end();
         cluster++) {
        int boundBoxCount = cluster->size();
        cout << "Box in cluster:" << boundBoxCount << endl;
        BoundingBox box = mergeCluster(*cluster, confidence_threshold);
        detectedObjects.push_back(mergeCluster(*cluster,
                                               confidence_threshold));
    }
    objects = detectedObjects;
}

BoundingBox NMSmax::mergeCluster(BoundingBoxCluster &cluster,
                                 const double &confidence_threshold) {
    return *max_element(cluster.begin(), cluster.end());
}

BoundingBox NMSavg::mergeCluster(BoundingBoxCluster &cluster,
                                 const double &confidence_threshold) {
    BoundingBox boundingBoxWithMaxConfidence = *max_element(cluster.begin(),
                                                            cluster.end());
    double maxConfidenceInCluster = boundingBoxWithMaxConfidence.confidence;

    vector<Rect> rectangleWithMaxConfidence;
    for (auto boundingBox = cluster.begin(); boundingBox != cluster.end();
         boundingBox++) {
        if (boundingBox->confidence
                > confidence_threshold*maxConfidenceInCluster) {
            rectangleWithMaxConfidence.push_back(boundingBox->rect);
        }
    }

    BoundingBox resultBoundingBox;
    resultBoundingBox.rect = avg_rect(rectangleWithMaxConfidence);
    resultBoundingBox.confidence = maxConfidenceInCluster;
    cout << "Confidence" << maxConfidenceInCluster << endl;
    return resultBoundingBox;
}

BoundingBox NMSweightedAvg::mergeCluster(BoundingBoxCluster &cluster,
                                         const double &confidence_threshold) {
    BoundingBox boundingBoxWithMaxConfidence = *max_element(cluster.begin(),
                                                            cluster.end());
    double maxConfidenceInCluster = boundingBoxWithMaxConfidence.confidence;

    vector<BoundingBox> rectangleWithMaxConfidence;
    for (auto boundingBox = cluster.begin(); boundingBox != cluster.end();
         boundingBox++) {
        if (boundingBox->confidence
                > confidence_threshold*maxConfidenceInCluster) {
            rectangleWithMaxConfidence.push_back(*boundingBox);
        }
    }

    BoundingBox resultBoundingBox;
    resultBoundingBox.rect = weightedAvg_rect(rectangleWithMaxConfidence);
    resultBoundingBox.confidence = maxConfidenceInCluster;
cout << "Confidence" << maxConfidenceInCluster << endl;
    return resultBoundingBox;
}

BoundingBox NMSintersect::mergeCluster(BoundingBoxCluster &cluster,
                                       const double &confidence_threshold) {
    BoundingBox boundingBoxWithMaxConfidence = *max_element(cluster.begin(),
                                                            cluster.end());
    double maxConfidenceInCluster = boundingBoxWithMaxConfidence.confidence;

    vector<Rect> rectangleWithMaxConfidence;
    for (auto boundingBox = cluster.begin(); boundingBox != cluster.end();
         boundingBox++) {
        if (boundingBox->confidence
                > confidence_threshold*maxConfidenceInCluster) {
            rectangleWithMaxConfidence.push_back(boundingBox->rect);
        }
    }

    BoundingBox resultBoundingBox;
    resultBoundingBox.rect = intersectRectangles(rectangleWithMaxConfidence);
    resultBoundingBox.confidence = maxConfidenceInCluster;

    return resultBoundingBox;
}
