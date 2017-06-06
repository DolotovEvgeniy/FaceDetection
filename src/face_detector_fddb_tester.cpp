// Copyright 2017 Dolotov Evgeniy

#include <string>
#include <vector>

#include "face_detector_fddb_tester.hpp"
#include "fddb_container.hpp"
#include "fddb_result_container.hpp"

using namespace std;
using namespace cv;
void FaceDetectorFDDBTester::test(FaceDetector *detector,
                                  string fddb_file, string fddb_folder,
                                  string result_file) {
    FDDBContainer testData;
    testData.load(fddb_file, fddb_folder);
    FDDBResultContainer resultData;

    for(int i = 0; i < testData.size(); ++i) {
        string img_path;
        Mat image;
        vector<Rect> objects;
        testData.next(img_path, image, objects);

        vector<Rect> detectedObjects;
        vector<float> confidence;
        detector->detect(image, detectedObjects, confidence);

        resultData.add(img_path, detectedObjects, confidence);
    }

    resultData.save(result_file);
}
