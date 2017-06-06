// Copyright 2017 Dolotov Evgeniy

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <exception>

#include "face_detector.hpp"
#include "fddb_container.hpp"
#include "fddb_result_container.hpp"

using namespace cv;
using namespace std;

static const char argsDefs[] =
        "{test_config        |    | Path to test configuration file }"
        "{detector_config    |    | Path to detector configuration file }";

void printHelp(ostream& os)
{
    os << "\tUsage: --test_config=path/to/test/config.xml "
       << "--detector_config=path/to/detector/config.xml" << endl;
}

void parseCommandLine(int argc, char *argv[],
                      FileStorage& test_config,
                      FileStorage& detector_config) throw(runtime_error)
{
    cv::CommandLineParser parser(argc, argv, argsDefs);

    string testConfigFileName = parser.get<string>("test_config");
    if (testConfigFileName.empty()) {
        throw runtime_error("Test configaration file is not specified.");
    }

    test_config.open(testConfigFileName, FileStorage::READ);
    if (test_config.isOpened() == false) {
        throw runtime_error("File " + testConfigFileName +
                            " not found. Exiting.");
    }

    string detectorConfigFileName = parser.get<string>("detector_config");
    if (detectorConfigFileName.empty()) {
        throw runtime_error("Detector configaration file is not specified.");
    }

    detector_config.open(detectorConfigFileName, FileStorage::READ);
    if (detector_config.isOpened() == false) {
        throw runtime_error("File " + detectorConfigFileName +
                            " not found. Exiting.");
    }
}

int main(int argc, char *argv[]) {
    FileStorage test_config, detector_config;
    try {
        parseCommandLine(argc, argv, test_config, detector_config);
    }
    catch(runtime_error &e) {
        cerr << e.what() << endl;
    }
    FaceDetector* detector = new FastCNNDetector(detector_config);

    FDDBContainer testData;
    testData.load(test_config["fddb_file"],
                  test_config["fddb_folder"]);
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

    resultData.save(test_config["result_file"]);
}
