// Copyright 2017 Dolotov Evgeniy

#ifndef FACE_DETECTOR_FDDB_TESTER_HPP
#define FACE_DETECTOR_FDDB_TESTER_HPP

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "face_detector.hpp"
#include "fddb_container.hpp"

class FaceDetectorFDDBTester {
public:
    void test(FaceDetector* detector,
              std::string fddb_file, std::string fddb_folder,
              std::string result_file);
};

#endif // FACE_DETECTOR_FDDB_TESTER_HPP
