// Copyright 2017 Dolotov Evgeniy


#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "fddb_result_container.hpp"

using namespace std;
using namespace cv;


void FDDBResultContainer::save(string file_name) {
    ofstream file(file_name);

    if (file.is_open() == false) {
        std::cerr << "Output file '" << file_name
                  << "' not created. Exiting" << std::endl;
        return;
    }
    for (int i = 0; i < size(); i++) {
        file << imagesPath[i] << endl;
        file << objectsList[i].size() << endl;
        for (size_t j = 0; j < objectsList[i].size(); j++) {
            file << objectsList[i][j].x << " ";
            file << objectsList[i][j].y << " ";
            file << objectsList[i][j].width << " ";
            file << objectsList[i][j].height << " ";
            file << confidenceList[i][j] << endl;
        }
    }
}

void FDDBResultContainer::add(string image_path, vector<Rect> objects,
                                vector<float> confidence) {
    imagesPath.push_back(image_path);

    vector<Rect> addedObjects;
    vector<float> addedObjectsConfidence;
    for (size_t i = 0; i < objects.size(); i++) {
        addedObjects.push_back(objects[i]);
        addedObjectsConfidence.push_back(confidence[i]);
    }

    objectsList.push_back(addedObjects);
    confidenceList.push_back(addedObjectsConfidence);
}

int FDDBResultContainer::size() {
    return imagesPath.size();
}

int FDDBResultContainer::detectedObjectsCount() {
    int count = 0;
    for (size_t i = 0; i < objectsList.size(); i++) {
        count+=objectsList[i].size();
    }

    return count;
}
