// Copyright 2017 Dolotov Evgeniy

#ifndef RECTANGLE_TRANSFORM_H
#define RECTANGLE_TRANSFORM_H

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <bounding_box.hpp>

cv::Point getRectangleCenter(const cv::Rect& rect);

cv::Rect makeRectangle(const cv::Point& center,
                       const int& width, const int& height);

cv::Rect weightedAvg_rect(const std::vector<BoundingBox>& objects);

cv::Rect avg_rect(const std::vector<cv::Rect>& rectangles);

cv::Rect scaleRect(const cv::Rect& rect, double scale);

cv::Rect intersectRectangles(const std::vector<cv::Rect>& rectangles);

double IOU(const cv::Rect& rect1, const cv::Rect& rect2);

double IOU(const BoundingBox& box1, const BoundingBox& box2);

#endif
