#pragma once
#include "yolo.h"
#include "yolov8_pose.h"

void setYolov8Params(utils::InitParameter& param, const int &size_w, const int &size_h, const float &conf_thresh, const float &iou_thresh)
{
	param.class_names = utils::dataSets::trail1;
	param.num_class = 1; // for coco
	param.batch_size = 1;
	param.dst_h = size_h;
	param.dst_w = size_w;
	param.src_h = size_h;
	param.src_w = size_w;
	param.input_output_names = { "images",  "output0" };
	param.conf_thresh = conf_thresh;
	param.iou_thresh = iou_thresh;
	param.save_path = "None";
}

class KeypointDetector{
public:
	KeypointDetector(const utils::InitParameter& param, const std::string &model_path);
	~KeypointDetector();

	std::tuple<cv::Mat, cv::Mat, std::vector<cv::Mat>> inference(const cv::Mat &frame, const utils::InitParameter& param);

private:

	YOLOv8Pose yolov8;
	int delay_time = 0;
	int batchi = 0;
	std::vector<cv::Mat> imgs_batch;
};