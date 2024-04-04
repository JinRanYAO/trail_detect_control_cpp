#include "app_yolov8_pose.h"

KeypointDetector::KeypointDetector(const utils::InitParameter& param, const std::string &model_path)
: yolov8(param) {

	imgs_batch.reserve(param.batch_size);

	std::vector<unsigned char> trt_file = utils::loadModel(model_path);
	if (!yolov8.init(trt_file))
	{
		sample::gLogError << "initEngine() ocur errors!" << std::endl;
	}
	yolov8.check();
}

KeypointDetector::~KeypointDetector()
{};

std::tuple<cv::Mat, cv::Mat, std::vector<cv::Mat>> KeypointDetector::inference(const cv::Mat &frame, const utils::InitParameter& param){

	imgs_batch.emplace_back(frame.clone());
    yolov8.copy(imgs_batch);
	utils::DeviceTimer d_t1; yolov8.preprocess(imgs_batch);  float t1 = d_t1.getUsedTime();
	utils::DeviceTimer d_t2; yolov8.infer();				  float t2 = d_t2.getUsedTime();
	utils::DeviceTimer d_t3; yolov8.postprocess(imgs_batch); float t3 = d_t3.getUsedTime();
	float avg_times[3] = { t1, t2, t3 };
	sample::gLogInfo << "preprocess time = " << avg_times[0] << "; "
		"infer time = " << avg_times[1] << "; "
		"postprocess time = " << avg_times[2] << std::endl;
	cv::Mat result_img;
	cv::Mat box;
	std::vector<cv::Mat> points;
	std::tie(result_img, box, points) = yolov8.showAndSave(param.class_names, delay_time, imgs_batch, avg_times);
	yolov8.reset();
	imgs_batch.clear();
	return std::make_tuple(result_img, box, points);
}