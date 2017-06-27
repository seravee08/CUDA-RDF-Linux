#include <preprocess/anno.hpp>


namespace preprocess{
	void Anno::setAnno(const std::vector<cv::Vec2f>& joints){
		joints_.resize(joints.size());
		for(unsigned int idx = 0; idx < joints.size(); ++idx){
			cv::Mat(joints[idx]).copyTo(joints_[idx]);
		}
	}

	void Anno::setAnno3d(const std::vector<cv::Vec3f>& joints3d){
		joints3d_.resize(joints3d.size());
		for(unsigned int idx = 0; idx < joints3d.size(); ++idx){
			for(int dim = 0; dim < 3; ++dim) {
				joints3d_[idx][dim] = joints3d[idx][dim];
			}
		}
	}

	std::vector<cv::Vec2f> Anno::getAnno(){
		return joints_;
	}

	std::vector<cv::Vec3f> Anno::getAnno3d(){
		return joints3d_;
	}

	void Anno::setFileNames(const std::string& id, const std::string& ts){
		id_ = id;
		ts_ = ts;
	}

	void Anno::getFileNames(std::string& id, std::string& ts){
		id = id_;
		ts = ts_;
	}
}