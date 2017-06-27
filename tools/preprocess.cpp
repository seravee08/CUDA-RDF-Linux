#include <preprocess/anno.hpp>
#include <preprocess/depthImage.hpp>
#include <preprocess/rgbImage.hpp>
#include <preprocess/io.hpp>
#include <preprocess/math.hpp>
#include <util/fileutil.h>

#include <iostream>
#include <vector>
#include <map>

#include <opencv2/opencv.hpp>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

int num;
int radius;
int out_dimension;
int outSize;
int idx = 0;
std::vector<preprocess::DepthImage> depth_;
std::vector<preprocess::RGBImage> rgb_;
std::vector<preprocess::Anno> anno_;
#define batchSize 5000
int left;
bool has3d = false;

typedef int (*BrewFunction)();
typedef std::map<std::string, BrewFunction> BrewMap;
BrewMap brew_map;

std::vector<boost::filesystem::path> all_depth_paths;
std::vector<boost::filesystem::path> all_depth_paths_after;
std::vector<boost::filesystem::path> all_rgb_paths_after;
std::vector<boost::filesystem::path> all_anno_paths_after;
std::vector<boost::filesystem::path> all_anno_3d_paths_after;
std::vector<boost::filesystem::path> all_3d_path;


#define RegisterBrewFunction(func) \
class __Register_##func { \
  public:  \
    __Register_##func() { \
    	brew_map[#func] = &func; \
    } \
}; \
__Register_##func register_##func; \

static BrewFunction GetBrewFunction(const std::string& name) {
	if(brew_map.count(name)) {
		return brew_map[name];
	}
	else {
		std::cout << "Available preprocess actions:";
		for(BrewMap::iterator it = brew_map.begin(); it != brew_map.end(); it++) {
			std::cout << "\t" << it->first;
		}
		std::cout << std::endl << "Unknown action: " << name << std::endl;
		return NULL;
	}
}

void readPath(std::string type) {
	std::string p = "../data";
	std::string p3d = p;
	if(type == "preprocessed") {
		p += "/image";
		p3d += "/label";
	}
	boost::filesystem::path path(p);
	boost::filesystem::path path3d(p3d);
	preprocess::IO io_;

	if(type == "preprocessed") {
		addPaths(path3d, ".*anno3d.txt", all_3d_path);
	}
	else {
		addPaths(path3d, ".*anno3d_blender.txt", all_3d_path);
	}

	if(all_3d_path.size() > 0) {
		has3d = true;
	}

	addPaths(path, ".*.exr", all_depth_paths);
	int num_assumed = all_depth_paths.size();
	num = num_assumed;

	//decide the actual num of images;
	for(int idx = 0; idx < num_assumed; idx++) {
		boost::filesystem::path depth_path = all_depth_paths[idx];
		boost::filesystem::path rgb_path;
		boost::filesystem::path anno_path;
		boost::filesystem::path anno3d_path;

		if(type == "raw") {
			rgb_path = io_.rgbRawPath(depth_path);
		    if(!boost::filesystem::exists(rgb_path)) {
			    num--;
			    continue;
		    }

		    anno_path = io_.annoRawPath(depth_path);
		    if(!boost::filesystem::exists(anno_path)) {
			    num--;
			    continue;
		    }

		    if(has3d) {
		    	anno3d_path = io_.anno3dRawPath(depth_path);
		    	if(!boost::filesystem::exists(anno3d_path)) {
			    	num--;
			    	continue;
		    	}
		    }
		}
		else if(type == "preprocessed") {
			rgb_path = io_.rgbPath(depth_path);
		    if(!boost::filesystem::exists(rgb_path)) {
			    num--;
			    continue;
		    }

		    anno_path = io_.annoPath(depth_path);
		    if(!boost::filesystem::exists(anno_path)) {
			    num--;
			    continue;
		    }

		    if(has3d) {
		    	anno3d_path = io_.anno3dPath(depth_path);
		    	if(!boost::filesystem::exists(anno3d_path)) {
			    	num--;
			    	continue;
		    	}
		    }
		}
		else {
			std::cerr << "Invalid type!" << std::endl;
		}

		all_depth_paths_after.push_back(depth_path);
		all_rgb_paths_after.push_back(rgb_path);
		all_anno_paths_after.push_back(anno_path);
		if(has3d) {
			all_anno_3d_paths_after.push_back(anno3d_path);
		}
	}

	if(num < num_assumed) {
		std::cout << "rgb images or annotations are missing!" << std::endl;
	}
}

void readData(std::string type, const int& start, const bool& last) {
	//read data
	depth_.resize(batchSize);
	rgb_.resize(batchSize);
	anno_.resize(batchSize);

	preprocess::IO io_;
	int batch_size = !last ? batchSize : left;

	for(int idx = 0; idx < batch_size; idx++) {
		//read data
		boost::filesystem::path depth_path = all_depth_paths_after[idx + start];
		std::string id, ts;
		io_.getIdTs(depth_path, id, ts);
		cv::Mat_<float> depth;
		if(type == "raw") {
			depth = io_.readRawDepth(depth_path);
		}
		else if(type == "preprocessed") {
			depth = io_.readRawDepth(depth_path);
		}
		else {
			std::cerr << "Invalid type!" << std::endl;
		}
		depth_[idx].setDepth(depth);
		depth_[idx].setFileNames(id, ts);

		boost::filesystem::path rgb_path = all_rgb_paths_after[idx + start];
		cv::Mat_<cv::Vec3i> rgb = io_.readRGB(rgb_path);
		rgb_[idx].setRGB(rgb);
		rgb_[idx].setFileNames(id, ts);

		boost::filesystem::path anno_path = all_anno_paths_after[idx + start];
		std::vector<cv::Vec2f> anno = io_.readAnno(anno_path);
		anno_[idx].setAnno(anno);
		anno_[idx].setFileNames(id, ts);

		if(has3d) {
			boost::filesystem::path anno3d_path = all_anno_3d_paths_after[idx + start];
			std::vector<cv::Vec3f> anno3d = io_.readAnno3d(anno3d_path);
			anno_[idx].setAnno3d(anno3d);
		}
	}
}

void readWrongData(std::string type, const int& start, const bool& last) {
	//read data
	depth_.resize(batchSize);
	rgb_.resize(batchSize);
	anno_.resize(batchSize);

	preprocess::IO io_;
	int batch_size = !last ? batchSize : left;

	for(int idx = 0; idx < batch_size; idx++) {
		//read data
		boost::filesystem::path depth_path = all_depth_paths_after[idx + start];
		std::string id, ts;
		io_.getIdTs(depth_path, id, ts);
		cv::Mat_<float> depth;
		if(type == "raw") {
			depth = io_.readWrongDepth(depth_path);
		}
		else if(type == "preprocessed") {
			depth = io_.readWrongDepth(depth_path);
		}
		else {
			std::cerr << "Invalid type!" << std::endl;
		}
		depth_[idx].setDepth(depth);
		depth_[idx].setFileNames(id, ts);

		boost::filesystem::path rgb_path = all_rgb_paths_after[idx + start];
		cv::Mat_<cv::Vec3i> rgb = io_.readRGB(rgb_path);
		rgb_[idx].setRGB(rgb);
		rgb_[idx].setFileNames(id, ts);

		boost::filesystem::path anno_path = all_anno_paths_after[idx + start];
		std::vector<cv::Vec2f> anno = io_.readAnno(anno_path);
		anno_[idx].setAnno(anno);
		anno_[idx].setFileNames(id, ts);

		if(has3d) {
			boost::filesystem::path anno3d_path = all_anno_3d_paths_after[idx + start];
			std::vector<cv::Vec3f> anno3d = io_.readAnno3d(anno3d_path);
			anno_[idx].setAnno3d(anno3d);
		}
	}
}

void freeData() {
	depth_.clear();
	rgb_.clear();
	anno_.clear();
}


void writeData(const bool& last) {
	boost::filesystem::path out_path("../new_data");
	if(!boost::filesystem::exists(out_path)){
		boost::filesystem::create_directory(out_path);
	}
	boost::filesystem::path out_path_image("../new_data/image");
	if(!boost::filesystem::exists(out_path_image)){
		boost::filesystem::create_directory(out_path_image);
	}
	boost::filesystem::path out_path_mask("../new_data/mask");
	if(!boost::filesystem::exists(out_path_mask)){
		boost::filesystem::create_directory(out_path_mask);
	}
	boost::filesystem::path out_path_label("../new_data/label");
	if(!boost::filesystem::exists(out_path_label)){
		boost::filesystem::create_directory(out_path_label);
	}

	preprocess::IO io_;

	int batch_size = !last ? batchSize : left;

	//output
	for(int idx = 0; idx < batch_size; idx++) {
		std::string id, ts;
	    depth_[idx].getFileNames(id, ts);

		boost::format fmt_depth("%s_%s_depth.exr");
	    boost::filesystem::path out_path_depth = out_path_image / (fmt_depth % id % ts).str();
	    io_.writeDepth(out_path_depth, depth_[idx].getDepth());

	    boost::format fmt_rgb("%s_%s_rgb.png");
	    boost::filesystem::path out_path_rgb = out_path_mask / (fmt_rgb % id % ts).str();
	    io_.writeRGB(out_path_rgb, rgb_[idx].getRGB());

	    boost::format fmt_anno("%s_%s_anno.txt");
	    boost::filesystem::path out_path_anno = out_path_label / (fmt_anno % id % ts).str();
	    io_.writeAnno(out_path_anno, anno_[idx].getAnno());

	    if(has3d) {
	    	boost::format fmt_anno3d("%s_%s_anno3d.txt");
	    	boost::filesystem::path out_path_anno3d = out_path_label / (fmt_anno3d % id % ts).str();
	    	io_.writeAnno3d(out_path_anno3d, anno_[idx].getAnno3d());
	    }
	}
}

void writeSingleData(const int& idx) {
	boost::filesystem::path out_path("../new_data");
	if(!boost::filesystem::exists(out_path)){
		boost::filesystem::create_directory(out_path);
	}
	boost::filesystem::path out_path_image("../new_data/image");
	if(!boost::filesystem::exists(out_path_image)){
		boost::filesystem::create_directory(out_path_image);
	}
	boost::filesystem::path out_path_mask("../new_data/mask");
	if(!boost::filesystem::exists(out_path_mask)){
		boost::filesystem::create_directory(out_path_mask);
	}
	boost::filesystem::path out_path_label("../new_data/label");
	if(!boost::filesystem::exists(out_path_label)){
		boost::filesystem::create_directory(out_path_label);
	}
	//output
	std::string id, ts;
	depth_[idx].getFileNames(id, ts);
	preprocess::IO io_;

	boost::format fmt_depth("%s_%s_depth.exr");
	boost::filesystem::path out_path_depth = out_path_image / (fmt_depth % id % ts).str();
	io_.writeDepth(out_path_depth, depth_[idx].getDepth());

	boost::format fmt_rgb("%s_%s_rgb.png");
	boost::filesystem::path out_path_rgb = out_path_mask / (fmt_rgb % id % ts).str();
	io_.writeRGB(out_path_rgb, rgb_[idx].getRGB());

    boost::format fmt_anno("%s_%s_anno.txt");
    boost::filesystem::path out_path_anno = out_path_label / (fmt_anno % id % ts).str();
    io_.writeAnno(out_path_anno, anno_[idx].getAnno());

    if(has3d) {
    	boost::format fmt_anno3d("%s_%s_anno3d.txt");
    	boost::filesystem::path out_path_anno3d = out_path_label / (fmt_anno3d % id % ts).str();
    	io_.writeAnno3d(out_path_anno3d, anno_[idx].getAnno3d());
    }
}


int pad_crop() {
	readPath("preprocessed");
	int count = num / batchSize;
	left = num % batchSize;
	bool last = false;
	if(left > 0) {
		std::cout << count + 1 << " batches in total!" << std::endl;
	}
	else {
		std::cout << count << " batches in total!" << std::endl;
	}

	for(int i = 0; i < count; i++) {
		readData("preprocessed", i * batchSize, last);

		for(int idx = 0; idx < batchSize; idx++) {
			bool qualified = preprocess::Math::isQualified(rgb_[idx]);
			bool legal = qualified && preprocess::Math::pad_crop(depth_[idx], rgb_[idx], anno_[idx], radius);
			if(legal) {
				preprocess::Math::normalizeHand(depth_[idx], rgb_[idx]);
				writeSingleData(idx);
			}
		}
		freeData();
		std::cout << "batch " << i + 1 << " finished!" << std::endl;
	}

	last = true;
	if(left > 0) {
		readData("preprocessed", count * batchSize, last);

		for(int idx = 0; idx < left; idx++) {
			bool qualified = preprocess::Math::isQualified(rgb_[idx]);
			bool legal = qualified && preprocess::Math::pad_crop(depth_[idx], rgb_[idx], anno_[idx], radius);
			if(legal) {
				preprocess::Math::normalizeHand(depth_[idx], rgb_[idx]);
				writeSingleData(idx);
			}
		}
		freeData();
		std::cout << "batch " << count + 1 << " finished!" << std::endl;
	}

	return 0;
}
RegisterBrewFunction(pad_crop);


int findCandidates() {
	readPath("preprocessed");
	int count = num / batchSize;
	left = num % batchSize;
	bool last = false;
	if(left > 0) {
		std::cout << count + 1 << " batches in total!" << std::endl;
	}
	else {
		std::cout << count << " batches in total!" << std::endl;
	}

	for(int i = 0; i < count; i++) {
		readData("preprocessed", i * batchSize, last);

		for(int idx = 0; idx < batchSize; idx++) {
			bool qualified = preprocess::Math::isQualified(rgb_[idx]);
			bool legal = qualified && preprocess::Math::findCandidates(depth_[idx], rgb_[idx], anno_[idx], outSize);
			if(legal) {
				preprocess::Math::normalizeHand(depth_[idx], rgb_[idx]);
				writeSingleData(idx);
			}
		}
		freeData();
		std::cout << "batch " << i + 1 << " finished!" << std::endl;
	}

	last = true;
	if(left > 0) {
		readData("preprocessed", count * batchSize, last);

		for(int idx = 0; idx < left; idx++) {
			bool qualified = preprocess::Math::isQualified(rgb_[idx]);
			bool legal = qualified && preprocess::Math::findCandidates(depth_[idx], rgb_[idx], anno_[idx], outSize);
			if(legal) {
				preprocess::Math::normalizeHand(depth_[idx], rgb_[idx]);
				writeSingleData(idx);
			}
		}
		freeData();
		std::cout << "batch " << count + 1 << " finished!" << std::endl;
	}

	return 0;
}
RegisterBrewFunction(findCandidates);


int filter() {
	readPath("preprocessed");
	int count = num / batchSize;
	left = num % batchSize;
	bool last = false;
	if(left > 0) {
		std::cout << count + 1 << " batches in total!" << std::endl;
	}
	else {
		std::cout << count << " batches in total!" << std::endl;
	}

	for(int i = 0; i < count; i++) {
		readData("preprocessed", i * batchSize, last);

		for(int idx = 0; idx < batchSize; idx++) {
			preprocess::Math::high_filter(depth_[idx]);
			preprocess::Math::normalizeMinusOneToOne(depth_[idx]);
		}

		writeData(last);
		freeData();
		std::cout << "batch " << i + 1 << " finished!" << std::endl;
	}

	last = true;
	if(left > 0) {
		readData("preprocessed", count * batchSize, last);

		for(int idx = 0; idx < left; idx++) {
			preprocess::Math::high_filter(depth_[idx]);
			preprocess::Math::normalizeMinusOneToOne(depth_[idx]);
		}

		writeData(last);
		freeData();
		std::cout << "batch " << count + 1 << " finished!" << std::endl;
	}
	return 0;
}
RegisterBrewFunction(filter)


int resize() {
	readPath("preprocessed");
	int count = num / batchSize;
	left = num % batchSize;
	bool last = false;
	if(left > 0) {
		std::cout << count + 1 << " batches in total!" << std::endl;
	}
	else {
		std::cout << count << " batches in total!" << std::endl;
	}

	for(int i = 0; i < count; i++) {
		readData("preprocessed", i * batchSize, last);

		for(int idx = 0; idx < batchSize; idx++) {
			preprocess::Math::scale(depth_[idx], rgb_[idx], anno_[idx], out_dimension);
		}

		writeData(last);
		freeData();
		std::cout << "batch " << i + 1 << " finished!" << std::endl;
	}

	last = true;
	if(left > 0) {
		readData("preprocessed", count * batchSize, last);

		for(int idx = 0; idx < left; idx++) {
			preprocess::Math::scale(depth_[idx], rgb_[idx], anno_[idx], out_dimension);
		}

		writeData(last);
		freeData();
		std::cout << "batch " << count + 1 << " finished!" << std::endl;
	}
	return 0;
}
RegisterBrewFunction(resize)


int calculateMeanImage() {
	readPath("preprocessed");
	int count = num / batchSize;
	left = num % batchSize;
	bool last = false;
	if(left > 0) {
		std::cout << count + 1 << " batches in total!" << std::endl;
	}
	else {
		std::cout << count << " batches in total!" << std::endl;
	}

	int height = outSize;
	int width = outSize;

	cv::Mat final(height, width, CV_32FC1, float(0.0));
	for(int i = 0; i < count; i++) {
		cv::Mat temp(height, width, CV_32FC1, float(0.0));
		readData("preprocessed", i * batchSize, last);

		for(int idx = 0; idx < batchSize; idx++) {
			cv::Mat_<float> depth = depth_[idx].getDepth();
			for(int j = 0; j < height; j++) {
				for(int k = 0; k < width; k++) {
					temp.at<float>(j, k) += depth(j, k);
				}
			}
		}

		temp = temp / batchSize;

		if(i == 0) {
			temp.copyTo(final);
		}
		else {
			for(int j = 0; j < height; j++) {
				for(int k = 0; k < width; k++) {
					final.at<float>(j, k) = (final.at<float>(j, k) + temp.at<float>(j, k)) / 2.0;
				}
			}
		}
		temp.release();
		std::cout << "batch " << i + 1 << " finished!" << std::endl;
	}

	last = true;
	if(left > 0) {
		cv::Mat temp(height, width, CV_32FC1, float(0.0));
		readData("preprocessed", count * batchSize, last);

		for(int idx = 0; idx < left; idx++) {
			cv::Mat_<float> depth = depth_[idx].getDepth();
			for(int j = 0; j < height; j++) {
				for(int k = 0; k < width; k++) {
					temp.at<float>(j, k) += depth(j, k);
				}
			}
		}

		temp = temp / left;

		for(int j = 0; j < height; j++) {
			for(int k = 0; k < width; k++) {
				final.at<float>(j, k) = (final.at<float>(j, k) + temp.at<float>(j, k)) / 2.0;
			}
		}

		temp.release();
		std::cout << "batch " << count + 1 << " finished!" << std::endl;
	}

	// write the final to file
	preprocess::IO io_;
	boost::filesystem::path out_file("../mean.exr");
	io_.writeDepth(out_file, final);
}
RegisterBrewFunction(calculateMeanImage)


int rename() {
	readPath("raw");
	int count = num / batchSize;
	left = num % batchSize;
	bool last = false;
	if(left > 0) {
		std::cout << count + 1 << " batches in total!" << std::endl;
	}
	else {
		std::cout << count << " batches in total!" << std::endl;
	}

	for(int i = 0; i < count; i++) {
		readData("raw", i * batchSize, last);
		writeData(last);
		freeData();
		std::cout << "batch " << i + 1 << " finished!" << std::endl;
	}

	last = true;
	if(left > 0) {
		readData("raw", count * batchSize, last);
		writeData(last);
		freeData();
		std::cout << "batch " << count + 1 << " finished!" << std::endl;
	}

	return 0;
}
RegisterBrewFunction(rename)


int correct() {
	readPath("raw");
	int count = num / batchSize;
	left = num % batchSize;
	bool last = false;
	if(left > 0) {
		std::cout << count + 1 << " batches in total!" << std::endl;
	}
	else {
		std::cout << count << " batches in total!" << std::endl;
	}

	for(int i = 0; i < count; i++) {
		readWrongData("raw", i * batchSize, last);

		for(int idx = 0; idx < batchSize; idx++) {
			preprocess::Math::offset(rgb_[idx]);
		}

		writeData(last);
		freeData();
		std::cout << "batch " << i + 1 << " finished!" << std::endl;
	}

	last = true;
	if(left > 0) {
		readWrongData("raw", count * batchSize, last);

		for(int idx = 0; idx < left; idx++) {
			preprocess::Math::offset(rgb_[idx]);
		}

		writeData(last);
		freeData();
		std::cout << "batch " << count + 1 << " finished!" << std::endl;
	}

	return 0;
}
RegisterBrewFunction(correct)


int main(int argc, char** argv){
	namespace po = boost::program_options;
	po::options_description desc("options");
	desc.add_options()
		("crop", po::value<int>(&radius), "crop the images centered at the mass of ROI, with the given radius.")
		("filter", "add high-pass filter.")
		("resize", po::value<int>(&out_dimension), "resize the images to the given size.")
		("rename", "rename all images for later processing")
		("crop_test_1", po::value<int>(&radius), "crop test 1")
		("crop_test_2", po::value<int>(&radius), "crop test 2")
		("pad_crop", po::value<int>(&radius), "pad and crop the images")
		("merge", "find the mean image of all masks")
		("calculateMeanImage", po::value<int>(&outSize), "calculat the mean file")
		("findCandidates", po::value<int>(&outSize), "find all candidates")
		("correct", "fix exr bug");

	po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    try {
    	po::notify(vm);
    }
    catch(std::exception& e) {
    	std::cerr << "Error: " << e.what() << std::endl;
    	return 1;
    }

    if(argc > 1) {
    	std::string command(argv[1]);
    	std::size_t start = command.find_last_of("-") + 1;
    	std::cout << command.substr(start) << "!" << std::endl;
    	return GetBrewFunction(command.substr(start))();
    }
    else {
    	std::cout << "Should have options!" << std::endl;
    }

    return 0;
}
