// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <format_reader_ptr.h>

#include <algorithm>
#include <inference_engine.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <samples/args_helper.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <string>
#include <vector>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define DEBUG_MSR

#include "salient_detect.h"

using namespace InferenceEngine;

struct param_vals pv;

/**
 * @brief Checks input args
 * @param argc number of args
 * @param argv list of input arguments
 * @return bool status true(Success) or false(Fail)
 */
bool ParseAndCheckCommandLine(int argc, char* argv[]) {

	std::string pname (argv[0]);
	int non_optional_params = 0;
    signed char op;
    std::string filename;
    char action = '\0';

    while ((op = (char)getopt(argc, argv, "h::m::i::l::c::d::")) != -1) {

        dbg ("op=%d,optind=%d,arg=%s",op,optind,argv[optind]);
        switch (op) {
        case 'h':
			showUsage(pname);
			showAvailableDevices();
			pv.parm_h = h;
			return false;
		case 'm':
			pv.parm_m = m;
			++non_optional_params;
			if (optarg)
				pv.modelPath = optarg;
			else
				pv.modelPath = argv[optind];
			break;
		case 'i':
			++non_optional_params;
			pv.parm_i = i;
			if (optarg)
				filename = optarg;
			else
				filename = argv[optind];
			pv.images.push_back(filename);
			break;
		case 'l':
			pv.parm_l = l;
			if (optarg)
				pv.cpu_plugin_layers = optarg;
			else
				pv.cpu_plugin_layers = argv[optind];
			break;
		case 'c':
			pv.parm_c = c;
			if (optarg)
				pv.gpu_config = optarg;
			else
				pv.gpu_config = argv[optind];
			break;
		case 'd':
			pv.parm_d = d;
			if (optarg)
				pv.device = optarg;
			else
				pv.device = argv[optind];
			break;
		default:
			break;
		}
    }
	if (non_optional_params != NON_OPTIONAL_PARAMS) {
		dbg ("%s","error: non_optional_params");
		showUsage(pname);
		return false;
	}
    return true;
}

object_rectangle::object_rectangle(float image_width, float image_height)
{
	xmin = 0;
	xmax = 0;
	ymin = 0;
	ymax = 0;
	imageWidth = image_width;
	imageHeight = image_height;
}

void scan_rects (std::vector<struct object_rectangle>& obre_vec,
	cv::Mat img, int line, int& col_left)
{
	int col_right;

	/* start pixel is always of scanning is always non-black */
	for (col_right = col_left; col_right < img.cols; ++col_right) {  // seek non-black line
		if (img.at<float>(line, col_right)<=0)	// black pixel
			break;
	}
	for (int i=0; i<=obre_vec.size(); i++) {
		if (i == obre_vec.size()) {
			struct object_rectangle rect (img.cols, img.rows);
			rect.xmin = col_left;
			rect.xmax = col_right;
			rect.ymin = line;
			rect.ymax = line;
			obre_vec.push_back(rect);
			break;
		}
		if ((abs(col_left - obre_vec[i].xmin) <= SIDE_LINE_DEVIATION) &&
				(abs(col_right - obre_vec[i].xmax) <= SIDE_LINE_DEVIATION) &&
				(line - obre_vec[i].ymax) == 1) {
			obre_vec[i].ymax=line;
			if (col_left < obre_vec[i].xmin)
				obre_vec[i].xmin = col_left;
			if (col_right > obre_vec[i].xmax)
				obre_vec[i].xmax = col_right;
			break;
		}
	}
	col_left = col_right;
}

/**
 * \brief The entry point for the Inference Engine object_detection sample
 * application \file object_detection_sample_ssd/main.cpp \example
 * object_detection_sample_ssd/main.cpp
 */
int main(int argc, char* argv[]) {
	try {
		        /** This sample covers certain topology and cannot be generalized for any
         * object detection one **/

        // ------------------------------ Get Inference Engine version
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << "\n";

        // --------------------------- Parsing and validation of input arguments
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        // ------------------------------ Read input
        /** This vector stores paths to the processed images **/
        std::vector<std::string> images = pv.images;
        if (images.empty())
            throw std::logic_error("No suitable images were found");

        // --------------------------- Step 1. Initialize inference engine core
        slog::info << "Loading Inference Engine" << slog::endl;
        InferenceEngine::Core ie;

        // ------------------------------ Get Available Devices
        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions(pv.device) << std::endl;

        if (pv.parm_l == l) {
            // Custom CPU extension is loaded as a shared library and passed as a
            // pointer to base extension
            IExtensionPtr extension_ptr = std::make_shared<Extension>(pv.cpu_plugin_layers);
            ie.AddExtension(extension_ptr);
            slog::info << "Custom extension loaded: " << pv.cpu_plugin_layers << slog::endl;

        }

        if (pv.parm_c == c && (pv.device == "GPU" || pv.device == "MYRIAD" ||
			pv.device == "HDDL")) {
            // Config for device plugin custom extension is loaded from an .xml
            // description
            ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, pv.gpu_config}}, pv.device);
            slog::info << "Config for " << pv.device << 
				" device plugin custom extension loaded: " << pv.gpu_config << slog::endl;
        }

        // ---------------------------------------------------------------------------
        // Step 2. Read a model in OpenVINO Intermediate Representation (.xml and
        // .bin files) or ONNX (.onnx file) format

        slog::info << "Loading network files:" << slog::endl << pv.modelPath << slog::endl;

        CNNNetwork network = ie.ReadNetwork(pv.modelPath);

        // --------------------------- Step 3. Configure input & output
        // -------------------------------- Prepare input blobs
        slog::info << "Preparing input blobs" << slog::endl;

        /** Taking information about all topology inputs **/
        InputsDataMap inputsInfo(network.getInputsInfo());

        /**
         * Some networks have SSD-like output format (ending with DetectionOutput
         * layer), but having 2 inputs as Faster-RCNN: one for image and one for
         * "image info".
         *
         * Although object_datection_sample_ssd's main task is to support clean SSD,
         * it could score the networks with two inputs as well. For such networks
         * imInfoInputName will contain the "second" input name.
         */
        if (inputsInfo.size() != 1 && inputsInfo.size() != 2)
            throw std::logic_error("Sample supports topologies only with 1 or 2 inputs");

        std::string imageInputName, imInfoInputName;

        InputInfo::Ptr inputInfo = nullptr;

        SizeVector inputImageDims;

        /** Iterating over all input blobs **/
        for (auto& item : inputsInfo) {
            /** Working with first input tensor that stores image **/
            if (item.second->getInputData()->getTensorDesc().getDims().size() == 4) {
                imageInputName = item.first;

                inputInfo = item.second;

                slog::info << "Batch size is " << std::to_string(network.getBatchSize()) << slog::endl;

                /** Creating first input blob **/
                Precision inputPrecision = Precision::U8;
                item.second->setPrecision(inputPrecision);
            } else if (item.second->getInputData()->getTensorDesc().getDims().size() == 2) {
                imInfoInputName = item.first;

                Precision inputPrecision = Precision::FP32;
                item.second->setPrecision(inputPrecision);
                if ((item.second->getTensorDesc().getDims()[1] != 3 && item.second->getTensorDesc().getDims()[1] != 6)) {
                    throw std::logic_error("Invalid input info. Should be 3 or 6 values length");
                }
            }
        }

        if (inputInfo == nullptr) {
            inputInfo = inputsInfo.begin()->second;
        }

        // --------------------------- Prepare output blobs
        slog::info << "Preparing output blobs" << slog::endl;

        OutputsDataMap outputsInfo(network.getOutputsInfo());

        std::string outputName;
        DataPtr outputInfo;

        outputInfo = outputsInfo.begin()->second;
        outputName = outputInfo->getName();
        // SSD has an additional post-processing DetectionOutput layer
        // that simplifies output filtering, try to find it.
        if (auto ngraphFunction = network.getFunction()) {
            for (const auto& out : outputsInfo) {
                for (const auto& op : ngraphFunction->get_ops()) {
                    if (op->get_type_info() == ngraph::op::DetectionOutput::type_info && op->get_friendly_name() == out.second->getName()) {
                        outputName = out.first;
                        outputInfo = out.second;
                        break;
                    }
                }
            }
        }

        if (outputInfo == nullptr) {
            throw std::logic_error("Can't find a DetectionOutput layer in the topology");
        }

        const SizeVector outputDims = outputInfo->getTensorDesc().getDims();

        const int maxProposalCount = outputDims[2];
        const int objectSize = outputDims[3];

/*
        if (objectSize != 352) {
            throw std::logic_error("Output item should have 7 as a last dimension");
        }
*/

        if (outputDims.size() != 4) {
            throw std::logic_error("Incorrect output dimensions for SSD model");
        }

        /** Set the precision of output data provided by the user, should be called
         * before load of the network to the device **/
        outputInfo->setPrecision(Precision::FP32);

        // --------------------------- Step 4. Loading model to the device
        slog::info << "Loading model to the device" << slog::endl;

        ExecutableNetwork executable_network = ie.LoadNetwork(network, pv.device,
			parseConfig(pv.gpu_config));

		/****** time start point *********/

		using clock = std::chrono::system_clock;
		using msec = std::chrono::duration<double, std::milli>;
		const auto before = clock::now();

        // --------------------------- Step 5. Create infer request
        slog::info << "Create infer request" << slog::endl;
        InferRequest infer_request = executable_network.CreateInferRequest();

        // --------------------------- Step 6. Prepare input
        /** Collect images data ptrs **/
        std::vector<std::shared_ptr<unsigned char>> imagesData, originalImagesData;
        std::vector<float> imageWidths, imageHeights;
        for (auto& i : images) {
            FormatReader::ReaderPtr reader(i.c_str());
            if (reader.get() == nullptr) {
                slog::warn << "Image " + i + " cannot be read!" << slog::endl;
                continue;
            }
            /** Store image data **/
            std::shared_ptr<unsigned char> originalData(reader->getData());
            std::shared_ptr<unsigned char> data(reader->getData(inputInfo->getTensorDesc().getDims()[3], inputInfo->getTensorDesc().getDims()[2]));
            if (data.get() != nullptr) {
                originalImagesData.push_back(originalData);
                imagesData.push_back(data);
                imageWidths.push_back(reader->width());
                imageHeights.push_back(reader->height());
            }
        }
        if (imagesData.empty())
            throw std::logic_error("Valid input images were not found!");

        size_t batchSize = network.getBatchSize();
        slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;
        if (batchSize != imagesData.size()) {
            slog::warn << "Number of images " + std::to_string(imagesData.size()) + " doesn't match batch size " + std::to_string(batchSize) << slog::endl;
            batchSize = std::min(batchSize, imagesData.size());
            slog::warn << "Number of images to be processed is " << std::to_string(batchSize) << slog::endl;
        }

        /** Creating input blob **/
        Blob::Ptr imageInput = infer_request.GetBlob(imageInputName);

        /** Filling input tensor with images. First b channel, then g and r channels
         * **/
        MemoryBlob::Ptr mimage = as<MemoryBlob>(imageInput);
        if (!mimage) {
            slog::err << "We expect image blob to be inherited from MemoryBlob, but "
                         "by fact we were not able "
                         "to cast imageInput to MemoryBlob"
                      << slog::endl;
            return 1;
        }
        // locked memory holder should be alive all time while access to its buffer
        // happens
        auto minputHolder = mimage->wmap();

        size_t num_channels = mimage->getTensorDesc().getDims()[1];
        size_t image_size = mimage->getTensorDesc().getDims()[3] * mimage->getTensorDesc().getDims()[2];

        unsigned char* data = minputHolder.as<unsigned char*>();

        /** Iterate over all input images limited by batch size  **/
        for (size_t image_id = 0; image_id < std::min(imagesData.size(), batchSize); ++image_id) {
            /** Iterate over all pixel in image (b,g,r) **/
            for (size_t pid = 0; pid < image_size; pid++) {
                /** Iterate over all channels **/
                for (size_t ch = 0; ch < num_channels; ++ch) {
                    /**          [images stride + channels stride + pixel id ] all in
                     * bytes            **/
                    data[image_id * image_size * num_channels + ch * image_size + pid] = imagesData.at(image_id).get()[pid * num_channels + ch];
                }
            }
        }

        if (imInfoInputName != "") {
            Blob::Ptr input2 = infer_request.GetBlob(imInfoInputName);
            auto imInfoDim = inputsInfo.find(imInfoInputName)->second->getTensorDesc().getDims()[1];

            /** Fill input tensor with values **/
            MemoryBlob::Ptr minput2 = as<MemoryBlob>(input2);
            if (!minput2) {
                slog::err << "We expect input2 blob to be inherited from MemoryBlob, "
                             "but by fact we were not able "
                             "to cast input2 to MemoryBlob"
                          << slog::endl;
                return 1;
            }
            // locked memory holder should be alive all time while access to its
            // buffer happens
            auto minput2Holder = minput2->wmap();
            float* p = minput2Holder.as<PrecisionTrait<Precision::FP32>::value_type*>();

            for (size_t image_id = 0; image_id < std::min(imagesData.size(), batchSize); ++image_id) {
                p[image_id * imInfoDim + 0] = static_cast<float>(inputsInfo[imageInputName]->getTensorDesc().getDims()[2]);
                p[image_id * imInfoDim + 1] = static_cast<float>(inputsInfo[imageInputName]->getTensorDesc().getDims()[3]);
                for (size_t k = 2; k < imInfoDim; k++) {
                    p[image_id * imInfoDim + k] = 1.0f;  // all scale factors are set to 1.0
                }
            }
        }

        // --------------------------- Step 7. Do inference
        slog::info << "Start inference" << slog::endl;
        infer_request.Infer();

        // --------------------------- Step 8. Process output
        slog::info << "Processing output blobs" << slog::endl;

        const Blob::Ptr output_blob = infer_request.GetBlob(outputName);
        MemoryBlob::CPtr moutput = as<MemoryBlob>(output_blob);
        if (!moutput) {
            throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                                   "but by fact we were not able to cast output to MemoryBlob");
        }
        // locked memory holder should be alive all time while access to its buffer
        // happens
        auto moutputHolder = moutput->rmap();

		const msec duration = clock::now() - before;
		std::cout << "CALCULATION TIME: " << duration.count() << "ms" << std::endl;

		/********************************* gk: output blob data ??? **********************/
#ifdef SALIENT
		const float* result = moutputHolder.as<const PrecisionTrait<Precision::FP32>::value_type*>();
// TODO: 352,352 have to be replaced by variables name taken from blob(?) node.
		float img_width=352;
		float img_height=352;
		cv::Mat image_f(cv::Size(img_width,img_height), CV_32FC1, (void*)result, cv::Mat::AUTO_STEP);
		cv::Mat image(cv::Size(img_width,img_height),CV_32FC1, (void*)result, cv::Mat::AUTO_STEP);

//------------------------- debug image printing -----------------------
		dbg ("image.chans=%d,image.rows=%d,image.cols=%d",image.channels(),
			image.rows,image.cols);
		dbg ("%s\n","pixels");
		for (int j = 0, k; j < img_width; ++j) {
			printf ("\nline %d\n",j);
			k=0;
			for (int i=0; i<img_height; i++) {
				printf ("%f ", image.at<float>(j,i));
				if (++k % 10 == 0) {k=0; printf("\n%03d,%03d   ",j,i);}
			}
		}
		printf ("\n");

//-------------------------- salience rectangles determination ---------
		
		std::vector<struct object_rectangle> obre;

		for (int j=0; j<img_width; ++j) {
			for (int i=0; i<img_height; ++i) {
				if (image.at<float>(j,i) > 0) {	// non-black pixel
					scan_rects (obre,image,j,i);
				}
			}
		}
		cv::Mat image_orig = cv::imread(images[0], cv::IMREAD_GRAYSCALE);
		int n = 0;
		dbg ("img_width=%f,imageWidths=%f,image_height=%f,imageHeight=%f",
			img_width,imageWidths[0],img_height,imageHeights[0]);
		for (struct object_rectangle rect : obre) {
			rect.real_xmin = imageWidths[0] / img_width * rect.xmin;
			rect.real_xmax = imageWidths[0] / img_width * rect.xmax;
			rect.real_ymin = imageHeights[0] / img_height * rect.ymin;
			rect.real_ymax = imageHeights[0] / img_height * rect.ymax;
			dbg ("n=%d, point_left = %g,%g, point_right = %g,%g", n++, rect.xmin,rect.ymin,
				rect.xmax, rect.ymax);
			dbg ("   real_point_left = %g,%g,   real_point_right = %g,%g",
				rect.real_xmin,rect.real_ymin,rect.real_xmax,rect.real_ymax);
			cv::Point p1;
			p1.x = rect.xmin;
			p1.y = rect.ymin;
			cv::Point p2;
			p2.x = rect.xmax;
			p2.y = rect.ymax;
			cv::rectangle(image, p1, p2, {255,255,255});
			p1.x = rect.real_xmin;
			p1.y = rect.real_ymin;
			p2.x = rect.real_xmax;
			p2.y = rect.real_ymax;
			cv::rectangle(image_orig, p1, p2, {255,255,255});
		}

//-------------------------- salience image dispay ---------------------
		cv::namedWindow ("salient");
		for (int keyshow=0;keyshow != 0x1b;) {
	        cv::imshow("salient",image);
    	    keyshow = cv::waitKey();
		}
        cv::destroyWindow("salient");
		cv::namedWindow ("real image");
		for (int keyshow=0;keyshow != 0x1b;) {
	        cv::imshow("real image",image_orig);
    	    keyshow = cv::waitKey();
		}
        cv::destroyWindow("real_image");
		bool ret = false;
		std::vector<int> jpg_param;
		jpg_param.push_back(cv::IMWRITE_PNG_COMPRESSION);
		jpg_param.push_back(9);
		if ((ret = imwrite("out_img.jpg",image_f))) {		// TODO: to make customized image path
                slog::info << "Image " << "out_img.jpg " << "created!" << slog::endl;
            } else {
                throw std::logic_error(std::string("Can't create a file"));
            }
		return 0;
#else
        const float* detection = moutputHolder.as<const PrecisionTrait<Precision::FP32>::value_type*>();

        std::vector<std::vector<int>> boxes(batchSize);
        std::vector<std::vector<int>> classes(batchSize);

        /* Each detection has image_id that denotes processed image */
        for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) {
            auto image_id = static_cast<int>(detection[curProposal * objectSize + 0]);
            if (image_id < 0) {
                break;
            }

            float confidence = detection[curProposal * objectSize + 2];
            auto label = static_cast<int>(detection[curProposal * objectSize + 1]);
            auto xmin = static_cast<int>(detection[curProposal * objectSize + 3] * imageWidths[image_id]);
            auto ymin = static_cast<int>(detection[curProposal * objectSize + 4] * imageHeights[image_id]);
            auto xmax = static_cast<int>(detection[curProposal * objectSize + 5] * imageWidths[image_id]);
            auto ymax = static_cast<int>(detection[curProposal * objectSize + 6] * imageHeights[image_id]);

            std::cout << "[" << curProposal << "," << label << "] element, prob = " << confidence << "    (" << xmin << "," << ymin << ")-(" << xmax << ","
                      << ymax << ")"
                      << " batch id : " << image_id;

            if (confidence > 0.5) {
                /** Drawing only objects with >50% probability **/
                classes[image_id].push_back(label);
                boxes[image_id].push_back(xmin);
                boxes[image_id].push_back(ymin);
                boxes[image_id].push_back(xmax - xmin);
                boxes[image_id].push_back(ymax - ymin);
                std::cout << " WILL BE PRINTED!";
            }
            std::cout << std::endl;
        }

        for (size_t batch_id = 0; batch_id < batchSize; ++batch_id) {
            addRectangles(originalImagesData[batch_id].get(), imageHeights[batch_id], imageWidths[batch_id], boxes[batch_id], classes[batch_id],
                          BBOX_THICKNESS);
            const std::string image_path = "out_" + std::to_string(batch_id) + ".bmp";
            if (writeOutputBmp(image_path, originalImagesData[batch_id].get(), imageHeights[batch_id], imageWidths[batch_id])) {
                slog::info << "Image " + image_path + " created!" << slog::endl;
            } else {
                throw std::logic_error(std::string("Can't create a file: ") + image_path);
            }
        }
#endif
		/******************************** gk: end of output blob data ********************/
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    } catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    slog::info << slog::endl
               << "This sample is an API example, for any performance measurements "
                  "please use the dedicated benchmark_app tool"
               << slog::endl;
    return 0;
}
