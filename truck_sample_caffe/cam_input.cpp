/*
 * input frames from librealsense recorded .bag files
*/

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2/rsutil.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#define _USE_MATH_DEFINES
#include <math.h>
#include <sys/queue.h>
#include <queue>
#include <unordered_set>
#include <map>
#include <functional>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <librealsense2/rs_advanced_mode.hpp>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/resource.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <cmath>

#include <sstream>
#include <iostream>
#include <fstream>
#include <streambuf>
#include <istream>
#include <vector>

#include <cam.h>

static rs2::device device;
ms_cam::ms_cam (std::string fl_png)
{
	png_fl = fl_png;
	ms_duration = 0;
	sec_fl = 0;
	msec_fl = 0;
	ms_time = 0;
	dbg ("%s",__func__);
#if 0
	fl_disassemble();	// extract seconds and milliseconds from file name
	std::string filename(path_fl);
	filename += "../../" + base_fl + ".bag";
	dbg ("filename=%s",filename.c_str());
#else
	std::string filename(fl_png);
#endif
	rs2::pipeline pipec(ctx);
	pipe = pipec;
	cfg.enable_device_from_file(filename.c_str());
	profile = pipe.start(cfg);
	device = profile.get_device();
	rs2::playback playback = device.as<rs2::playback>();
	playback.set_real_time(true);
	ms_duration = playback.get_duration().count() / 1000000000;
	dbg ("ms_duration = %d",ms_duration);
}

#if 0
void ms_cam::fl_disassemble()
{
	char tmp_fl [256];
	char path_fl_c [256];
	int n;
	int m;
	int pos;

	pos = png_fl.find_last_of ("/");
	if (pos != std::string::npos) {
		n = png_fl.copy(tmp_fl, sizeof(tmp_fl),pos+1);
		m = png_fl.copy (path_fl_c, pos+1, 0);
	} else {
		n = png_fl.copy(tmp_fl, sizeof(tmp_fl));
		m=0;
	}
	tmp_fl[n] = '\0';
	path_fl_c[m] = '\0';
	dbg ("tmp_fl=%s",tmp_fl);
	dbg ("path_fl_c=%s",path_fl_c);
	std::string filename(tmp_fl);
	pos = filename.find("_Infrared");
	filename.copy (tmp_fl, pos, 0);
	tmp_fl[pos]='\0';
	base_fl = tmp_fl;
	n=base_fl.find("convert_");
	n+=sizeof("convert_") - 1;
	n=base_fl.copy(tmp_fl,sizeof(tmp_fl),n);
	tmp_fl[n]=0;
	base_fl = tmp_fl;
	path_fl = path_fl_c; 
	pos=filename.find("2_");
	n = filename.copy(tmp_fl, SEC_LEN, pos+2);
	tmp_fl[n]='\0';
	dbg ("n=%d,tmp_fl=%s",n,tmp_fl);
	std::string sec_fl_string(tmp_fl);
#if 0
	sec_fl = std::stol(sec_fl_string) % 100;
#else
	sec_fl = 0;
#endif
}
#endif

bool ms_cam::ms_cam_skip (time_t f_time, bool color)
{
	while ((ms_time < f_time) && (ms_time < ms_duration)) {
		frames = pipe.wait_for_frames();
		rs2::playback playback = device.as<rs2::playback>();
		ms_time = playback.get_position();
		ms_time /= 1000000000;
	}
	if (!color)
		cam_frame = frames.get_infrared_frame();
	else
		cam_frame = frames.get_color_frame();
	dbg ("ms_time=%d",ms_time);
	return true;
}

