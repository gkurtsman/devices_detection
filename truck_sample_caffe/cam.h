#ifndef __CAM_H__
#define __CAM_H__
#endif

#define DEBUG_MSR
// #undef DEBUG_MSR

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#ifdef DEBUG_MSR
#define dbg(format, ...) \
do { \
	printf("%s#%d:" format "\n", __FILENAME__,__LINE__, __VA_ARGS__); \
} while (0)
#else
#define dbg(cinfo, format, ...)
#endif

#define SEC_LEN 11

struct _config
{
public:
	bool usemean;
	bool brightness;
	bool color;
	bool crop;
	_config (char*);
	_config ();
};

class ms_cam
{
private:
	time_t ms_duration;
	rs2::context ctx;
	rs2::config cfg;
	rs2::pipeline_profile profile;
	void fl_disassemble();  // extract seconds and milliseconds from file name

public:
	time_t sec_fl;	// seconds in file name
	time_t msec_fl;	// milliseconds in file name
	time_t ms_time;
	std::string base_fl; // base name of file
	std::string path_fl; // path to file
	std::string png_fl;
	std::string bag_fl;
	rs2::frameset frames;
	rs2::pipeline pipe;
	rs2::device device;
	rs2::frame cam_frame;

	ms_cam (std::string png_fl);
	bool ms_stream_step (void);	/* to get one frames collection using
								  pipe.wait_for_frames() library function */
	bool ms_cam_skip (time_t f_time, bool color); /* seek frame(s) with
													 camera parameters time value */
};
