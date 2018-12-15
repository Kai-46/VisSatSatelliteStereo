#include "ceres/ceres.h"
#include "glog/logging.h"

#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <limits>

using std::cout;
using std::vector;
using std::string;
using std::istringstream;
using std::ifstream;
using std::ofstream;
using std::stod;
using std::runtime_error;
using std::setprecision;

using ceres::AutoDiffCostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

#define INIT_VAL -1e10

typedef std::numeric_limits<double> dbl;

struct RPCCamera {
    RPCCamera() {}
    
    double row_numera[20] = {INIT_VAL};
    double row_denomi[20] = {INIT_VAL};
    double col_numera[20] = {INIT_VAL};
    double col_denomi[20] = {INIT_VAL};
    double lat_off = INIT_VAL, lat_scale = INIT_VAL;
    double lon_off = INIT_VAL, lon_scale = INIT_VAL;
    double alt_off = INIT_VAL, alt_scale = INIT_VAL;
    double row_off = INIT_VAL, row_scale = INIT_VAL;
    double col_off = INIT_VAL, col_scale = INIT_VAL;
};

struct Observation {
    Observation(RPCCamera* cam): cam(cam) {}
    
    RPCCamera* cam = NULL;
    double row = INIT_VAL;
    double col = INIT_VAL;
};

struct ReprojResidual {
    ReprojResidual(Observation* pixel): pixel(pixel) {}
    
    template <typename T>
    bool operator() (const T* const lat, const T* const lon, const T* const alt,
                     T* residuals) const {
        RPCCamera& cam = *(this->pixel->cam);
        
        T lat_normed = (lat[0] - T(cam.lat_off)) / T(cam.lat_scale);
        T lon_normed = (lon[0] - T(cam.lon_off)) / T(cam.lon_scale);
        T alt_normed = (alt[0] - T(cam.alt_off)) / T(cam.alt_scale);
        
        T row_numera = this->apply_poly(cam.row_numera, lat_normed, lon_normed, alt_normed);
        T row_denomi = this->apply_poly(cam.row_denomi, lat_normed, lon_normed, alt_normed);
        
        T predict_row = row_numera / row_denomi * T(cam.row_scale) + T(cam.row_off);
        
        T col_numera = this->apply_poly(cam.col_numera, lat_normed, lon_normed, alt_normed);
        T col_denomi = this->apply_poly(cam.col_denomi, lat_normed, lon_normed, alt_normed);
        
        T predict_col = col_numera / col_denomi * T(cam.col_scale) + T(cam.col_off);
        
        residuals[0] = predict_row - T(this->pixel->row);
        residuals[1] = predict_col - T(this->pixel->col);
        return true;
    }
    
private:
    template <typename T>
    T apply_poly(const double* const poly, T x, T y, T z) const {
        T out = T(poly[0]);
        out += poly[1]*y + poly[2]*x + poly[3]*z;
        out += poly[4]*y*x + poly[5]*y*z +poly[6]*x*z;
        out += poly[7]*y*y + poly[8]*x*x + poly[9]*z*z;
        out += poly[10]*x*y*z;
        out += poly[11]*y*y*y;
        out += poly[12]*y*x*x + poly[13]*y*z*z + poly[14]*y*y*x;
        out += poly[15]*x*x*x;
        out += poly[16]*x*z*z + poly[17]*y*y*z + poly[18]*x*x*z;
        out += poly[19]*z*z*z;
        
        return out;
    }
    
private:
    Observation* pixel = NULL;
};


double triangulate(vector<Observation*>& pixels, vector<double>& initial, vector <double>& final) {
    double lat = initial[0];
    double lon = initial[1];
    double alt = initial[2];
    
    Problem problem;
    for (int i=0; i < pixels.size(); ++i) {
        problem.AddResidualBlock(
                                 new AutoDiffCostFunction<ReprojResidual, 2, 1, 1, 1>(new ReprojResidual(pixels[i])),
                                 NULL, &lat, &lon, &alt);
    }
    
    Solver::Options options;
    options.max_num_iterations = 25;
    // options.function_tolerance = 1e-10;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    
    Solver::Summary summary;

//    // for debug
//    double init_error = 0.;
//    problem.Evaluate(Problem::EvaluateOptions(), &init_error, NULL, NULL, NULL);
//    // note that ceres add 1/2 before the cost function
//    init_error = sqrt(init_error * 2 / pixels.size());

    Solve(options, &problem, &summary);
    cout << summary.BriefReport() << "\n";
    
    cout << "\nInitial Point: (" << initial[0] << "," << initial[1] << "," << initial[2] << ")\n";
    cout << "Final Point:  (" << lat << "," << lon << "," << alt << ")\n";
    
    assert (final.size() == 3);
    final[0] = lat;
    final[1] = lon;
    final[2] = alt;
    
    double final_error = 0.0;
    problem.Evaluate(Problem::EvaluateOptions(), &final_error, NULL, NULL, NULL);
    // note that ceres add 1/2 before the cost function
    final_error = sqrt(final_error * 2 / pixels.size());

    // cout << "Inside ceres: init_error: " << init_error << " pixels, " << "final_error: " << final_error << " pixels\n";
    return final_error;
}


void readline(const string& line, Observation* pixel) {
    istringstream iss(line);
    
    iss >> pixel->col;
    iss >> pixel->row;
    
    for (int i = 0; i < 20; ++i) {
        iss >> pixel->cam->row_numera[i];
    }
    
    for (int i = 0; i < 20; ++i) {
        iss >> pixel->cam->row_denomi[i];
    }
    
    for (int i = 0; i < 20; ++i) {
        iss >> pixel->cam->col_numera[i];
    }
    
    for (int i = 0; i < 20; ++i) {
        iss >> pixel->cam->col_denomi[i];
    }
    
    iss >> pixel->cam->lat_off >> pixel->cam->lat_scale;
    iss >> pixel->cam->lon_off >> pixel->cam->lon_scale;
    iss >> pixel->cam->alt_off >> pixel->cam->alt_scale;
    iss >> pixel->cam->row_off >> pixel->cam->row_scale;
    iss >> pixel->cam->col_off >> pixel->cam->col_scale;
}

// read task file; each file is a feature track
// each row represents an observation
void execute_task(string in_file, string out_file) {
    ifstream infile;
    infile.open(in_file);
    
    if (!infile) {
        throw runtime_error("unable to open " + in_file);
    }
    
    string line;
    
    vector<RPCCamera*> cams;
    vector<Observation*> pixels;
    
    vector<double> initial(3);
    getline(infile, line);
    istringstream iss(line);
    iss >> initial[0] >> initial[1] >> initial[2];
    
    getline(infile, line); // re-projection error
    double init_error = stod(line);
    
    vector<string> content;
    while (getline(infile, line)) {
        cams.push_back(new RPCCamera());
        pixels.push_back(new Observation(cams.back()));
        readline(line, pixels.back());
        
        content.push_back(line);
    }
    infile.close();
    
    vector<double> final(3);
    
    double final_error = triangulate(pixels, initial, final);
    
    cout << "\ninitial re-projection error (pixels): " << init_error << "\n";
    cout << "final re-projection error (pixels): " << final_error << "\n\n";
    
    ofstream outfile;
    outfile.open(out_file);
    if (!outfile) {
        throw runtime_error("unable to open " + out_file);
    }
    
    // c++ double has 17 decimal digit precision
    outfile << setprecision(dbl::max_digits10);
    outfile << final[0] << " " << final[1] << " " << final[2] << "\n";
    outfile << final_error << "\n";
    
    for (int i = 0; i < content.size(); ++i) {
        outfile << content[i] << "\n";
    }
    outfile.close();
    
    // free memory
    for (int i = 0; i < cams.size(); ++i) {
        delete cams[i];
    }
    
    for (int i = 0; i < pixels.size(); ++i) {
        delete pixels[i];
    }
}


int main(int argc, char** argv) {
    // program name, in_file, out_file
    assert(argc == 3);
    google::InitGoogleLogging(argv[0]);
    string in_file = string(argv[1]);
    string out_file = string(argv[2]);
    
    // string in_file = "/Users/kai/satellite_stereo/example_triangulate/task.txt";
    // string out_file = "/Users/kai/satellite_stereo/example_triangulate/task_result.txt";
    // c++ double has 17 decimal digit precision
    cout << setprecision(dbl::max_digits10);
    execute_task(in_file, out_file);
    return 0;
}
