#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

int main() {
    // Input two points in 3D
    double x1=3, y1=0, z1=19.75, x2=16, y2 = 272.68, z2 = 19.75;


    // Direction vector
    double dx = x2 - x1;
    double dy = y2 - y1;
    double dz = z2 - z1;

    vector<double> delta_y = {11.09,18.86,17.95,17.05,16.16,15.35,14.54,13.81,13.09,12.43,11.78,11.19,10.60,10.07,9.54,9.06,8.59,8.16,7.73,7.34,6.95,6.61,6.26,5.95,2.52};
    vector<double> y_values;
    double t = 0.0;
    double total_length = 0.0;
    for (double d : delta_y) total_length += d;
    cout << "total length: " << total_length - y2 << ", number of elements in delta_y: " << delta_y.size() << '\n';

    // Output in the requested format
    cout << fixed << setprecision(3);
    cout << "[\n";
    double cum_dist = 0.0;
    for (size_t i = 0; i < delta_y.size(); ++i) {
        cum_dist += delta_y[i];
        double param = cum_dist / total_length; // normalized parameter along the line [0,1]
        double px = x1 + dx * param;
        double py = y1 + dy * param;
        double pz = z1 + dz * param;
        cout << "[" << px << ", " << py << ", " << pz << "]";
        if (i != delta_y.size() - 1) cout << ",";
        cout << endl;
    }
    cout << "]" << endl;

    return 0;
}
