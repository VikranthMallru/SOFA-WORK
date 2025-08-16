#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

int main() {
    // Input two points in 3D
    double x1=1.5, y1=0, z1=22, x2=17, y2=250, z2=22;

    // Direction vector
    double dx = x2 - x1;
    double dy = y2 - y1;
    double dz = z2 - z1;

    vector<double> delta_y = {1.57,10.41, 22.23, 19.56, 19.56, 17.21, 17.21, 15.15, 15.15, 13.33, 13.33, 11.73, 11.73, 10.32, 10.32, 9.08, 9.08, 7.99, 7.99, 7.05};

    double total_length = 0.0;
    for (double d : delta_y) total_length += d;
    cout << "total length: " << total_length << ", number of elements in delta_y: " << delta_y.size() << '\n';

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
        cout << "[" << px << ", " << py + 1 << ", " << pz << "]";
        if (i != delta_y.size() - 1) cout << ",";
        cout << endl;
    }
    cout << "]" << endl;

    return 0;
}
