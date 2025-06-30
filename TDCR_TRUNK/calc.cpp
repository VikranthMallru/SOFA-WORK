#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

int main() {
    vector<double> delta_y = {0,21.60, 20.30, 19.09, 17.94, 16.86, 15.85, 14.90, 14.01, 13.17, 12.38, 11.63, 10.94, 10.28, 9.66,9.08,8.54,8.03,7.54,7.09,6.67,6.27,5.89,5.54,5.20,4.89,4.60,4.32,4.06,3.82,3.59};
    vector<double> y_values;
    vector<double> x_values;
    const double x = 24.76;
    const double delta = 0;
    // Calculate cumulative y values
    double y = 0.0;
    for (double dy : delta_y) {
        y += dy;
        // cout<<" "<<y;
        y_values.push_back(y);
    }
    cout<<'\n';
    // Output in the requested format
    cout << fixed << setprecision(3);
    cout << "[\n";
    for (size_t i = 0; i < y_values.size(); ++i) {
        cout << "[" << ((i%2) ? x+delta : x-delta) << ", " << y_values[i] << ", " << (y_values[i]+40.4)/16.2<< "]";
        if (i != y_values.size() - 1) cout << ",";
        cout << endl;
    }
    cout << "]" << endl;

    return 0;
}

//     vector<double> delta_y = {21.60, 20.30, 19.09, 17.94, 16.86, 15.85, 14.90, 14.01, 13.17, 12.38, 11.63, 10.94, 10.28, 9.66,9.08,8.54,8.03,7.54,7.09,6.67,6.27,5.89,5.54,5.20,4.89,4.60,4.32,4.06,3.82,3.59};
