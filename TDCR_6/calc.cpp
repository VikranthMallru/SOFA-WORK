#include<bits/stdc++.h>
using namespace std;

void printVec(vector<double> v){
    for(auto u:v){
        cout<<u<<'\n';
    }
}
// Function to interpolate points between two 3D points based on given delta y's
vector<vector<double>> interpolatePoints(
    const vector<double>& deltaYs,
    const array<double, 3>& point1,
    const array<double, 3>& point2)
{
    vector<vector<double>> result;
    double y1 = point1[1], y2 = point2[1];
    double x1 = point1[0], x2 = point2[0];
    double z1 = point1[2], z2 = point2[2];
    double y = 0;
    // printVec(deltaYs);
    for (double dy : deltaYs) {
        // Linear interpolation factor
        double t = (dy - y1) / (y2 - y1);
        double x = x1 + t * (x2 - x1);
        double z = z1 + t * (z2 - z1);
        result.push_back({x, dy, z-0.5});
    }
    return result;
}
int main() {
    // Example usage
    vector<double> deltaYs = {23.5,21.86,20.33,18.90,17.58,16.35,15.20,14.14,13.15,12.23,11.37,10.58,9.84,9.15,8.51,7.91,7.36,6.84,6.36,5.92,5.50,5.12,4.76}; // Change as needed
    vector<double> DY;
    float sum = 10;
    for(auto i = 0 ; i < deltaYs.size();i++){
        DY.push_back(sum + deltaYs[i]/2);
        sum += deltaYs[i];
    }
    // printVec(DY);
    array<double, 3> point1 = {(60.52)/2,0,5.26};
    array<double, 3> point2 = {(60.52)/2, 290 , ((60.52-30)/2)+3};

    vector<vector<double>> points = interpolatePoints(DY, point1, point2);

    cout << "[\n";
    // Printing the interpolated y's (not the original delta y's) in the 2nd position
    for (size_t i = 0; i < points.size(); ++i) {
        cout << "[";
        for (size_t j = 0; j < points[i].size(); ++j) {
            cout << setprecision(5)<<points[i][j];
            if (j + 1 < points[i].size()) cout << ", ";
        }
        cout << "]";
        if (i + 1 < points.size()) cout << "," << endl;
    }
    cout << "\n]" << endl;

    return 0;
}