#include <bits/stdc++.h>
using namespace std;


auto printVec(vector<float> v){
    for(auto u:v){
        cout<<setprecision(4)<<u<<' ';
    }
    cout<<'\n';
}
int main()
{
    vector<float> y = {7.36, 6.91, 6.49, 6.09, 5.72, 5.37, 5.05, 4.74, 4.45, 4.18, 3.92, 3.68, 3.46, 3.25, 3.05, 2.86, 2.69, 2.52, 2.37};
    float sum = 0;
    vector<float> ans;
    for(auto u:y){
        if(u!=y[0]){
            cout<<sum+(u/(2.0))<<' ';
            ans.push_back(sum+(u/(2.0)));
        }
        sum += u;
    }
    cout<<'\n';
    vector<float> cable1x;vector<float> cable2x;
    for(auto u: ans){
        cable1x.push_back((u+13.929)/9.705);
        cable2x.push_back((238.967-u)/9.705);
    }
    cout<<"Cable1:\n";
    printVec(cable1x);
    cout<<"Cable2:\n";
    printVec(cable2x);
    return 0;
}