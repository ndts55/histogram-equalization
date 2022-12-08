#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

class PGM {
private:
    int width, height, max;
    string img;
public:
    explicit PGM(const string &path) {
        width = 0;
        height = 0;
        img = "";


        ifstream ifs(path);
        stringstream ss;
        string line;
        // Version
        getline(ifs, line);
        cout << "Version: " << line << endl;

        // Dimensions
        ss << ifs.rdbuf();
        ss >> width >> height;
        cout << "Width: " << width << "\tHeight: " << height << endl;

        // Max
        ss >> max;
        cout << "Max: " << max << endl;

        // Image Data
        img = ss.str();
    }

    string Img() { return img; }
};
