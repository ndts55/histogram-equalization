#include <iostream>
#include "imageio.h"

using namespace std;

int main() {
    cout << "Contrast MPI" << endl;
    PGM pgm("in.pgm");
    cout << pgm.Img().size() << endl;
}
