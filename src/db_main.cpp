#include <iostream>
#include "dbmanager.h"
#include "face_detection.h"
#include "io_utils.h"

using namespace std;

int main(int argc, char *argv[])
{
    FaceDetection faceDet;
    DBManager dbmanager;
    {
        auto data = LoadJSon(argv[1]);
    }


    return 0;
}
