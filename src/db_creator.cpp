#include <iostream>
#include "dbmanager.h"
#include "face_detection.h"
#include "io_utils.h"

using namespace std;

int main(int argc, char *argv[])
{
    DBManager dbmanager;
    FaceDetection faceDet;
    {
        auto data = LoadJSon(argv[1]);
        const string mpath = data["db"].get<string>();
        const int embedding_len = data["embedding_len"].get<int>();
        if (!dbmanager.Open(mpath,true,embedding_len)){
            return -1;
        }
        faceDet.Init(data["face_detection"]);
    }


    return 0;
}
