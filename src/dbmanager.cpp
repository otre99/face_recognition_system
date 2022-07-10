#include "dbmanager.h"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <filesystem>

double CalcL2Norm(const vector<float> &x)
{
    double r=0;
    for (auto xi : x){
        r += xi*xi;
    }
    return sqrt(r);
}
void Normalize(vector<float> &x){

    auto r = CalcL2Norm(x);
    for (auto &xi : x){
        xi/=r;
    }
}

DBManager::~DBManager()
{
    Close();
}


bool DBManager::Open(const string &mpath, bool write, int32_t embedding_len)
{
    if (write==false){
        iofile_.open(mpath, ios::binary|ios::in);
        if (!iofile_){
            cerr << "Error opening file " << mpath << " for read "<< endl;
            return false;
        }
        iofile_.read(reinterpret_cast<char*>(&embedding_len_), sizeof(int32_t));
        if ( embedding_len_<=0 ){
            cerr << "Wrong 'embedding_len' in " << mpath << " file. embedding_len = " << embedding_len << endl;
            return false;
        }
        const size_t RECORD_SIZE = 32 + sizeof(float)*embedding_len_;
        iofile_.seekg(0, ios::end);

        size_t data_size = iofile_.tellg();
        data_size -= sizeof(int32_t);

        if ( data_size%RECORD_SIZE != 0 ){
            cerr << "Wrong size in file " << mpath << endl;
            return false;
        }

        size_t n = data_size/RECORD_SIZE;
        embedding_data_.clear();
        Data dd;
        dd.embedding.resize(embedding_len_);
        cout << "Found " << n << " entries in file " << mpath << endl;
        for (int i=0; i<n; ++i){
            iofile_.read(reinterpret_cast<char*>(dd.face_id), 32);
            iofile_.read(reinterpret_cast<char*>(dd.embedding.data()), sizeof(float)*embedding_len_);
            embedding_data_.push_back(dd);
        }
        return true;
    }


    if (embedding_len <= 0){
        cerr << "Wrong value of 'embedding_len' " << endl;
        return false;
    }

    if ( filesystem::exists(mpath) ){
        cout << " File '" << mpath << "' already exists, so the new data will be added to the end" << endl;
        iofile_.open(mpath);
        int32_t n;
        iofile_.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
        iofile_.close();
        if (n != embedding_len){
            cerr << "ERROR: Impossible to add new data to the file of '" << mpath
                 << "' because the length of the new face embedding differs from those in the file" << endl;
            iofile_.close();
            return false;
        }
        iofile_.open(mpath, ios::binary|ios::ate);
    } else {
        cout << "Creating dataset '" << mpath << "'" << endl;
        embedding_len_ = embedding_len;
        iofile_.open(mpath,ios::binary|ios::out);
        if (!iofile_){
            cerr << "Error creating file " << mpath << " for write "<< endl;
            return false;
        }
        iofile_.write(reinterpret_cast<char*>(&embedding_len_), sizeof(int32_t));
    }
    return true;
}

void DBManager::Close() {
    if (iofile_.is_open())
        iofile_.close();
}

bool  DBManager::AddData(const char *faceId, const vector<float> &data, bool is_normalized)
{
    if (!iofile_.is_open()){
        cerr << " Not output file!" << endl;
        return false;
    }

    if ( strlen(faceId)>KEY_SIZE-1){
        cerr << " FaceId lenght must be less than " << KEY_SIZE << endl;
        return false;
    }

    if ( data.size() != embedding_len_  ){
        cerr << " Wrong size of embedding data " << endl;
        return false;
    }

    char key[32];
    strcpy(key, faceId);
    iofile_.write(key,KEY_SIZE);
    if (is_normalized) {
        iofile_.write(reinterpret_cast<const char*>(data.data()), sizeof(float)*embedding_len_);
    } else {
        auto cpy = data;
        Normalize(cpy);
        iofile_.write(reinterpret_cast<const char*>(cpy.data()), sizeof(float)*embedding_len_);
    }
    return true;
}


