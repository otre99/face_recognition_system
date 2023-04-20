# face_recognition_system
Face Recognition (under construction)

# Compile

The project depends on OpenCV 4.* (Tested with OpenCV 4.5.5)
```
mkdir build
cd build 
cmake ../src && make -j4
```

# Run simple example:

*   Download and unzip sample [data](https://drive.google.com/file/d/17jTx4Zhg1McQJGw5xEs312sAuFSY4ZeS/view?usp=sharing) 

```
    unzip host_storage.zip
```

*   Create face emdbbing from images
```
./build/dbcreator host_storage/config/config_rbf320.json
```

This will create a dataset file (`host_storage/db.dataset`) that containts the corresponding face embeddings

*   Run face recognition program 
```
./build/face_recognition_system host_storage/config/config_rbf320.json 
```

![](assets/demo.GIF)

The program logs can be found in the `host_storage/captures` folder.


You can play with your web can if you change set `"video_input": "/dev/video0"` in file [config_rbf320.json](./config/config_rbf320.json). 
Also you can adjust the others parameters 


