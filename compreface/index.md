# Compreface


### 1. usage

- **Step 1.** Install and run CompreFace using our [Getting Started guide](https://gitee.com/mirrors/compreface/blob/master/README.md#getting-started-with-compreface)
- **Step 2.**  sign up for the system and log in.
- **Step 3.** Create an application (left section) using the "Create" link at the bottom of the page. An application is where you can create and manage your Face Collections.
- **Step 4.** Enter your application by clicking on its name. Here you will have two options: you can either add new users and manage their access roles or create new [Face Services](https://gitee.com/mirrors/compreface/blob/master/docs/Face-services-and-plugins.md).
- **Step 5.** To recognize subjects among the known subjects, you need to create Face Recognition Service. After creating a new Face Service, you will see it in the Services List with an appropriate name and API key. After this step, you can look at our [demos](https://gitee.com/mirrors/compreface/blob/master/docs/How-to-Use-CompreFace.md#demos).
- **Step 6.** To add known subjects to your Face Collection of Face Recognition Service, you can use REST API. Once you’ve [uploaded all known faces](https://gitee.com/mirrors/compreface/blob/master/docs/Rest-API-description.md#add-an-example-of-a-subject), you can test the collection using [REST API](https://gitee.com/mirrors/compreface/blob/master/docs/Rest-API-description.md#recognize-faces-from-a-given-image) or the TEST page. We recommend that you use an image size no higher than 5MB, as it could slow down the request process. The supported image formats include JPEG/PNG/JPG/ICO/BMP/GIF/TIF/TIFF.
- **Step 7.** Upload your photo and let our open-source face recognition system match the image against the Face Collection. If you use a UI for face recognition,  you will see the original picture with marks near every face. If you use [REST API](https://gitee.com/mirrors/compreface/blob/master/docs/Rest-API-description.md#recognize-faces-from-a-given-image), you will receive a response in JSON format.
- demos: tutorial_demo.html： 上传人脸数据文件；webcam_demo.html: 调用实时摄像头实现实时人脸识别。

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210913213134551.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210913213315241.png)



![CompreFace architecture diagram](https://gitee.com/github-25970295/blogpictureV2/raw/master/107855144-5db83580-6e29-11eb-993a-46cdc0c82812.png)

### 2. component

- **Balancer+UI:**   compreface-fe, runs Nginx, proxies user requests to admin and api servers;
- **Admin-server:** compreface-admin, responsible for all operations that are done on UI, connects to PostgreSQL database to store the data
- **Api servers:** compreface-api, handle all user API calls, `face recognition, face detection, and face verification`, the data synchronization is implemented via `postgreSQL notifications`.
- **Embedding Server:** compreface-core, responsible for running networks, which are stateless servers.
- **PostgreSQL:** password, url, user;

### 3. custom buids

>a trade off between the face recognition accuracy, max throughput of the system and even hardware support.

```yaml
compreface-core:
    image: ${registry}compreface-core:${CORE_VERSION}
    container_name: "compreface-core"
    ports:
      - "3300:3000"
    runtime: nvidia
    build:
      context: ../embedding-calculator
      args:
        - FACE_DETECTION_PLUGIN=insightface.FaceDetector@retinaface_r50_v1
        - CALCULATION_PLUGIN=insightface.Calculator@arcface_r100_v1
        - EXTRA_PLUGINS=insightface.LandmarksDetector,insightface.GenderDetector,insightface.AgeDetector
        - BASE_IMAGE=${registry}compreface-core-base:base-cuda100-py37
        - GPU_IDX=0
    environment:
      - ML_PORT=3000
```

### 4. Face Services

* Face recognition service (Face identification); Face detection service; Face verification service; Age detection plugin; Gender detection plugin; Landmarks detection plugin; Calculator plugin

```shell
curl  -X POST "http://localhost:8000/api/v1/recognition/recognize?plugins=age,gender,landmarks" \
-H "Content-Type: multipart/form-data" \
-H "x-api-key: <faces_recognition_api_key>" \
-F file=<local_file>
```

+ [Face Recognition Service Endpoints](#face-recognition-service-endpoints)
  + [Add a Subject](#add-a-subject)
  + [Rename a Subject](#rename-a-subject)
  + [Delete a Subject](#delete-a-subject)
  + [Delete All Subjects](#delete-all-subjects)
  + [List Subjects](#list-subjects)
  + [Add an Example of a Subject](#add-an-example-of-a-subject)
  + [Recognize Faces from a Given Image](#recognize-faces-from-a-given-image)
  + [List of All Saved Examples of the Subject](#list-of-all-saved-examples-of-the-subject)
  + [Delete All Examples of the Subject by Name](#delete-all-examples-of-the-subject-by-name)
  + [Delete an Example of the Subject by ID](#delete-an-example-of-the-subject-by-id)
  + [Direct Download an Image example of the Subject by ID](#direct-download-an-image-example-of-the-subject-by-id)
  + [Download an Image example of the Subject by ID](#download-an-image-example-of-the-subject-by-id)
  + [Verify Faces from a Given Image](#verify-faces-from-a-given-image)
+ [Face Detection Service](#face-detection-service)
+ [Face Verification Service](#face-verification-service)
+ [Base64 Support](#base64-support)

---

> 作者: liudongdong  
> URL: https://liudongdong1.github.io/compreface/  

