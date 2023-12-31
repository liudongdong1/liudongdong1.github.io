# CompreFaceLearning


## 1. Configuration

* `registry` - this is the docker hub registry. For release and pre-build images, it should be set to `exadel/` value
* `postgres_password` - password for Postgres database. It should be changed for production systems from the default value.
* `postgres_domain` - the domain where Postgres database is run
* `postgres_port` - Postgres database port
* `enable_email_server` - if true, it will enable email verification for users. You should set email_host, email_username, and email_password variables for the correct work.
* `email_host` - a host of the email provider. It should be set if `enable_email_server` variable is true
* `email_username` - a username of email provider for authentication. It should be set if `enable_email_server` variable is true
* `email_password` - a password of email provider for authentication. It should be set if `enable_email_server` variable is true
* `email_from` - this value will see users in `from` fields when they receive emails from CompreFace. Corresponds to `From` field in rfc2822. Optional, if not set, then `email_username` will be used instead
* `save_images_to_db` - should the CompreFace save photos to the database. Be careful, [migrations](Face-data-migration.md) could be run only if this value is `true`
* `compreface_api_java_options` - java options of compreface-api container
* `compreface_admin_java_options` - java options of compreface-admin container
* `ADMIN_VERSION` - docker image tag of compreface-admin container
* `API_VERSION` - docker image tag of compreface-api container
* `FE_VERSION` - docker image tag of compreface-fe container
* `CORE_VERSION` - docker image tag of compreface-core container

| Custom-build               | Base library                                              | CPU                     | GPU                 | Face detection model / accuracy on [WIDER Face (Hard)](https://paperswithcode.com/sota/face-detection-on-wider-face-hard) | Face recognition model / accuracy on [LFW](https://paperswithcode.com/sota/face-verification-on-labeled-faces-in-the) | Age and gender detection                                     | Comment                                        |
| -------------------------- | --------------------------------------------------------- | ----------------------- | ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------- |
| FaceNet (default)          | [FaceNet](https://github.com/davidsandberg/facenet)       | x86 (AVX instructions)  | not supported       | MTCNN / 80.9%                                                | FaceNet (20180402-114759) / 99.63%                           | Custom, the model is taken [here](https://github.com/GilLevi/AgeGenderDeepLearning) | For general purposes. Support CPU without AVX2 |
| Mobilenet                  | [InsightFace](https://github.com/deepinsight/insightface) | x86 (AVX2 instructions) | not supported       | RetinaFace-MobileNet0.25 / 82.5%                             | MobileFaceNet,ArcFace / 99.50%                               | InsightFace                                                  | The fastest model among CPU only models        |
| Mobilenet-gpu              | [InsightFace](https://github.com/deepinsight/insightface) | x86 (AVX2 instructions) | GPU (CUDA required) | RetinaFace-MobileNet0.25 / 82.5%                             | MobileFaceNet,ArcFace / 99.50%                               | InsightFace                                                  | The fastest model                              |
| SubCenter-ArcFace-r100     | [InsightFace](https://github.com/deepinsight/insightface) | x86 (AVX2 instructions) | not supported       | retinaface_r50_v1 / 91.4%                                    | arcface-r100-msfdrop75 / 99.80%                              | InsightFace                                                  | The most accurate model, but the most slow     |
| SubCenter-ArcFace-r100-gpu | [InsightFace](https://github.com/deepinsight/insightface) | x86 (AVX2 instructions) | GPU (CUDA required) | retinaface_r50_v1 / 91.4%                                    | arcface-r100-msfdrop75 / 99.80%                              | InsightFace                                                  | The most accurate model                        |

## 2. Run

- run custom-builds

```dockerfile
docker-compose up -d
#http://localhost:8000/login 
docker-compose start
docker-compose stop
```

- build own custom-build

1. Upload your model to Google Drive and add it to one the following files into the `Calculator` class:
   - embedding-calculator/src/services/facescan/plugins/facenet/facenet.py
   - embedding-calculator/src/services/facescan/plugins/insightface/insightface.py

2. Take the `docker-compose` file from `/dev` folder as a template
3. Specify new model name in build arguments. For more information look at [this documentation](https://github.
   com/exadel-inc/CompreFace/tree/master/embedding-calculator#run-service). E.g. here is a part of `docker-compose` file for building with custom model with GPU support.

**Step 1.** Install and run CompreFace using our [Getting Started guide](../README.md#getting-started-with-compreface)

**Step 2.** You need to sign up for the system and log in into the account you’ve just created or use the one you already have. After that, the system redirects you to the main page.

**Step 3.** Create an application (left section) using the "Create" link at the bottom of the page. An application is where you can create and manage your Face Collections.

**Step 4.** Enter your application by clicking on its name. Here you will have two options: you can either add new users and manage their access roles or create new [Face Services](Face-services-and-plugins.md).

**Step 5.** To recognize subjects among the known subjects, you need to create Face Recognition Service. After creating a new Face Service, you will see it in the Services List with an appropriate name and API key. After this step, you can look at our [demos]

**Step 6.** To add known subjects to your Face Collection of Face Recognition Service, you can use REST API. 
Once you’ve [uploaded all known faces](Rest-API-description.md#add-an-example-of-a-subject),you can test the collection using [REST API](Rest-API-description.md#recognize-faces-from-a-given-image) or the TEST page. 
We recommend that you use an image size no higher than 5MB, as it could slow down the request process. The supported image formats include JPEG/PNG/JPG/ICO/BMP/GIF/TIF/TIFF.

**Step 7.** Upload your photo and let our open-source face recognition system match the image against the Face Collection. If you use a UI for face recognition, you will see the original picture with marks near every face. If you use [REST API](Rest-API-description.md#recognize-faces-from-a-given-image), you will receive a response in JSON format.

```js
function saveNewImageToFaceCollection(elem) {
    let subject = encodeURIComponent(document.getElementById("subject").value);
    let apiKey = document.getElementById("apiKey").value;
    let formData = new FormData();
    let photo = elem.files[0];

    formData.append("file", photo);

    fetch('http://localhost:8000/api/v1/recognition/faces/?subject=' + subject,
        {
            method: "POST",
            headers: {
                "x-api-key": apiKey
            },
            body: formData
        }
    ).then(r => r.json()).then(
        function (data) {
            console.log('New example was saved', data);
        })
        .catch(function (error) {
            alert('Request failed: ' + JSON.stringify(error));
        });
}
```

This function sends the image to our server and shows results in a text area:

```js
function recognizeFace(elem) {
    let apiKey = document.getElementById("apiKey").value;
    let formData = new FormData();
    let photo = elem.files[0];

    formData.append("file", photo);

    fetch('http://localhost:8000/api/v1/recognition/recognize',
        {
            method: "POST",
            headers: {
                "x-api-key": apiKey
            },
            body: formData
        }
    ).then(r => r.json()).then(
        function (data) {
            document.getElementById("result").innerHTML = JSON.stringify(data);
        })
        .catch(function (error) {
            alert('Request failed: ' + JSON.stringify(error));
        });
}
```



![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210626200731382.png)

## 3. RestAPI

### Add a Subject

Create a new subject in Face Collection. Creating a subject is an optional step, 
you can [upload an example](#add-an-example-of-a-subject) without an existing subject, and a subject will be created automatically.

```shell
curl -X POST "http://localhost:8000/api/v1/recognition/subjects" \
-H "Content-Type: application/json" \
-H "x-api-key: <service_api_key>" \
-d '{"subject: <subject_name>"}'
```

| Element      | Description | Type   | Required | Notes                                                        |
| ------------ | ----------- | ------ | -------- | ------------------------------------------------------------ |
| Content-Type | header      | string | required | application/json                                             |
| x-api-key    | header      | string | required | api key of the Face recognition service, created by the user |
| subject      | body param  | string | required | is the name of the subject. It can be a person name, but it can be any string |

Response body on success:

```json
{
  "subject": "<subject_name>"
}
```

| Element | Type   | Description                |
| ------- | ------ | -------------------------- |
| subject | string | is the name of the subject |

### Rename a Subject

```since 0.6 version```

Rename existing subject. If a new subject name already exists, 
subjects are merged - all faces from the old subject name are **reassigned** to the subject with the new name, old subject removed.  

```shell
curl -X PUT "http://localhost:8000/api/v1/recognition/subjects/<subject>" \
-H "Content-Type: application/json" \
-H "x-api-key: <service_api_key>" \
-d '{"subject: <subject_name>"}'
```

| Element      | Description | Type   | Required | Notes                                                        |
| ------------ | ----------- | ------ | -------- | ------------------------------------------------------------ |
| Content-Type | header      | string | required | application/json                                             |
| x-api-key    | header      | string | required | api key of the Face recognition service, created by the user |
| subject      | body param  | string | required | is the name of the subject. It can be a person name, but it can be any string |

Response body on success:

```json
{
  "updated": "true|false"
}
```

| Element | Type    | Description       |
| ------- | ------- | ----------------- |
| updated | boolean | failed or success |

### Delete a Subject

```since 0.6 version```

Delete existing subject and all saved faces.

```shell
curl -X DELETE "http://localhost:8000/api/v1/recognition/subjects/<subject>" \
-H "Content-Type: application/json" \
-H "x-api-key: <service_api_key>"
```

| Element      | Description | Type   | Required | Notes                                                        |
| ------------ | ----------- | ------ | -------- | ------------------------------------------------------------ |
| Content-Type | header      | string | required | application/json                                             |
| x-api-key    | header      | string | required | api key of the Face recognition service, created by the user |
| subject      | body param  | string | required | is the name of the subject. It can be a person name, but it can be any string |

Response body on success:

```json
{
  "subject": "<subject_name>"
}
```

| Element | Type   | Description                |
| ------- | ------ | -------------------------- |
| subject | string | is the name of the subject |

### Delete All Subjects

```since 0.6 version```

Delete all existing subjects and all saved faces.

```shell
curl -X DELETE "http://localhost:8000/api/v1/recognition/subjects" \
-H "Content-Type: application/json" \
-H "x-api-key: <service_api_key>"
```

| Element      | Description | Type   | Required | Notes                                                        |
| ------------ | ----------- | ------ | -------- | ------------------------------------------------------------ |
| Content-Type | header      | string | required | application/json                                             |
| x-api-key    | header      | string | required | api key of the Face recognition service, created by the user |

Response body on success:

```json
{
  "deleted": "<count>"
}
```

| Element | Type    | Description                |
| ------- | ------- | -------------------------- |
| deleted | integer | number of deleted subjects |

### List Subjects

```since 0.6 version```

This returns all subject related to Face Collection.  

```shell
curl -X GET "http://localhost:8000/api/v1/recognition/subjects/" \
-H "Content-Type: application/json" \
-H "x-api-key: <service_api_key>"
```

| Element      | Description | Type   | Required | Notes                                                        |
| ------------ | ----------- | ------ | -------- | ------------------------------------------------------------ |
| Content-Type | header      | string | required | application/json                                             |
| x-api-key    | header      | string | required | api key of the Face recognition service, created by the user |

Response body on success:

```json
{
  "subjects": [
    "<subject_name1>",
    "<subject_name2>"
    ]
}
```

| Element  | Type  | Description                             |
| -------- | ----- | --------------------------------------- |
| subjects | array | the list of subjects in Face Collection |

### Add an Example of a Subject

This creates an example of the subject by saving images. You can add as many images as you want to train the system. Image should 
contain only one face.

```http request
curl -X POST "http://localhost:8000/api/v1/recognition/faces?subject=<subject>&det_prob_threshold=<det_prob_threshold>" \
-H "Content-Type: multipart/form-data" \
-H "x-api-key: <service_api_key>" \
-F file=@<local_file> 
```

| Element            | Description | Type   | Required | Notes                                                        |
| ------------------ | ----------- | ------ | -------- | ------------------------------------------------------------ |
| Content-Type       | header      | string | required | multipart/form-data                                          |
| x-api-key          | header      | string | required | api key of the Face recognition service, created by the user |
| subject            | param       | string | required | is the name you assign to the image you save                 |
| det_prob_threshold | param       | string | optional | minimum required confidence that a recognized face is actually a face. Value is between 0.0 and 1.0. |
| file               | body        | image  | required | allowed image formats: jpeg, jpg, ico, png, bmp, gif, tif, tiff, webp. Max size is 5Mb |

Response body on success:  

```json
{
  "image_id": "6b135f5b-a365-4522-b1f1-4c9ac2dd0728",
  "subject": "subject1"
}
```

| Element  | Type   | Description                |
| -------- | ------ | -------------------------- |
| image_id | UUID   | UUID of uploaded image     |
| subject  | string | Subject of the saved image |

### Recognize Faces from a Given Image

To recognize faces from the uploaded image:  

```http request
curl  -X POST "http://localhost:8000/api/v1/recognition/recognize?limit=<limit>&prediction_count=<prediction_count>&det_prob_threshold=<det_prob_threshold>&face_plugins=<face_plugins>&status=<status>" \
-H "Content-Type: multipart/form-data" \
-H "x-api-key: <service_api_key>" \
-F file=<local_file>
```

| Element            | Description | Type    | Required | Notes                                                        |
| ------------------ | ----------- | ------- | -------- | ------------------------------------------------------------ |
| Content-Type       | header      | string  | required | multipart/form-data                                          |
| x-api-key          | header      | string  | required | api key of the Face recognition service, created by the user |
| file               | body        | image   | required | allowed image formats: jpeg, jpg, ico, png, bmp, gif, tif, tiff, webp. Max size is 5Mb |
| limit              | param       | integer | optional | maximum number of faces on the image to be recognized. It recognizes the biggest faces first. Value of 0 represents no limit. Default value: 0 |
| det_prob_threshold | param       | string  | optional | minimum required confidence that a recognized face is actually a face. Value is between 0.0 and 1.0. |
| prediction_count   | param       | integer | optional | maximum number of subject predictions per face. It returns the most similar subjects. Default value: 1 |
| face_plugins       | param       | string  | optional | comma-separated slugs of face plugins. If empty, no additional information is returned. [Learn more](Face-services-and-plugins.md) |
| status             | param       | boolean | optional | if true includes system information like execution_time and plugin_version fields. Default value is false |

Response body on success:

```json
{
  "result" : [ {
    "age" : [ 25, 32 ],
    "gender" : "female",
    "embedding" : [ 9.424854069948196E-4, "...", -0.011415496468544006 ],
    "box" : {
      "probability" : 1.0,
      "x_max" : 1420,
      "y_max" : 1368,
      "x_min" : 548,
      "y_min" : 295
    },
    "landmarks" : [ [ 814, 713 ], [ 1104, 829 ], [ 832, 937 ], [ 704, 1030 ], [ 1017, 1133 ] ],
    "subjects" : [ {
      "similarity" : 0.97858,
      "subject" : "subject1"
    } ],
    "execution_time" : {
      "age" : 28.0,
      "gender" : 26.0,
      "detector" : 117.0,
      "calculator" : 45.0
    }
  } ],
  "plugins_versions" : {
    "age" : "agegender.AgeDetector",
    "gender" : "agegender.GenderDetector",
    "detector" : "facenet.FaceDetector",
    "calculator" : "facenet.Calculator"
  }
}
```

| Element                    | Type    | Description                                                  |
| -------------------------- | ------- | ------------------------------------------------------------ |
| age                        | array   | detected age range. Return only if [age plugin](Face-services-and-plugins.md#face-plugins) is enabled |
| gender                     | string  | detected gender. Return only if [gender plugin](Face-services-and-plugins.md#face-plugins) is enabled |
| embedding                  | array   | face embeddings. Return only if [calculator plugin](Face-services-and-plugins.md#face-plugins) is enabled |
| box                        | object  | list of parameters of the bounding box for this face         |
| probability                | float   | probability that a found face is actually a face             |
| x_max, y_max, x_min, y_min | integer | coordinates of the frame containing the face                 |
| landmarks                  | array   | list of the coordinates of the frame containing the face-landmarks. Return only if [landmarks plugin](Face-services-and-plugins.md#face-plugins) is enabled |
| subjects                   | list    | list of similar subjects with size of <prediction_count> order by similarity |
| similarity                 | float   | similarity that on that image predicted person               |
| subject                    | string  | name of the subject in Face Collection                       |
| execution_time             | object  | execution time of all plugins                                |
| plugins_versions           | object  | contains information about plugin versions                   |


### List of All Saved Examples of the Subject

To retrieve a list of subjects saved in a Face Collection:

```http request
curl -X GET "http://localhost:8000/api/v1/recognition/faces?page=<page>&size=<size>" \
-H "x-api-key: <service_api_key>" \
```

| Element   | Description | Type    | Required | Notes                                                        |
| --------- | ----------- | ------- | -------- | ------------------------------------------------------------ |
| x-api-key | header      | string  | required | api key of the Face recognition service, created by the user |
| page      | param       | integer | optional | page number of examples to return. Can be used for pagination. Default value is 0. Since 0.6 version |
| size      | param       | integer | optional | faces on page (page size). Can be used for pagination. Default value is 20. Since 0.6 version |

Response body on success:

```
{
  "faces": [
    {
      "image_id": <image_id>,
      "subject": <subject>
    },
    ...
  ],
  "page_number": 0,
  "page_size": 10,
  "total_pages": 2,
  "total_elements": 12
}
```

| Element        | Type    | Description                                                  |
| -------------- | ------- | ------------------------------------------------------------ |
| face.image_id  | UUID    | UUID of the face                                             |
| fase.subject   | string  | <subject> of the person, whose picture was saved for this api key |
| page_number    | integer | page number                                                  |
| page_size      | integer | **requested** page size                                      |
| total_pages    | integer | total pages                                                  |
| total_elements | integer | total faces                                                  |


### Delete All Examples of the Subject by Name

To delete all image examples of the <subject>:

```http request
curl -X DELETE "http://localhost:8000/api/v1/recognition/faces?subject=<subject>" \
-H "x-api-key: <service_api_key>"
```

| Element   | Description | Type   | Required | Notes                                                        |
| --------- | ----------- | ------ | -------- | ------------------------------------------------------------ |
| x-api-key | header      | string | required | api key of the Face recognition service, created by the user |
| subject   | param       | string | optional | is the name subject. If this parameter is absent, all faces in Face Collection will be removed |

Response body on success:

```
{
    "deleted": <count>
}
```

| Element | Type    | Description             |
| ------- | ------- | ----------------------- |
| count   | integer | Number of deleted faces |



### Delete an Example of the Subject by ID

Endpoint to delete an image by ID. If no image found by id - 404.

```http request
curl -X DELETE "http://localhost:8000/api/v1/recognition/faces/<image_id>" \
-H "x-api-key: <service_api_key>"
```

| Element   | Description | Type   | Required | Notes                                                        |
| --------- | ----------- | ------ | -------- | ------------------------------------------------------------ |
| x-api-key | header      | string | required | api key of the Face recognition service, created by the user |
| image_id  | variable    | UUID   | required | UUID of the removing face                                    |

Response body on success:

```
{
  "image_id": <image_id>,
  "subject": <subject>
}
```

| Element  | Type   | Description                                                  |
| -------- | ------ | ------------------------------------------------------------ |
| image_id | UUID   | UUID of the removed face                                     |
| subject  | string | <subject> of the person, whose picture was saved for this api key |


### Direct Download an Image example of the Subject by ID

```since 0.6 version```

You can paste this URL into the <img> html tag to show the image.

```http request
curl -X GET "http://localhost:8000/api/v1/static/<service_api_key>/images/<image_id>"
```

| Element         | Description | Type   | Required | Notes                                                        |
| --------------- | ----------- | ------ | -------- | ------------------------------------------------------------ |
| service_api_key | variable    | string | required | api key of the Face recognition service, created by the user |
| image_id        | variable    | UUID   | required | UUID of the image to download                                |

Response body is binary image. Empty bytes if image not found.


### Download an Image example of the Subject by ID

```since 0.6 version```

To download an image example of the Subject by ID:

```http request
curl -X GET "http://localhost:8000/api/v1/recognition/faces/<image_id>/img"
-H "x-api-key: <service_api_key>"
```

| Element   | Description | Type   | Required | Notes                                                        |
| --------- | ----------- | ------ | -------- | ------------------------------------------------------------ |
| x-api-key | header      | string | required | api key of the Face recognition service, created by the user |
| image_id  | variable    | UUID   | required | UUID of the image to download                                |

Response body is binary image. Empty bytes if image not found.

### Verify Faces from a Given Image

To compare faces from the uploaded images with the face in saved image ID:

```http request
curl -X POST "http://localhost:8000/api/v1/recognition/faces/<image_id>/verify?
limit=<limit>&det_prob_threshold=<det_prob_threshold>&face_plugins=<face_plugins>&status=<status>" \
-H "Content-Type: multipart/form-data" \
-H "x-api-key: <service_api_key>" \
-F file=<local_file>
```


| Element            | Description | Type    | Required | Notes                                                        |
| ------------------ | ----------- | ------- | -------- | ------------------------------------------------------------ |
| Content-Type       | header      | string  | required | multipart/form-data                                          |
| x-api-key          | header      | string  | required | api key of the Face recognition service, created by the user |
| image_id           | variable    | UUID    | required | UUID of the verifying face                                   |
| file               | body        | image   | required | allowed image formats: jpeg, jpg, ico, png, bmp, gif, tif, tiff, webp. Max size is 5Mb |
| limit              | param       | integer | optional | maximum number of faces on the target image to be recognized. It recognizes the biggest faces first. Value of 0 represents no limit. Default value: 0 |
| det_prob_threshold | param       | string  | optional | minimum required confidence that a recognized face is actually a face. Value is between 0.0 and 1.0. |
| face_plugins       | param       | string  | optional | comma-separated slugs of face plugins. If empty, no additional information is returned. [Learn more](Face-services-and-plugins.md) |
| status             | param       | boolean | optional | if true includes system information like execution_time and plugin_version fields. Default value is false |

Response body on success:

```json
{
  "result": [
    {
      "age" : [ 25, 32 ],
      "gender" : "female",
      "embedding" : [ -0.049007344990968704, "...", -0.01753818802535534 ],
      "box" : {
        "probability" : 0.9997453093528748,
        "x_max" : 205,
        "y_max" : 167,
        "x_min" : 48,
        "y_min" : 0
      },
      "landmarks" : [ [ 260, 129 ], [ 273, 127 ], [ 258, 136 ], [ 257, 150 ], [ 269, 148 ] ],
      "similarity" : 0.97858,
      "execution_time" : {
        "age" : 59.0,
        "gender" : 30.0,
        "detector" : 177.0,
        "calculator" : 70.0
      }
    }
  ],
  "plugins_versions" : {
    "age" : "agegender.AgeDetector",
    "gender" : "agegender.GenderDetector",
    "detector" : "facenet.FaceDetector",
    "calculator" : "facenet.Calculator"
  }
}
```

| Element                    | Type    | Description                                                  |
| -------------------------- | ------- | ------------------------------------------------------------ |
| age                        | array   | detected age range. Return only if [age plugin](Face-services-and-plugins.md#face-plugins) is enabled |
| gender                     | string  | detected gender. Return only if [gender plugin](Face-services-and-plugins.md#face-plugins) is enabled |
| embedding                  | array   | face embeddings. Return only if [calculator plugin](Face-services-and-plugins.md#face-plugins) is enabled |
| box                        | object  | list of parameters of the bounding box for this face         |
| probability                | float   | probability that a found face is actually a face             |
| x_max, y_max, x_min, y_min | integer | coordinates of the frame containing the face                 |
| landmarks                  | array   | list of the coordinates of the frame containing the face-landmarks. Return only if [landmarks plugin](Face-services-and-plugins.md#face-plugins) is enabled |
| similarity                 | float   | similarity that on that image predicted person               |
| execution_time             | object  | execution time of all plugins                                |
| plugins_versions           | object  | contains information about plugin versions                   |

## Face Detection Service

To detect faces from the uploaded image:

```http request
curl  -X POST "http://localhost:8000/api/v1/detection/detect?limit=<limit>&det_prob_threshold=<det_prob_threshold>&face_plugins=<face_plugins>&status=<status>" \
-H "Content-Type: multipart/form-data" \
-H "x-api-key: <service_api_key>" \
-F file=<local_file>
```


| Element            | Description | Type    | Required | Notes                                                        |
| ------------------ | ----------- | ------- | -------- | ------------------------------------------------------------ |
| Content-Type       | header      | string  | required | multipart/form-data                                          |
| x-api-key          | header      | string  | required | api key of the Face Detection service, created by the user   |
| file               | body        | image   | required | image where to detect faces. Allowed image formats: jpeg, jpg, ico, png, bmp, gif, tif, tiff, webp. Max size is 5Mb |
| limit              | param       | integer | optional | maximum number of faces on the image to be recognized. It recognizes the biggest faces first. Value of 0 represents no limit. Default value: 0 |
| det_prob_threshold | param       | string  | optional | minimum required confidence that a recognized face is actually a face. Value is between 0.0 and 1.0 |
| face_plugins       | param       | string  | optional | comma-separated slugs of face plugins. If empty, no additional information is returned. [Learn more](Face-services-and-plugins.md) |
| status             | param       | boolean | optional | if true includes system information like execution_time and plugin_version fields. Default value is false |

Response body on success:

```json
{
  "result" : [ {
    "age" : [ 25, 32 ],
    "gender" : "female",
    "embedding" : [ -0.03027934394776821, "...", -0.05117142200469971 ],
    "box" : {
      "probability" : 0.9987509250640869,
      "x_max" : 376,
      "y_max" : 479,
      "x_min" : 68,
      "y_min" : 77
    },
    "landmarks" : [ [ 156, 245 ], [ 277, 253 ], [ 202, 311 ], [ 148, 358 ], [ 274, 365 ] ],
    "execution_time" : {
      "age" : 30.0,
      "gender" : 26.0,
      "detector" : 130.0,
      "calculator" : 49.0
    }
  } ],
  "plugins_versions" : {
    "age" : "agegender.AgeDetector",
    "gender" : "agegender.GenderDetector",
    "detector" : "facenet.FaceDetector",
    "calculator" : "facenet.Calculator"
  }
}
```

| Element                    | Type    | Description                                                  |
| -------------------------- | ------- | ------------------------------------------------------------ |
| age                        | array   | detected age range. Return only if [age plugin](Face-services-and-plugins.md#face-plugins) is enabled |
| gender                     | string  | detected gender. Return only if [gender plugin](Face-services-and-plugins.md#face-plugins) is enabled |
| embedding                  | array   | face embeddings. Return only if [calculator plugin](Face-services-and-plugins.md#face-plugins) is enabled |
| box                        | object  | list of parameters of the bounding box for this face (on processedImage) |
| probability                | float   | probability that a found face is actually a face (on processedImage) |
| x_max, y_max, x_min, y_min | integer | coordinates of the frame containing the face (on processedImage) |
| landmarks                  | array   | list of the coordinates of the frame containing the face-landmarks. Return only if [landmarks plugin](Face-services-and-plugins.md#face-plugins) is enabled |
| execution_time             | object  | execution time of all plugins                                |
| plugins_versions           | object  | contains information about plugin versions                   |


## Face Verification Service

To compare faces from given two images:

```http request
curl  -X POST "http://localhost:8000/api/v1/verification/verify?limit=<limit>&prediction_count=<prediction_count>&det_prob_threshold=<det_prob_threshold>&face_plugins=<face_plugins>&status=<status>" \
-H "Content-Type: multipart/form-data" \
-H "x-api-key: <service_api_key>" \
-F source_image=<local_check_file>
-F target_image=<local_process_file>
```


| Element            | Description | Type    | Required | Notes                                                        |
| ------------------ | ----------- | ------- | -------- | ------------------------------------------------------------ |
| Content-Type       | header      | string  | required | multipart/form-data                                          |
| x-api-key          | header      | string  | required | api key of the Face verification service, created by the user |
| source_image       | body        | image   | required | file to be verified. Allowed image formats: jpeg, jpg, ico, png, bmp, gif, tif, tiff, webp. Max size is 5Mb |
| target_image       | body        | image   | required | reference file to check the source file. Allowed image formats: jpeg, jpg, ico, png, bmp, gif, tif, tiff, webp. Max size is 5Mb |
| limit              | param       | integer | optional | maximum number of faces on the target image to be recognized. It recognizes the biggest faces first. Value of 0 represents no limit. Default value: 0 |
| det_prob_threshold | param       | string  | optional | minimum required confidence that a recognized face is actually a face. Value is between 0.0 and 1.0. |
| face_plugins       | param       | string  | optional | comma-separated slugs of face plugins. If empty, no additional information is returned. [Learn more](Face-services-and-plugins.md) |
| status             | param       | boolean | optional | if true includes system information like execution_time and plugin_version fields. Default value is false |

Response body on success:

```json
{
  "result" : [{
    "source_image_face" : {
      "age" : [ 25, 32 ],
      "gender" : "female",
      "embedding" : [ -0.0010271212086081505, "...", -0.008746841922402382 ],
      "box" : {
        "probability" : 0.9997453093528748,
        "x_max" : 205,
        "y_max" : 167,
        "x_min" : 48,
        "y_min" : 0
      },
      "landmarks" : [ [ 92, 44 ], [ 130, 68 ], [ 71, 76 ], [ 60, 104 ], [ 95, 125 ] ],
      "execution_time" : {
        "age" : 85.0,
        "gender" : 51.0,
        "detector" : 67.0,
        "calculator" : 116.0
      }
    },
    "face_matches": [
      {
        "age" : [ 25, 32 ],
        "gender" : "female",
        "embedding" : [ -0.049007344990968704, "...", -0.01753818802535534 ],
        "box" : {
          "probability" : 0.99975,
          "x_max" : 308,
          "y_max" : 180,
          "x_min" : 235,
          "y_min" : 98
        },
        "landmarks" : [ [ 260, 129 ], [ 273, 127 ], [ 258, 136 ], [ 257, 150 ], [ 269, 148 ] ],
        "similarity" : 0.97858,
        "execution_time" : {
          "age" : 59.0,
          "gender" : 30.0,
          "detector" : 177.0,
          "calculator" : 70.0
        }
      }],
    "plugins_versions" : {
      "age" : "agegender.AgeDetector",
      "gender" : "agegender.GenderDetector",
      "detector" : "facenet.FaceDetector",
      "calculator" : "facenet.Calculator"
    }
  }]
}
```

| Element                    | Type    | Description                                                  |
| -------------------------- | ------- | ------------------------------------------------------------ |
| source_image_face          | object  | additional info about source image face                      |
| face_matches               | array   | result of face verification                                  |
| age                        | array   | detected age range. Return only if [age plugin](Face-services-and-plugins.md#face-plugins) is enabled |
| gender                     | string  | detected gender. Return only if [gender plugin](Face-services-and-plugins.md#face-plugins) is enabled |
| embedding                  | array   | face embeddings. Return only if [calculator plugin](Face-services-and-plugins.md#face-plugins) is enabled |
| box                        | object  | list of parameters of the bounding box for this face         |
| probability                | float   | probability that a found face is actually a face             |
| x_max, y_max, x_min, y_min | integer | coordinates of the frame containing the face                 |
| landmarks                  | array   | list of the coordinates of the frame containing the face-landmarks. Return only if [landmarks plugin](Face-services-and-plugins.md#face-plugins) is enabled |
| similarity                 | float   | similarity between this face and the face on the source image |
| execution_time             | object  | execution time of all plugins                                |
| plugins_versions           | object  | contains information about plugin versions                   |



## Base64 Support

`since 0.5.1 version`

Except `multipart/form-data`, all CompreFace endpoints, that require images as input, accept images in `Base64` format. 
The base rule is to use `Content-Type: application/json` header and send JSON in the body. 
The name of the JSON parameter coincides with the name of the `multipart/form-data` parameter.

### Add an Example of a Subject, Base64

Full description [here](#add-an-example-of-a-subject).

```http request
curl -X POST "http://localhost:8000/api/v1/recognition/faces?subject=<subject>&det_prob_threshold=<det_prob_threshold>" \
-H "Content-Type: application/json" \
-H "x-api-key: <service_api_key>" \
-d {"file": "<base64_value>"}
```

### Recognize Faces from a Given Image, Base64

Full description [here](#recognize-faces-from-a-given-image).

```http request
curl  -X POST "http://localhost:8000/api/v1/recognition/recognize?limit=<limit>&prediction_count=<prediction_count>&det_prob_threshold=<det_prob_threshold>&face_plugins=<face_plugins>&status=<status>" \
-H "Content-Type: application/json" \
-H "x-api-key: <service_api_key>" \
-d {"file": "<base64_value>"}
```

### Verify Faces from a Given Image, Base64

Full description [here](#verify-faces-from-a-given-image).

```http request
curl -X POST "http://localhost:8000/api/v1/recognition/faces/<image_id>/verify?
limit=<limit>&det_prob_threshold=<det_prob_threshold>&face_plugins=<face_plugins>&status=<status>" \
-H "Content-Type: application/json" \
-H "x-api-key: <service_api_key>" \
-d {"file": "<base64_value>"}
```

### Face Detection Service, Base64

Full description [here](#face-detection-service).

```http request
curl  -X POST "http://localhost:8000/api/v1/detection/detect?limit=<limit>&det_prob_threshold=<det_prob_threshold>&face_plugins=<face_plugins>&status=<status>" \
-H "Content-Type: application/json" \
-H "x-api-key: <service_api_key>" \
-d {"file": "<base64_value>"}
```

### Face Verification Service, Base64

Full description [here](#face-verification-service).

```http request
curl -X POST "http://localhost:8000/api/v1/verification/verify?limit=<limit>&prediction_count=<prediction_count>&det_prob_threshold=<det_prob_threshold>&face_plugins=<face_plugins>&status=<status>" \
-H "Content-Type: application/json" \
-H "x-api-key: <service_api_key>" \
-d {"source_image": "<source_image_base64_value>", "target_image": "<target_image_base64_value>"}
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/comprefacelearning/  

