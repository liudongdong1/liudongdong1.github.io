# Wandb_intro


>  Use W&B's lightweight, interoperable tools to quickly `track experiments, version and iterate on datasets, evaluate model performance, reproduce models, visualize results and spot regressions, and share findings with colleagues. `

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210816095604114.png)

### 0. Command

- **setup wandb**

```shell
#pip install wandb
#wandb login
```

- **start a new run**

```python
import wandb
wandb.init(project="my-test-project")
```

- **Track metrics**

```python
wandb.log({'accuracy': train_acc, 'loss': train_loss})
```

- **track hypermeters**

```python
wandb.config.dropout = 0.2
```

### 1. experiment tracking

- [`wandb.init()`](): Initialize a new run at the top of your script. This returns a `Run` object and creates a local directory where all logs and files are saved, then streamed asynchronously to a W&B server. If you want to use a private server instead of our hosted cloud server, we offer [Self-Hosting]().

- [`wandb.config`](): Save a dictionary of hyperparameters such as learning rate or model type. The model settings you capture in config are useful later to organize and query your results.

- [`wandb.log()`](): Log metrics over time in a training loop, such as accuracy and loss. By default, when you call `wandb.log` it appends a new step to the `history` object and updates the `summary` object.
  - `history`: An array of dictionary-like objects that tracks metrics over time. These time series values are shown as default line plots in the UI.
  - `summary`: By default, the final value of a metric logged with wandb.log(). You can set the summary for a metric manually to capture the highest accuracy or lowest loss instead of the final value. These values are used in the table, and plots that compare runs — for example, you could visualize at the final accuracy for all runs in your project.

- [**`wandb.log_artifact`**](): Save outputs of a run, like the model weights or a table of predictions. This lets you track not just model training, but` all the pipeline steps that affect the final model`.

> **Config**: `Track hyperparameters, architecture, dataset, and anything else you'd like to use to reproduce your model`. These will show up in columns— use config columns to group, sort, and filter runs dynamically in the app.
>
> **Project**: A project is `a set of experiments you can compare together.` `Each project gets a dedicated dashboard page`, and you can easily `turn on and off different groups of runs to compare different model versions`.
>
> **Notes**: A quick commit message to yourself, the note can be set from your script and is editable in the table.
>
> **Tags**: Identify baseline runs and favorite runs. You can filter runs using tags, and they're editable in the table.

```python
import wandb
config = dict (
  learning_rate = 0.01,
  momentum = 0.2,
  architecture = "CNN",
  dataset_id = "peds-0192",
  infra = "AWS",
)
wandb.init(
  project="detect-pedestrians",
  notes="tweak baseline",
  tags=["baseline", "paper1"],
  config=config,
)
```

#### .1. log

```python
wandb.log({"loss": 0.314, "epoch": 5,
           "inputs": wandb.Image(inputs),
           "logits": wandb.Histogram(ouputs),
           "captions": wandb.HTML(captions)})
```

- **Compare the best accuracy**: To compare the best value of a metric across runs, set the summary value for that metric. By default, summary is set to the last value you logged for each key. This is useful in the table in the UI, where you can sort and filter runs based on their summary metrics — so you could compare runs in a table or bar chart based on their *best* accuracy, instead of final accuracy. For example, you could set summary like so: `wandb.run.summary["best_accuracy"] = best_accuracy`
- **Multiple metrics on one chart**: Log multiple metrics in the same call to `wandb.log`, like this: `wandb.log({"acc'": 0.9, "loss": 0.1})`  and they will both be available to plot against in the UI
- **Custom x-axis**: Add a custom x-axis to the same log call to visualize your metrics against a different axis in the W&B dashboard. For example: `wandb.log({'acc': 0.9, 'epoch': 3, 'batch': 117})`
- **Log rich media and charts**: `wandb.log` supports the logging of a wide variety of data types, from [media like images and videos]() to [tables]() and [charts]().

##### 1. images

```python
#logging arrays as images
images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")
wandb.log({"examples": images})
#logging PIL images
images = [PIL.Image.fromarray(image) for image in image_array]
wandb.log({"examples": [wandb.Image(image) for image in images]})
#logging images from files
im = PIL.fromarray(...)
rgb_im = im.convert('RGB')
rgb_im.save('myimage.jpg')
wandb.log({"example": wandb.Image("myimage.jpg")})
```

###### .1. [semantic task](https://colab.research.google.com/drive/1SOVl3EvW82Q4QKJXX6JtHye4wFix_P4J)

###### .2. [bounding boxes](https://app.wandb.ai/stacey/yolo-drive/reports/Bounding-Boxes-for-Object-Detection--Vmlldzo4Nzg4MQ
)

`box_data`: a list of dictionaries, one for each box. The box dictionary format is described below.

- `position`: a dictionary representing the position and size of the box in one of two formats, as described below. Boxes need not all use the same format.
  - *Option 1:* `{"minX", "maxX", "minY", "maxY"}`. Provide a set of coordinates defining the upper and lower bounds of each box dimension.
  - *Option 2:* `{"middle", "width", "height"}`.  Provide a set of coordinates specifying the `middle `coordinates as `[x,y]`, and `width` and `height` as scalars.
- `class_id`: an integer representing the class identity of the box. See `class_labels` key below.
- `scores`: a dictionary of string labels and numeric values for scores. Can be used for filtering boxes in the UI.
- `domain`: specify the units/format of the box coordinates. **Set this to "pixel"** if the box coordinates are expressed in pixel space (i.e. as integers within the bounds of the image dimensions). By default, the domain is assumed to be a fraction/percentage of the image (a floating point number between 0 and 1).
- `box_caption`: (optional) a string to be displayed as the label text on this box 

`class_labels`: (optional) A dictionary mapping `class_id`s to strings. By default we will generate class labels `class_0`, `class_1`, etc. 

- example: https://wandb.ai/stacey/yolo-drive/reports/Bounding-Boxes-for-Object-Detection--Vmlldzo4Nzg4MQ

```python
class_id_to_label = {
    1: "car",
    2: "road",
    3: "building",
    ....
}
img = wandb.Image(image, boxes={
    "predictions": {
        "box_data": [{
            # one box expressed in the default relative/fractional domain
            "position": {
                "minX": 0.1,
                "maxX": 0.2,
                "minY": 0.3,
                "maxY": 0.4
            },
            "class_id" : 2,
            "box_caption": class_id_to_label[2],
            "scores" : {
                "acc": 0.1,
                "loss": 1.2
            },
            # another box expressed in the pixel domain
            # (for illustration purposes only, all boxes are likely
            # to be in the same domain/format)
            "position": {
                "middle": [150, 20],
                "width": 68,
                "height": 112
            },
            "domain" : "pixel",
            "class_id" : 3,
            "box_caption": "a building",
            "scores" : {
                "acc": 0.5,
                "loss": 0.7
            },
            ...
            # Log as many boxes an as needed
        }
        ],
        "class_labels": class_id_to_label
    },
    # Log each meaningful group of boxes with a unique key name
    "ground_truth": {
    ...
    }
})
wandb.log({"driving_scene": img})
```

##### 2. histograms

```python
np_hist_grads = np.histogram(grads, density=True, range=(0., 1.))
wandb.log({"gradients": wandb.Histogram(np_hist_grads)})
```

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210816102708235.png)

##### 3. 3D visualization

- **3d object**

```python
#3d object
wandb.log({"generated_samples":
           [wandb.Object3D(open("sample.obj")),
            wandb.Object3D(open("sample.gltf")),
            wandb.Object3D(open("sample.glb"))]})
```

- **piont cloud**

```python
point_cloud = np.array([[0, 0, 0, COLOR...], ...])

wandb.log({"point_cloud": wandb.Object3D(point_cloud)})
```

> `points`is a numpy array with the same format as the simple point cloud renderer shown above.
>
> `boxes` is a numpy array of python dictionaries with three attributes:
>
> - `corners`- a list of eight corners
> - `label`- a string representing the label to be rendered on the box (Optional)
> - `color`- rgb values representing the color of the box 
>
> type` is a string representing the scene type to render. Currently the only supported value is `lidar/beta

```python
# Log points and boxes in W&B
point_scene = wandb.Object3D({
    "type": "lidar/beta",
    "points": np.array(  # add points, as in a point cloud
        [
            [0.4, 1, 1.3], 
            [1, 1, 1], 
            [1.2, 1, 1.2]
        ]
    ),
    "boxes": np.array(  # draw 3d boxes
        [
            {
                "corners": [
                    [0,0,0],
                    [0,1,0],
                    [0,0,1],
                    [1,0,0],
                    [1,1,0],
                    [0,1,1],
                    [1,0,1],
                    [1,1,1]
                ],
                "label": "Box",
                "color": [123, 321, 111],
            },
            {
                "corners": [
                    [0,0,0],
                    [0,2,0],
                    [0,0,2],
                    [2,0,0],
                    [2,2,0],
                    [0,2,2],
                    [2,0,2],
                    [2,2,2]
                ],
                "label": "Box-2",
                "color": [111, 321, 0],
            }
        ]
      ),
      "vectors": np.array(  # add 3d vectors
          [
              {"start": [0, 0, 0], "end": [0.1, 0.2, 0.5]}
          ]
      )
})
wandb.log({"point_scene": point_scene})
```

#### .2. [charts](https://docs.wandb.ai/guides/track/log/plots)

- Line

```python
data = [[x, y] for (x, y) in zip(x_values, y_values)]
table = wandb.Table(data=data, columns = ["x", "y"])
wandb.log({"my_custom_plot_id" : wandb.plot.line(table, "x", "y",
           title="Custom Y vs X Line Plot")})
```

- scatter

```python
data = [[x, y] for (x, y) in zip(class_x_scores, class_y_scores)]
table = wandb.Table(data=data, columns = ["class_x", "class_y"])
wandb.log({"my_custom_id" : wandb.plot.scatter(table,
                            "class_x", "class_y")})
```

- bar

```python
data = [[label, val] for (label, val) in zip(labels, values)]
table = wandb.Table(data=data, columns = ["label", "value"])
wandb.log({"my_bar_chart_id" : wandb.plot.bar(table, "label",
                               "value", title="Custom Bar Chart")
```

- histrogram

```python
data = [[s] for s in scores]
table = wandb.Table(data=data, columns=["scores"])
wandb.log({'my_histogram': wandb.plot.histogram(table, "scores",
                           title="Histogram")})
```

- multi-line

```python
wandb.log({"my_custom_id" : wandb.plot.line_series(
          xs=[0, 1, 2, 3, 4],
          ys=[[10, 20, 30, 40, 50], [0.5, 11, 72, 3, 41]],
          keys=["metric Y", "metric Z"],
          title="Two Random Metrics",
          xname="x units")})
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/wandb_intro/  

