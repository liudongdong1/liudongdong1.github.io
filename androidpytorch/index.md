# AndroidPytorch


### 1. 模型转化

```python
import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

model = torchvision.models.mobilenet_v3_small(pretrained=True)
#model.load_state_dict(torch.load(model_pth)) # 加载参数
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)  # 模型转化
optimized_traced_model = optimize_for_mobile(traced_script_module)
optimized_traced_model.save("model.pt") # 保存文件
optimized_traced_model._save_for_lite_interpreter("app/src/main/assets/model.pt")
```

### 2. Gradle 依赖

```gradle
implementation 'org.pytorch:pytorch_android_lite:1.9.0'
implementation 'org.pytorch:pytorch_android_torchvision:1.9.0'
```

### 3. 封装函数

```java
public class MainActivity extends AppCompatActivity {

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    Bitmap bitmap = null;
    Module module = null;
    try {
      // creating bitmap from packaged into app android asset 'image.jpg',
      // app/src/main/assets/image.jpg
      bitmap = BitmapFactory.decodeStream(getAssets().open("image.jpg"));
      // loading serialized torchscript module from packaged into app android asset model.pt,
      // app/src/model/assets/model.pt
      module = LiteModuleLoader.load(assetFilePath(this, "model.pt"));
    } catch (IOException e) {
      Log.e("PytorchHelloWorld", "Error reading assets", e);
      finish();
    }

    // showing image on UI
    ImageView imageView = findViewById(R.id.image);
    imageView.setImageBitmap(bitmap);

    // preparing input tensor
    final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);

    // running the model
    final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

    // getting tensor content as java array of floats
    final float[] scores = outputTensor.getDataAsFloatArray();

    // searching for the index with maximum score
    float maxScore = -Float.MAX_VALUE;
    int maxScoreIdx = -1;
    for (int i = 0; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxScoreIdx = i;
      }
    }

    String className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];

    // showing className on UI
    TextView textView = findViewById(R.id.text);
    textView.setText(className);
  }

  /**
   * Copies specified asset to the file in /files app directory and returns this file absolute path.
   *
   * @return absolute file path
   */
  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }
}
```
- [测试获取top5 的分类结果](https://github.com/liudongdong1/HC05)
```java
public class RecognizeTorch {
    private Module module=null;

    private RecognizeTorch(){
    }
    private static class Inner {
        private static final RecognizeTorch instance = new RecognizeTorch();
    }
    public static RecognizeTorch getSingleton(){
        return RecognizeTorch.Inner.instance;
    }
    public Boolean initializeModel(Context context) throws IOException {
        module = LiteModuleLoader.load(assetFilePath(context, Constant.MODEL_PATH));
        if(module!=null){
            return true;
        }
        return false;
    }
    public String getRecognizeResult(FlexWindow flexWindow){
        float[] data=new float[5*5];
        ArrayList<Float> inputList=new ArrayList<>();
        ArrayList<Double> arrayList=new ArrayList<Double>();
        for(Double value :flexWindow.getSingleFlexData((int)(flexWindow.getSize()/6*4)).getFlexData()){
            arrayList.add(value);
        }
        for(int i=0;i<arrayList.size();i++){
            inputList.add(arrayList.get(i).floatValue());
            for(int j=0;j<arrayList.size();j++){
                if(i!=j){
                    inputList.add(arrayList.get(i).floatValue()-arrayList.get(j).floatValue());
                }
            }
        }
        long[] shape={1,5,5};
        for(int i=0;i<25;i++){
            data[i]=inputList.get(i).floatValue();
        }
        Tensor input_tensor= Tensor.fromBlob(data,shape);
        System.out.println(input_tensor.toString());
        return getRecognizeReuslt(input_tensor);
    }
    public String getRecognizeReuslt(Tensor inputTensor){
        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        // getting tensor content as java array of floats
        final float[] scores = outputTensor.getDataAsFloatArray();

        // searching for the index with maximum score
       /* float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxScoreIdx = i;
            }
        }*/
        int[] Index = new int[scores.length];
        Index = ArrayHelper.Arraysort(scores);

        for (int i = 0; i < 10; i++) {
            System.out.println(Index[i] + "：" + scores[i]);
        }
        String classname = Constant.Gesture_CHAR_CLASSES[Index[0]];
        return classname;
    }


    /**
     * Copies specified asset to the file in /files app directory and returns this file absolute path.
     * @return absolute file path
     * */
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}
```

### Resource

- https://blog.csdn.net/qq_40507857/article/details/118755061
- 实例代码pytorch 官方：https://github.com/liudongdong1/android-demo-app



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/androidpytorch/  

