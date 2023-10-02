# WindowStructure


#### 1.  CircularFifoQueue

> - 通过改变start 和end的索引位置，实现先进先出的操作，同时获取值的时候按照进入的顺序。
> - [依赖包](https://mvnrepository.com/)：implementation group: 'org.apache.commons', name: 'commons-collections4', version: '4.4'

```java
public class FlexWindow {
    private String gestureName;   //手势名称
    private static volatile CircularFifoQueue<FlexData> winFlexData; //滑动窗口用于保存传感器数据
    //todo  添加视图显示效果
    private String gestureImageURL;  //手势对应图片地址
    private String voiceURL;   //手势对应声音地址

    /**
     * @function： 静态内部单例构造模式
     * */
    private FlexWindow(){
        gestureName="None";
        winFlexData=new CircularFifoQueue<FlexData>(Constant.Flex_WINDOW_SIZE);
    }
    private static class Inner {
        private static final FlexWindow instance = new FlexWindow();
    }
    public static FlexWindow getSingleton(){
        return Inner.instance;
    }

    /**
     * @function： 添加flexdata数据到显示操作窗口
     * */
    public void addFlexData(FlexData flexData){
        winFlexData.offer(flexData);
    }
    //todo 这里需要测试一下这个输出是否正确，顺序是否有问题
    /**
     * @function: 获取 窗口Flexdata数值
     * */
    @RequiresApi(api = Build.VERSION_CODES.N)
    public List<FlexData> getFlexData(){
        List<FlexData> result = winFlexData.stream().collect(Collectors.toList());   //查看源代码是按照插入的顺序进行访问的  测试通过，还是保持之前的结构
        //Collections.reverse(result);
        return result;
    }

    public String getGestureName(){
        return gestureName;
    }

    public void setGestureName(String gestureName) {
        this.gestureName = gestureName;
    }

    public void clearData(){
        gestureName="None";
        winFlexData.clear();
    }
    public int getSize(){
        return winFlexData.size();
    }
    public FlexData getSingleFlexData(int i){
        return winFlexData.get(i);
    }
}
```

#### 2. Deque 实现

```java
public class MovingAverages {
    private Deque<Double> queue;
    private int size;
    private double sum;
    public MovingAverages() {
        this.queue = new ArrayDeque<>();
        this.size = Constant.AVERAGE_WINDOW_SIZE;
        this.sum = 0;
    }
    public double next(Double val) {
        sum+=val;
        if (queue.size()==size){
            sum-=queue.pollFirst();
        }
        queue.offerLast(val);
        return sum/queue.size();
    }
}
```

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/windowstructure/  

