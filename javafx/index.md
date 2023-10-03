# javaFx


> JavaFX是用于构建富Internet应用程序的Java库。 使用此库编写的应用程序可以跨多个平台一致地运行。 使用JavaFX开发的应用程序可以在各种设备上运行，例如台式计算机，移动电话，电视，平板电脑等。
>
> 为了开发具有丰富功能的**Client Side Applications** ，程序员过去依赖于各种库来添加诸如媒体，UI控件，Web，2D和3D等功能.JavaFX在单个库中包含所有这些功能。 
>
> JavaFX提供了丰富的图形和媒体API，并通过硬件加速图形利用现代**Graphical Processing Unit** 。 JavaFX还提供了接口，开发人员可以使用这些接口组合图形动画和UI控件。
>
> **Canvas and Printing API** - JavaFX提供了Canvas，一种即时模式的渲染API。 在包**javafx.scene.canvas**它包含一组canvas类，我们可以使用它直接在JavaFX场景的区域内绘制。 JavaFX还在包**javafx.print**提供用于打印目的的类。

### 0. Architecture

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210324204040.png)

- **javafx.animation** - 包含用于向JavaFX节点添加基于`过渡的动画（如填充，淡入淡出，旋转，缩放和平移）`的类。
- **javafx.application** - 包含一组负责``JavaFX应用程序生命周期的类``。
- **javafx.css** - 包含用于向JavaFX GUI应用程序``添加类似CSS样式的类`。
- **javafx.event** - 包含用于`传递和处理JavaFX事件的类和接口`。
- **javafx.geometry** - 包含用于`定义2D对象并对其执行操作的类`。
- **javafx.stage** - 此包包含`JavaFX应用程序的顶级容器类`。
- **javafx.scene** - 此包提供了`支持场景图的类和接口`。 此外，它还提供了`子包，如画布，图表，控件，效果，图像，输入，布局，媒体，绘画，形状，文本，转换，Web等`。有几个组件支持JavaFX丰富的API 。
- Prism是一种**high performance hardware–accelerated graphical pipeline** ，用于在JavaFX中渲染图形。 它可以渲染2-D和3-D图形。
- GWT提供管理Windows，定时器，曲面和事件队列的服务。 GWT将JavaFX平台连接到本机操作系统。
-  WebView是JavaFX的组件，用于处理此内容。 它使用一种名为**Web Kit**的技术，这是一种内部开源Web浏览器引擎。 该组件支持不同的Web技术，如HTML5，CSS，JavaScript，DOM和SVG。
- **JavaFX media engine**基于称为**Streamer**的开源引擎。 该媒体引擎支持视频和音频内容的回放。

### 1. Scene Builder

> **Scene Builder** - JavaFX提供名为Scene Builder的应用程序。 在将此应用程序集成到IDE（例如Eclipse和NetBeans）中时，用户可以访问拖放设计界面，该界面用于开发FXML应用程序（就像Swing Drag＆Drop和DreamWeaver应用程序一样）。`可以通过该工具进行GUI可视化拖拽编辑`
>
> - IDEA集成教程：https://blog.csdn.net/u011781521/article/details/86632482

- 页面编辑，并将Fxml文件生成对应的代码；

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210326195015.png)

### 2. TestFx

> TestFX allows developers to write simple assertions to simulate user interactions and verify expected states of JavaFX scene-graph nodes.

### 3. Tutorial

#### 3.1. StartPipeline

> 在代码中***`javafx.application.Application 类是所有Fx程序的入口，`\***每一个javaFx程序都可以理解为是一个应用，Fx程序中的自定义类继承并重写其start()类后，便具备了作为当前Fx程序的启动入口。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210324204704.png)

```java
public class Main extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception {
        FXMLLoader loader = new FXMLLoader(getClass().getResource("/com/ita/fxml/mediaplayer.fxml"));
        BorderPane pane = loader.load();
        Scene scene = new Scene(pane, 650, 400);  //在实例化场景时，必须将根对象传递给场景类的构造函数。
        primaryStage.setScene(scene);   //将场景对象附加到舞台。
        
        MediaPlayerController controller = ((MediaPlayerController) loader.getController());
        // Load Playlist FXML and inject controller/root
        FXMLLoader playListLoader = new FXMLLoader(getClass().getResource("/com/ita/fxml/playlist.fxml"));
        playListLoader.load();
        controller.injectPlayListController((PlaylistController) playListLoader.getController());
        controller.injectPlayListRoot(playListLoader.getRoot());
        bindSize(controller, scene);
        controller.setStage(primaryStage);
        primaryStage.show();   // 必须通过show 才会显示场景
        controller.applyDragAndDropFeatures(scene);
    }

    private void bindSize(MediaPlayerController controller, Scene scene){
        controller.timerSliderWidthProperty().bind(scene.widthProperty().subtract(500));
        controller.mediaViewWidthProperty().bind(scene.widthProperty());
        controller.mediaViewHeightProperty().bind(scene.heightProperty().subtract(70));
    }
   
    //start() - 要写入JavaFX图形代码的入口点方法。
	//stop() - 一个可以被覆盖的空方法，在这里你可以编写停止应用程序的逻辑。
	//init() - 一个可以被覆盖的空方法，但是你不能在这个方法中创建阶段或场景。

    public static void main(String[] args) {
        launch(args);
    }
}
```

- **Control** - 它是用户界面控件的基类，如**Accordion, ButtonBar, ChoiceBox, ComboBoxBase, HTMLEditor, etc. This class belongs to the package javafx.scene.control** 。
- **Pane** - 窗格是所有布局窗格的基类，例如**AnchorPane, BorderPane, DialogPane**等。此类属于一个名为**AnchorPane, BorderPane, DialogPane**的包。
- chart: 有两个子类，分别是**PieChart**和**XYChart** 。 这两个又具有子类，如**AreaChart, BarChart, BubbleChart**等，用于在JavaFX中绘制不同类型的XY平面图。

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210324204939.png)

#### 3.2. Event

> - **Foreground Events** - 需要用户直接交互的事件。 它们是作为人与图形用户界面中的图形组件交互的结果而生成的。 例如，单击按钮，移动鼠标，通过键盘输入字符，从列表中选择项目，滚动页面等。
> - **Background Events** - 需要最终用户交互的事件称为后台事件。 操作系统中断，硬件或软件故障，计时器到期，操作完成是后台事件的示例。

- **Mouse Event** - 这是单击鼠标时发生的输入事件。 它由名为**MouseEvent**的类表示。 它包括鼠标单击，鼠标按下，鼠标释放，鼠标移动，鼠标输入目标，鼠标退出目标等操作。
- **Key Event** - 这是一个输入事件，指示节点上发生的键击。 它由名为**KeyEvent**的类表示。 此事件包括按下键，释放键和键入键等操作。
- **Drag Event** - 这是拖动鼠标时发生的输入事件。 它由名为**DragEvent**的类表示。 它包括拖动输入，拖放，拖动输入目标，拖动退出目标，拖动等操作。
- **Window Event** - 这是与窗口显示/隐藏操作相关的事件。 它由名为**WindowEvent**的类表示。 它包括窗口隐藏，显示窗口，隐藏窗口，窗口显示等操作。

- 事件传递链
  - 捕获阶段：将传递到调度链中的所有节点（从上到下）。 如果这些节点中的任何节点具有为生成的事件注册的**filter** ，则将执行该**filter** 。 如果调度链中没有节点具有生成事件的过滤器，则将其传递到目标节点，最后目标节点处理该事件。
  - 冒泡阶段：事件从目标节点传播到阶段节点（从下到上）。 如果事件调度链中的任何节点具有为生成的事件注册的**handler** ，则将执行该**handler** 。 如果这些节点都没有处理事件的处理程序，则事件到达根节点，最后完成该过程。
  - 事件处理程序和过滤器：事件过滤器和处理程序包含处理事件的应用程序逻辑。 节点可以注册到多个处理程序/过滤器。 在父子节点的情况下，您可以为父节点提供公共过滤器/处理程序，它将作为所有子节点的默认处理。
  - 添加删除事件： 使用**Node**类的**addEventFilter()**方法注册此过滤器。

```java
//Creating the mouse event handler 
EventHandler<MouseEvent> eventHandler = new EventHandler<MouseEvent>() { 
   @Override 
   public void handle(MouseEvent e) { 
      System.out.println("Hello World"); 
      circle.setFill(Color.DARKSLATEBLUE);  
   } 
};   
//Adding event Filter 
Circle.addEventFilter(MouseEvent.MOUSE_CLICKED, eventHandler);

//---------注册方法二：
playButton.setOnMouseClicked((new EventHandler<MouseEvent>() { 
   public void handle(MouseEvent event) { 
      System.out.println("Hello World"); 
      pathTransition.play(); 
   } 
}));
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210324210040.png)

#### 3.3. UI控件

> JavaFX在包**javafx.scene.control**提供了几个类。 为了创建各种GUI组件（控件），JavaFX支持多种控件，如日期选择器，按钮文本字段等。

- **UI elements** - 这些是用户最终看到并与之交互的核心可视元素。 JavaFX提供了大量广泛使用的常用元素列表，从基本到复杂，我们将在本教程中介绍。
- **Layouts** - 它们定义了如何在屏幕上组织UI元素，并为GUI（图形用户界面）提供最终外观。 这部分将在布局章节中介绍。
- **Behavior** - 这些是用户与UI元素交互时发生的事件。 

| S.No |                          控制和描述                          |
| :--: | :----------------------------------------------------------: |
|  1   |           **Label**Label对象是用于放置文本的组件。           |
|  2   |             **Button**该类创建一个带标签的按钮。             |
|  3   | **ColorPicker**ColorPicker提供了一个控件窗格，旨在允许用户操作和选择颜色。 |
|  4   | **CheckBox**CheckBox是一个图形组件，可以处于on（true）或off（false）状态。 |
|  5   | **RadioButton**RadioButton类是一个图形组件，可以在组中处于ON（true）或OFF（false）状态。 |
|  6   |     **ListView**ListView组件向用户显示文本项的滚动列表。     |
|  7   | **TextField**TextField对象是一个文本组件，允许编辑单行文本。 |
|  8   | **PasswordField**PasswordField对象是专门用于输入密码的文本组件。 |
|  9   | **Scrollbar**Scrollbar控件表示滚动条组件，以便用户可以从值范围中进行选择。 |
|  10  | **FileChooser**FileChooser控件表示用户可以从中选择文件的对话窗口。 |
|  11  | **ProgressBar**随着任务进展完成，进度条显示任务的完成百分比。 |
|  12  | **Slider**滑块允许用户通过在有界区间内滑动旋钮以图形方式选择值。 |

#### 3.4. charts

> 通常，图表是数据的图形表示。 有各种各样的图表来表示数据，如**Bar Chart, Pie Chart, Line Chart, Scatter Chart,**等。JavaFX支持各种**Pie Charts**和**XY Charts** 。 在XY平面上表示的图表包括**AreaChart, BarChart, BubbleChart, LineChart, ScatterChart, StackedAreaChart, StackedBarChart,**等。每个图表由一个类表示，所有这些图表都属于包**javafx.scene.chart** 。 名为**Chart**的类是JavaFX中所有图表的基类， **XYChart**是在XY平面上绘制的所有图表的基类。

- 定义图表的轴： 轴是表示X或Y轴的抽象类。 它有两个子类来定义每种类型的轴，即**CategoryAxis**和**NumberAxis** 
- 实例化相应的类
- 准备并将数据传递到图表

#### 3.5.  Layout Panes

- 创建节点。
- 实例化所需布局的相应类。
- 设置布局的属性。
- 将所有创建的节点添加到布局中。
- JavaFx chart 库： https://github.com/HanSolo/charts  提供各种使用图表

| S.No |                          形状和描述                          |
| :--: | :----------------------------------------------------------: |
|  1   | [HBox](https://www.iowiki.com/javafx/layout_panes_hbox.html)HBox布局将应用程序中的所有节点排列在一个水平行中。包**javafx.scene.layout**名为**HBox**的类表示文本水平框布局。 |
|  2   | [VBox](https://www.iowiki.com/javafx/layout_panes_vbox.html)VBox布局将我们应用程序中的所有节点排列在一个垂直列中。包**javafx.scene.layout**名为**VBox**的类表示文本垂直框布局。 |
|  3   | [BorderPane](https://www.iowiki.com/javafx/layout_borderpane.html)边框窗格布局将应用程序中的节点排列在顶部，左侧，右侧，底部和中心位置。包**javafx.scene.layout**名为**BorderPane**的类表示边框窗格布局。 |
|  4   | [StackPane](https://www.iowiki.com/javafx/layout_stackpane.html)堆栈窗格布局将应用程序中的节点排列在另一个上面，就像在堆栈中一样。 首先添加的节点位于堆栈的底部，下一个节点位于堆栈的顶部。包**javafx.scene.layout**名为**StackPane**的类表示堆栈窗格布局。 |
|  5   | [TextFlow](https://www.iowiki.com/javafx/layout_panes_textflow.html)文本流布局在单个流中排列多个文本节点。包**javafx.scene.layout**名为**TextFlow**的类表示文本流布局。 |
|  6   | [AnchorPane](https://www.iowiki.com/javafx/layout_anchorpane.html)“锚点”窗格布局将应用程序中的节点锚定在距窗格特定距离处。包**javafx.scene.layout**名为**AnchorPane**的类表示Anchor窗格布局。 |
|  7   | [TilePane](https://www.iowiki.com/javafx/layout_tilepane.html)Tile窗格布局以均匀大小的tile的形式添加应用程序的所有节点。包**javafx.scene.layout**名为**TilePane**的类表示TilePane布局。 |
|  8   | [GridPane](https://www.iowiki.com/javafx/layout_gridpane.html)网格窗格布局将应用程序中的节点排列为行和列的网格。 使用JavaFX创建表单时，此布局非常方便。包**javafx.scene.layout**名为**GridPane**的类表示GridPane布局。 |
|  9   | [FlowPane](https://www.iowiki.com/javafx/layout_flowpane.html)流窗格布局包装流中的所有节点。 水平流动窗格将窗格的元素包裹在其高度，而垂直流动窗格将元素包裹在其宽度处。名为**FlowPane**的类**javafx.scene.layout**表示Flow Pane布局。 |

#### 3.6. CSS样式

- **Selector** - 选择器是将应用样式的HTML标记。 这可以是任何标签，如**《h1》**或**《table》**等。
- **Property** - 属性是HTML标记的一种属性。 简单来说，所有HTML属性都转换为CSS属性。 它们可以是颜色， **border**等。
- **Value** - 将值分配给属性。 例如，颜色属性可以具有**red**或**#F1F1F1** 。

```java
//添加自己的样式表
Scene scene = new Scene(new Group(), 500, 400); 
scene.getStylesheets().add("path/stylesheet.css");
//添加内联样式表
.button { 
   -fx-background-color: red; 
   -fx-text-fill: white; 
}
```

- Music Relative Dashboar: [XR3Player](https://github.com/goxr3plus/XR3Player)   如果后期学习语言这部分处理，学习该项目代码。

> - Done ✔️
>   - Support almost all audio formats through smart converting to .mp3
>   - Amazing Audio Spectrum Visualizers
>   - Audio Amplitudes Waveform
>   - Chromium Web Browser
>   - Full Dropbox access
>   - Multiple User Accounts
>   - Configurable via multiple settings
>   - Advanced Tag Editor
>   - File Organizer and Explorer
>   - Multiple Libraries/Playlists support
>   - System monitor ( CPU , RAM )
>   - Audio Effects and Filters
> - TODO 🚧
>   - *XR3Player is actively developed. More features will come!*
>   - Support all audio file formats by default
>   - Support all video file formats by default
>   - Speech Recongition
>   - Smart AI Assistant
>   - Online Subscription website
>   - Android and IOS applications

![](https://user-images.githubusercontent.com/20374208/48313813-34fdc180-e5ca-11e8-9da7-c6148dc0cbe5.png)

- Dashboard sumary: https://github.com/HanSolo/tilesfx

![Overview](https://raw.githubusercontent.com/HanSolo/tilesfx/master/TilesFX.png)

- [JavaFx Material Design Library](https://github.com/sshahine/JFoenix)

![Demo demonstration](https://camo.githubusercontent.com/cb983c05cd402e70b436e27d6e8a3d4850e2fc14f531dafd77027252bc9e8203/68747470733a2f2f692e696d6775722e636f6d2f686459466a34642e676966)

#### 3.7. 项目目录

```xml
src/main
  ├──java
     ├── controllers
        ├──Screen1controller.java
        ├──Screen2controller.java
     ├── service
        ├──Service1.java
     ├── dao(persist)
        ├── SaveProducts.java
  ├──resources
     ├──view
        ├──screen1.fxml
        ├──screen2.fxml
     ├──css
        ├──style.css
     ├──images
        ├──img1.jpg
        ├──img2.jpg
```

### 4. Project

#### 4.1. **[ maps](https://github.com/gluonhq/maps)**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210324215744.png)

#### 4.2. [CalendarFX](https://github.com/dlsc-software-consulting-gmbh/CalendarFX)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210324215823.png)

#### 4.3. [octillect](https://github.com/zero-based/octillect)

![demo](https://github.com/MonicaTanios/octillect/raw/gh-pages/assets/drag-and-filter.gif?raw=true)

#### 4.4. [MusicPlayer](https://github.com/Mpmart08/MusicPlayer)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210326195314.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/Package model.png)

### 5. Resource

- 快速教程： https://www.iowiki.com/javafx/javafx_application.html
- javaFx 一些比较好的repository： https://github.com/HanSolo?after=Y3Vyc29yOnYyOpK5MjAyMC0xMC0yM1QxNTo1MjoxNSswODowMM4SRZB1&tab=repositories

### 6. 问题记录

> Intellij IDEA 'Error:java: 无效的源发行版:13'

- 查看本机的jdk版本：命令提示符输入：java -version

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210326093757.png)

- 修改IDEA当中的Project项目中的jdk版本

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210326093819.png)

- 修改modules中的sdk

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210326093845.png)

- File-->Setting -->Build,Execution,Deployment-->Complier-->Java Complier

![](https://gitee.com/github-25970295/blogImage/raw/master/img/20210326093916.png)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/javafx/  

