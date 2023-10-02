# CompositeMode


> **组合模式(Composite Pattern)**：组合多个对象形成树形结构以表示具有 "整体—部分" 关系的层次结构。组合模式对单个对象（即叶子对象）和组合对象（即容器对象）的使用具有一致性，组合模式又可以称为 "整体—部分"(Part-Whole) 模式，它是一种对象结构型模式。

> - **Component（抽象构件）**：它可以是接口或抽象类，为叶子构件和容器构件对象声明接口，在该角色中可以包含所有子类共有行为的声明和实现。在抽象构件中定义了访问及管理它的子构件的方法，如增加子构件、删除子构件、获取子构件等。
> - **Leaf（叶子构件）**：它在组合结构中表示叶子节点对象，叶子节点没有子节点，它实现了在抽象构件中定义的行为。对于那些访问及管理子构件的方法，可以通过异常等方式进行处理。
> - **Composite（容器构件）**：它在组合结构中表示容器节点对象，容器节点包含子节点，其子节点可以是叶子节点，也可以是容器节点，它提供一个集合用于存储子节点，实现了在抽象构件中定义的行为，包括那些访问及管理子构件的方法，在其业务方法中可以递归调用其子节点的业务方法。
>
> 组合模式的**关键是定义了一个抽象构件类，它既可以代表叶子，又可以代表容器**，而客户端针对该抽象构件类进行编程，无须知道它到底表示的是叶子还是容器，可以对其进行统一处理。**同时容器对象与抽象构件类之间还建立一个聚合关联关系**，在容器对象中既可以包含叶子，也可以包含容器，以此实现递归组合，形成一个树形结构。

### 1. 文件

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210705151004227.png)

```java
public abstract class Component {

    public String getName() {
        throw new UnsupportedOperationException("不支持获取名称操作");
    }

    public void add(Component component) {
        throw new UnsupportedOperationException("不支持添加操作");
    }

    public void remove(Component component) {
        throw new UnsupportedOperationException("不支持删除操作");
    }

    public void print() {
        throw new UnsupportedOperationException("不支持打印操作");
    }

    public String getContent() {
        throw new UnsupportedOperationException("不支持获取内容操作");
    }
}
```

```java
public class File extends Component {
    private String name;
    private String content;

    public File(String name, String content) {
        this.name = name;
        this.content = content;
    }

    @Override
    public String getName() {
        return this.name;
    }

    @Override
    public void print() {
        System.out.println(this.getName());
    }

    @Override
    public String getContent() {
        return this.content;
    }
}
```

```java
public class Folder extends Component {
    private String name;
    private List<Component> componentList = new ArrayList<Component>();
    public Integer level;

    public Folder(String name) {
        this.name = name;
    }

    @Override
    public String getName() {
        return this.name;
    }

    @Override
    public void add(Component component) {
        this.componentList.add(component);
    }

    @Override
    public void remove(Component component) {
        this.componentList.remove(component);
    }

    @Override
    public void print() {
        System.out.println(this.getName());
        if (this.level == null) {
            this.level = 1;
        }
        String prefix = "";
        for (int i = 0; i < this.level; i++) {
            prefix += "\t- ";
        }
        for (Component component : this.componentList) {
            if (component instanceof Folder){
                ((Folder)component).level = this.level + 1;
            }
            System.out.print(prefix);
            component.print();
        }
        this.level = null;
    }
}
```

### 2. 透明&安全

- `透明组合模式`中，抽象构件角色中声明了所有用于管理成员对象的方法，譬如在示例中 `Component` 声明了 `add`、`remove` 方法，这样做的好处是确保所有的构件类都有相同的接口。透明组合模式也是组合模式的标准形式。透明组合模式的缺点是不够安全，因为叶子对象和容器对象在本质上是有区别的，叶子对象不可能有下一个层次的对象，即不可能包含成员对象，因此为其提供 `add()`、`remove()` 等方法是没有意义的，这在编译阶段不会出错，但在运行阶段如果调用这些方法可能会出错（如果没有提供相应的错误处理代码）
- `安全组合模式`中，在抽象构件角色中没有声明任何用于管理成员对象的方法，而是在容器构件 `Composite` 类中声明并实现这些方法。 `java.awt` 和 `swing` 中的组合模式即为安全组合模式.

### 3. Java.awt

- AWT(Abstract Window Toolkit)：抽象窗口工具集，是第一代的Java GUI组件。绘制依赖于底层的操作系统。基本的AWT库处理用户界面元素的方法是把这些元素的创建和行为委托给每个目标平台上（Windows、 Unix、 Macintosh等）的本地GUI工具进行处理。

- Swing，不依赖于底层细节，是轻量级的组件。现在多是基于Swing来开发。

> - 基本组件又称构件，诸如按钮、文本框之类的图形界面元素。
> - 容器是一种比较特殊的组件，可以容纳其他组件，容器如窗口、对话框等。所有的容器类都是 `java.awt.Container` 的直接或间接子类

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210705151544734.png)

### Resource

- https://juejin.cn/post/6844903687228407821

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/compositemode/  

