# STL关联式容器-RB-tree


> 左旋： 对节点 X 进行左旋，也就说让节点 X 成为左节点。
>
> 右旋： 对节点 X 进行右旋，也就说让节点 X 成为右节点。

### 1. 单旋转&双旋转

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220411221730369.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220411221617231.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220411221810843.png)

### 2. RB-Tree

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220411221916666.png)

### 3. 节点插入

- **S为黑色，且X为外侧插入节点，先对P，G做一次单旋转，在更改P，G颜色**

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220411222419521.png)

- **S为黑色，且X为内侧插入，先对P，X做一次单旋转，并更改G，X的颜色，在将结果对G做一次单旋转**

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220411222557816.png)

- **S为红色，X为外侧插入，先对 P和G做一次单旋转，并改变X的颜色，如果GG为黑色则成功，否则进入第四种情况**

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220411222840936.png)

- S为红色，且X为外侧插入，先对PG做一次单旋转，改变X红色，如果GG为红色

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220412111653784.png)

### 4. 数据结构

#### .1. RB-tree 节点设计

```c++
struct __rb_tree_node_base{
    typedef __rb_tree_color__type color_type;
    typedef __rb_tree_color_node_base* base_ptr;
    color_type color;
    base_ptr parent;
    base_ptr left;
    base_ptr right;
}
template <class Value>
struct __rb_tree_node: public __rb_tree_node_base{
    typedef __rb_tree_node<Value>* link_type;
    Value value_field;
} 	
```

#### .2. 迭代器设计

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220411231528032.png)

```c++
void increment(){
	if(node->right!=0){     //右节点不为空，右节点为空俩种情况
        node=node->right;
        while(node->left!=0){
            node=node->left;
        }
    }else{
        base_ptr y=node->parent;
        while(node==y->right){  //如果当前节点本身式右子节点
            node=y;
            y=y->parent;
        }
        if(node->right!=y){
            node=y;
        }
    }
};
void decreament(){
    //如果是红节点，且父节点的父节点等于自己；
    if(node->color==__rb_tree_red && node->parent->parent==node){
        node=node->right;
    }else if(node->left!=0){
        base_ptr y=node->left;
        while(y->right!=0){
            y=y->right;
        }
        node=y;
    }else{
        base_ptr y=node->parent;
        while(node==y->left){
            node=y;
            y=y->parent;
        }
        node=y;
    }
};
```

#### .3. RB-tree 的数据结构

```c++
template <class Key, class Value, class KeyOfValue, class Compare, class Alloc=alloc>
class rb_tree{
protected:
    typedef void* void_pointer;
    typedef __rb_tree_node_base* baseptr;
    typedef __rb_tree_node<Value> rb_tree_node;
    typedef simple_alloc<rb_tree_node,Alloc> tr_tree_node_allocator;
    typedef __rb_tree_color_type color_type;
public:
    typedef Key key_type;
    typedef Value value_type;
    typedef Value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef rb_tree_node* link_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
protected:
    link_type get_node(){return rb_tree_node_allocator::allocate();};
    void put_node(link_type p){rb_tree_node_allocator::deallocate(p)};
}
```

### 5. 内存构造(没理解header为什么这样设计)

```c++
void init(){
    header=get_node();
    color(head)=__rb_tree_red;
    root()=0;
    leftmost()=header;
    rightmost()=header;
}
```

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220412100924975.png)

### 6. 插入节点

#### .1. insert_equal()

```c++
template <class Key, class Value, class KeyOfValue, class Compare, class Allloc>
typename rb_tree<Key, Value, KeyOfValue,Compare,Alloc>::iterator rb_tree<Key, Value, KeyOfValue, Compare, Alloc>::insert_equal(const Value& v){
    link_type y=header;
    link_type x=root();
    while(x!=0){
        y=x;
        x=key_compare(KeyOfValue()(v),key(x))?left(x):right(x);
    }
    return __insert(x,y,v);
}
```

#### .2. insert_unique()

> 若待插入的 key 与某个结点相同 (设为 p)，则在 while 循环中，`某一次 x = p 后，大于等于向右走，则下一次 x = p.right，由于 v 的值一定小于 p 的右子树中任何一个值`，所以进入 p 的右子树后，x 一定是一直向左走直到节点 y（ｙ的左儿子为空）。`则 y 为 p 的右子树最小值，iterator (p) = iterator (y) - 1，p 即为代码中的 j`。若ｖ与ｊ的值不同，则可以执行插入操作，否则返回ｊ和 false。

```c++
template <class Key, class Value, class KeyOfValue, class Compare, class Allloc>
pair<typename rb_tree<Key, Value, KeyOfValue,Compare,Alloc>::iterator,bool> rb_tree<Key, Value, KeyOfValue, Compare, Alloc>::insert_unique(const Value& v){
	link_type y=header;
    link_type x=root();
    bool comp=true;
    while(x!=0){
        y=x;
        comp=key_compre(KeyOfValue()(v),key(x));
        x=com?left(x):right(x);
    }
    iterator j=iterator(y); //令迭代器j指向插入点的父节点y
    if(comp){    //如果离开while循环，comp为真，表示遇到大，将插入左侧
        if(j==begin()){   //如果插入点父节点为最左节点
            return pair<iterator,bool>(__insert(x,y,v),true);
        }else{
            --j;
        }
    }
    if(key_compare(key(j.node),KeyOfValue()(v))){
        //小于新值，表示遇到小，将插入右侧
        return pair<iterator,bool>(__insert(x,y,v),true);
    }
    return pair<iterator,bool>(j,false);
}
```

#### .3. __insert() 底层实现

```c++
template <class Key, class Value, class KeyOfValue, class Compare, class Alloc>
typename rb_tree<Key, Value, KeyOfValue, Compare, Alloc>::iterator
rb_tree<Key, Value, KeyOfValue, Compare, Alloc>::
__insert(base_ptr x_, base_ptr y_, const Value& v) {
// 参数x_ 为新插入节点，參數y_ 为插入点父节点，参数v为新值。
  link_type x = (link_type) x_;
  link_type y = (link_type) y_;
  link_type z;
    
  if (y == header || x != 0 || key_compare(KeyOfValue()(v), key(y))) {      
    z = create_node(v);  
    left(y) = z;          // 这使y为header时，leftmost()=z
    if (y == header) {          //
      root() = z;
      rightmost() = z;
    }
    else if (y == leftmost())   
      leftmost() = z;           
  }
  else {
    z = create_node(v);    
    right(y) = z;           
    if (y == rightmost())
      rightmost() = z;          //维护rightmost()，使它永远指向最右节点
  }
  parent(z) = y;        
  left(z) = 0;      
  right(z) = 0;         
  __rb_tree_rebalance(z, header->parent);   //进行树的平衡旋转
  ++node_count;          // 节点数加一
  return iterator(z);   //返回迭代器，指向新增的节点
}
```

#### .4. _Rb_tree_rebalance 

是一个全局函数，用来对 RB-tree 进行调整下面的函数就是上面所说的 “由上而下的程序”。从源代码可以看到，有些时候只需调整节点颜色，有些时候需要做单旋转或双旋转；有些时候要左旋，有些时候要右旋

```c++
//全局函数，重新令树平衡（改变颜色及旋转）
//参数1位新增节点，参数2位root
inline void 
_Rb_tree_rebalance(_Rb_tree_node_base* __x, _Rb_tree_node_base*& __root)
{
  __x->_M_color = _S_rb_tree_red; //新增节点必为红
  while (__x != __root && __x->_M_parent->_M_color == _S_rb_tree_red) {//父节点为红
    if (__x->_M_parent == __x->_M_parent->_M_parent->_M_left) { //父节点为祖父节点的左孩子
      _Rb_tree_node_base* __y = __x->_M_parent->_M_parent->_M_right;//令y为伯父节点
      if (__y && __y->_M_color == _S_rb_tree_red) { //伯父节点存在，且为红
        __x->_M_parent->_M_color = _S_rb_tree_black; //更改父节点为黑
        __y->_M_color = _S_rb_tree_black; //更高伯父节点为红
        __x->_M_parent->_M_parent->_M_color = _S_rb_tree_red; //更高祖父节点为红
        __x = __x->_M_parent->_M_parent;
      }
      else {//无伯父节点，或伯父节点为红
        if (__x == __x->_M_parent->_M_right) { //新节点为父节点的右孩子
          __x = __x->_M_parent;
          _Rb_tree_rotate_left(__x, __root); //第一参数为左旋点
        }
        __x->_M_parent->_M_color = _S_rb_tree_black; //改变颜色
        __x->_M_parent->_M_parent->_M_color = _S_rb_tree_red;
        _Rb_tree_rotate_right(__x->_M_parent->_M_parent, __root);//第一参数为右旋点
      }
    }
    else {//父节点为祖父节点的右孩子
      _Rb_tree_node_base* __y = __x->_M_parent->_M_parent->_M_left;//令y为伯父节点
      if (__y && __y->_M_color == _S_rb_tree_red) {//有伯父节点，且为红
        __x->_M_parent->_M_color = _S_rb_tree_black;//更改父节点为黑
        __y->_M_color = _S_rb_tree_black;//更改伯父节点为黑
        __x->_M_parent->_M_parent->_M_color = _S_rb_tree_red;//更改祖父节点为红
        __x = __x->_M_parent->_M_parent;//准备继续向上层检查
      }
      else { //无伯父节点，或伯父节点为黑
        if (__x == __x->_M_parent->_M_left) { //如果新节点为父节点的左孩子
          __x = __x->_M_parent;
          _Rb_tree_rotate_right(__x, __root);//第一参数为右旋点
        }
        __x->_M_parent->_M_color = _S_rb_tree_black;//更改颜色
        __x->_M_parent->_M_parent->_M_color = _S_rb_tree_red;
        _Rb_tree_rotate_left(__x->_M_parent->_M_parent, __root);//第一参数为左旋点
      }
    }
  }
  __root->_M_color = _S_rb_tree_black;//根节点永远为黑
}
```

#### .5. _Rb_tree_rotate_left 

```c++
//全局函数
//新节点必为红节点，如果插入之处父节点也为红色，就违反了规则，此时需要做树形旋转（及颜色改变,颜色改变不是在这里）
inline void 
_Rb_tree_rotate_left(_Rb_tree_node_base* __x, _Rb_tree_node_base*& __root)
{
  //x为旋转点
  _Rb_tree_node_base* __y = __x->_M_right; //令y为旋转点的右子节点
  __x->_M_right = __y->_M_left;
  if (__y->_M_left !=0)
    __y->_M_left->_M_parent = __x; //别忘了设定父节点
  __y->_M_parent = __x->_M_parent;
 
  //令y完全顶替x的地位（必须将x对齐父节点的关系完全接收过来）
  if (__x == __root) //x为根节点
    __root = __y;
  else if (__x == __x->_M_parent->_M_left) //x为其父节点的左子节点
    __x->_M_parent->_M_left = __y;
  else
    __x->_M_parent->_M_right = __y; //x为其父节点的右子节点
  __y->_M_left = __x;
  __x->_M_parent = __y;
}
```

#### .6. _Rb_tree_rotate_right

```c++
//全局函数
//新节点必为红节点，如果插入之处父节点也为红色，就违反了规则，此时需要做树形旋转（及颜色改变,颜色改变不是在这里）
inline void 
_Rb_tree_rotate_right(_Rb_tree_node_base* __x, _Rb_tree_node_base*& __root)
{
  //y为旋转点
  _Rb_tree_node_base* __y = __x->_M_left; //令y为旋转点的左子节点
  __x->_M_left = __y->_M_right;
  if (__y->_M_right != 0)
    __y->_M_right->_M_parent = __x; //别忘了设定父节点
  __y->_M_parent = __x->_M_parent;
 
  //令y完全顶替x的地位（必须将x对齐父节点的关系完全接收过来）
  if (__x == __root)//x为根节点
    __root = __y;
  else if (__x == __x->_M_parent->_M_right)//x为其父节点的右子节点
    __x->_M_parent->_M_right = __y;
  else//x为其父节点的左子节点
    __x->_M_parent->_M_left = __y;
  __y->_M_right = __x;
  __x->_M_parent = __y;
}
```
#### .7. demo

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220412110231119.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220412110454831.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220412110812090.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220412110857840.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220412111010428.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220412111050677.png)

### 7. 元素查找

```c++
template <class key, class Value,class KeyOfValue, class Compare, class Alloc>
typename rb_tree<Key, Value, keyOfValue, Compare, Alloc>::iterator rb_tree<Key, Value, KeyOfValue, Compare, Alloc>:: find(const key &k){
    link_type y=header;
    link_type x=root();
    while(x!=0){
        if(!key_compare(key(x),k)){
            y=x;
            x=left(x);
        }else{
            x=right(x);
        }
    }
    iterator j=iterator(y);
    return (j==end()||key_compare(k,key(j.node)))?end();j
}
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/stl%E5%85%B3%E8%81%94%E5%BC%8F%E5%AE%B9%E5%99%A8-rb-tree/  

