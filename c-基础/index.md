# C++基础


> 1) 旧的 C++ 头文件，如 iostream.h、fstream.h 等将会继续被支持，尽管它们不在官方标准中。`这些头文件的内容不在命名空间 std 中`。
>
> 2) 新的 C++ 头文件，如 iostream、fstream 等包含的基本功能和对应的旧版头文件相似，但`头文件的内容在命名空间 std 中`。
> 3) 标准 C 头文件如 stdio.h、stdlib.h 等继续被支持。`头文件的内容不在 std 中`。
> 4) 具有 C 库功能的新 C++ 头文件具有如 cstdio、cstdlib 这样的名字。它们提供的内容和相应的旧的 C 头文件相同，只是内容在 std 中
> 5) 对于不带`.h` 的头文件，所有的符号都位于命名空间 std 中，使用时需要声明命名空间 std；对于带`.h` 的头文件，没有使用任何命名空间，所有符号都位于全局作用域。

#### 1. 函数默认参数

- C++ 规定，默认参数只能放在形参列表的最后，而且一旦为某个形参指定了默认值，那么它后面的所有形参都必须有默认值。

```c++
float d = 10.8;
void func(int n, float b=d+2.9, char c='@'){
    cout<<n<<", "<<b<<", "<<c<<endl;
}
```

#### 2. 函数重载

- **原理**

C++ 代码在`编译时会根据参数列表对函数进行重命名`，例如 `void Swap(int a, int b)` 会被重命名为`_Swap_int_int`，`void Swap(float x, float y)` 会被重命名为`_Swap_float_float`。当发生函数调用时，编译器会根据传入的实参去逐个匹配，以选择对应的函数，如果匹配失败，编译器就会报错，这叫做重载决议（Overload Resolution）。

- 二义性  

#### 3. 类和对象

> 类是创建对象的模板，一个类可以创建多个对象，每个对象都是类类型的一个变量；创建对象的过程也叫类的实例化。每个对象都是类的一个具体实例（Ins[tan](https://xinbaoku.com/ref/tan.html)ce），拥有类的成员变量和成员函数。
>
> 类只是一种复杂数据类型的声明，不占用内存空间。而对象是类这种数据类型的一个变量，或者说是通过类这种数据类型创建出来的一份实实在在的数据，所以占用内存空间。

- 一种是`在栈上创建，形式和定义普通变量类似`；Student stu;   Student *pStu = &stu;
- 一种是`在堆上使用 new 关键字创建`，必须要用一个指针指向它，读者要记得 delete 掉不再使用的对象。Student *pStu = **new** Student;

> - 在类的内部（定义类的代码内部），无论成员被声明为 public、protected 还是 private，都是可以互相访问的，没有访问权限的限制。
> - 在类的外部（定义类的代码之外），只能通过对象访问成员，并且通过对象只能访问 public 属性的成员，不能访问 private、protected 属性的成员。

```c++
#include <iostream>
using namespace std;

//类的声明
class Student{
private:  //私有的
    char *m_name;
    int m_age;
    float m_score;
public:  //共有的
    void setname(char *name);
    void setage(int age);
    void setscore(float score);
    void show();
};

//成员函数的定义
void Student::setname(char *name){
    m_name = name;
}
void Student::setage(int age){
    m_age = age;
}
void Student::setscore(float score){
    m_score = score;
}
void Student::show(){
    cout<<m_name<<"的年龄是"<<m_age<<"，成绩是"<<m_score<<endl;
}

int main(){
    //在栈上创建对象
    Student stu;
    stu.setname("小明");
    stu.setage(15);
    stu.setscore(92.5f);
    stu.show();

    //在堆上创建对象
    Student *pstu = new Student;
    pstu -> setname("李华");
    pstu -> setage(16);
    pstu -> setscore(96);
    pstu -> show();

    return 0;
}
```

##### .1. 构造函数

- 名字和类名相同，没有返回值，不需要用户显式调用（用户也不能调用），而是`在创建对象时自动执行`。
- 一旦用户自己定义了构造函数，不管有几个，也不管形参如何，编译器都不再自动生成。

```c++
//采用初始化列表
Student::Student(char *name, int age, float score): m_name(name), m_age(age), m_score(score){
    //TODO:
}
```

```c++
Student(string name = "", int age = 0, float score = 0.0f);  //普通构造函数
Student(const Student &stu);  //拷贝构造函数（声明）
```

- **为什么必须是当前类的引用呢？**
  - 如果拷贝构造函数的参数不是当前类的引用，而是当前类的对象，那么在调用拷贝构造函数时，会将`另外一个对象直接传递给形参，这本身就是一次拷贝，会再次调用拷贝构造函数`，然后又将一个对象直接传递给了形参，将继续调用拷贝构造函数…… 这个过程会一直持续下去，没有尽头，陷入死循环。只有当参数是当前类的引用时，才不会导致再次调用拷贝构造函数，这不仅是逻辑上的要求，也是 C++ 语法的要求。
- 当以拷贝的方式初始化一个对象时，会调用拷贝构造函数；当给一个对象赋值时，会调用重载过的赋值运算符。 即使我们没有显式的重载赋值运算符，编译器也会以默认地方式重载它。默认重载的赋值运算符功能很简单，就是将原有对象的所有成员变量一一赋值给新对象，这和默认拷贝构造函数的功能类似。

```c++
Array &Array::operator=(const Array &arr){  //重载赋值运算符
    if( this != &arr){  //判断是否是给自己赋值
        this->m_len = arr.m_len;
        free(this->m_p);  //释放原来的内存
        this->m_p = (int*)calloc( this->m_len, sizeof(int) );
        memcpy( this->m_p, arr.m_p, m_len * sizeof(int) );
    }
    return *this;
}
```

##### .2. 析构函数

- 析构函数（Destructor）也是一种特殊的成员函数，没有返回值，不需要程序员显式调用（程序员也没法显式调用），而是在销毁对象时自动执行。构造函数的名字和类名相同，而析构函数的名字是在类名前面加一个 `~` 符号。
- 析构函数`没有参数，不能被重载`，因此一个类只能有一个析构函数。如果用户没有定义，编译器会自动生成一个默认的析构函数。
- 在函数内部创建的对象是局部对象，它和局部变量类似，位于栈区，函数执行结束时会调用这些对象的析构函数。
- new 创建的对象位于堆区，通过 delete 删除时才会调用析构函数；如果没有 delete，析构函数就不会被执行。

##### .3. this指针

- this 是 [C++](https://xinbaoku.com/cplus/) 中的一个关键字，也是一个 const [指针](https://xinbaoku.com/c/80/)，它指向当前对象，通过它可以访问当前对象的所有成员。
- this `只能用在类的内部`，通过 this 可以访问类的所有成员，包括 private、protected、public 属性的。
- 只有当对象被创建后 this 才有意义，因此不能在 static 成员函数中使用（后续会讲到 static 成员）。

##### .4. 静态变量

- static 成员变量属于类，不属于某个具体的对象，即使创建多个对象，也只为 m_total 分配一份内存，所有对象使用的都是这份内存中的数据。
- static 成员变量的内存既不是在声明类时分配，也不是在创建对象时分配，而是在（类外）初始化时分配。反过来说，没有在类外初始化的 static 成员变量不能使用。
- static 成员变量不占用对象的内存，而是在所有对象之外开辟内存，即使不创建对象也可以访问。
- 静态成员变量必须初始化，而且只能在类体外进行
- `初始化时可以赋初值，也可以不赋值。如果不赋值，那么会被默认初始化为 0。``全局数据区的变量都有默认的初始值 0`，而`动态数据区（堆区、栈区）变量的默认值是不确定的`，一般认为是垃圾值。

```c++
int Student::m_total = 10;
//通过类类访问 static 成员变量
Student::m_total = 10;
//通过对象来访问 static 成员变量
Student stu("小明", 15, 92.5f);
stu.m_total = 20;
//通过对象指针来访问 static 成员变量
Student *pstu = new Student("李华", 16, 96);
pstu -> m_total = 20;
```

##### .5. 静态成员函数

- 普通成员函数可以访问所有成员（包括成员变量和成员函数），`静态成员函数只能访问静态成员`。普通成员函数有 this 指针，可以访问类中的任意成员；而静态成员函数没有 this 指针，只能访问静态成员（包括静态成员变量和静态成员函数）。
- 编译器在编译一个`普通成员函数时，会隐式地增加一个形参 this，并把当前对象的地址赋值给 this`，所以普通成员函数只能在创建对象后通过对象来调用，因为它需要当前对象的地址。而静态成员函数可以通过类来直接调用，编译器不会为它增加形参 this，它不需要当前对象的地址，所以不管有没有创建对象，都可以调用静态成员函数。

##### .6. const

- 函数开头的 const 用来修饰函数的返回值，表示返回值是 const 类型，也就是不能被修改，例如 `const char * getname()`。
- 函数头部的结尾加上 const 表示常成员函数，这种函数只能读取成员变量的值，而不能修改成员变量的值，例如 `char * getname() const`。
- const 也可以用来修饰对象，称为常对象。一旦将对象定义为常对象之后，就`只能调用类的 const 成员（包括 const 成员变量和 const 成员函数）了`。

##### .7. 友元函数

- 借助友元（friend），可以`使得其他类中的成员函数以及全局范围内的函数访问当前类的 private 成员`。
- `在友元函数中不能直接访问类的成员，必须要借助对象`
- `友元的关系是单向的而不是双向的`。如果声明了类 B 是类 A 的友元类，不等于类 A 是类 B 的友元类，类 A 中的成员函数不能访问类 B 中的 private 成员。
- `友元的关系不能传递`。如果类 B 是类 A 的友元类，类 C 是类 B 的友元类，不等于类 C 是类 A 的友元类。

```c++
#include <iostream>
using namespace std;

class Student{
public:
    Student(char *name, int age, float score);
public:
    friend void show(Student *pstu);  //将show()声明为友元函数
private:
    char *m_name;
    int m_age;
    float m_score;
};

Student::Student(char *name, int age, float score): m_name(name), m_age(age), m_score(score){ }

//非成员函数
void show(Student *pstu){
    cout<<pstu->m_name<<"的年龄是 "<<pstu->m_age<<"，成绩是 "<<pstu->m_score<<endl;
}

int main(){
    Student stu("小明", 15, 90.6);
    show(&stu);  //调用友元函数
    Student *pstu = new Student("李磊", 16, 80.5);
    show(pstu);  //调用友元函数

    return 0;
}
```

##### .8.  class & struct

- 使用 class 时，类中的成员默认都是 private 属性的；而使用 struct 时，结构体中的成员默认都是 public 属性的。
- class 继承默认是 private 继承，而 struct 继承默认是 public 继承。
- class 可以使用模板，而 struct 不能。

##### .9. string函数

- 变量 s2 在定义的同时被初始化为 "c plus plus"。与 C 风格的字符串不同，`string 的结尾没有结束标志'\0'`。

```c++
#include <iostream>
#include <string>
using namespace std;

int main(){
    string s1;  //变量 s1 只是定义但没有初始化，编译器会将默认值赋给 s1，默认值是 ""，也即空字符串。
    string s2 = "c plus plus"; //变量 s2 在定义的同时被初始化为 "c plus plus"。与 C 风格的字符串不同，string 的结尾没有结束标志'\0'。
    string s3 = s2;
    string s4 (5, 's');
    return 0;
}
```

#### 4. 引用

- 在`将引用作为函数返回值时不能返回局部数据（例如局部变量、局部对象、局部数组等）的引用`，因为当函数调用完成后局部数据就会被销毁，有可能在下次使用时数据就不存在了，C++ 编译器检测到该行为时也会给出警告。
- `引用变量在功能上等于一个指针常量`，即`一旦指向某一个单元就不能在指向别处`。在底层，引用变量由指针按照指针常量的方式实现。存放被引用对象的地址, 利用特殊手段能够找到这个引用变量的地址并修改其自身在内存中的值，从而实现与其他对象的绑定。

```c++
#include <iostream>
using namespace std;
void swap1(int a, int b);
void swap2(int *p1, int *p2);
void swap3(int &r1, int &r2);
int main() {
    int num1, num2;
    cout << "Input two integers: ";
    cin >> num1 >> num2;
    swap1(num1, num2);
    cout << num1 << " " << num2 << endl;
    cout << "Input two integers: ";
    cin >> num1 >> num2;
    swap2(&num1, &num2);
    cout << num1 << " " << num2 << endl;
    cout << "Input two integers: ";
    cin >> num1 >> num2;
    swap3(num1, num2);
    cout << num1 << " " << num2 << endl;
    return 0;
}
//直接传递参数内容，不能达到交换两个数的值的目的
void swap1(int a, int b) {
    int temp = a;
    a = b;
    b = temp;
}
//传递指针，能够达到交换两个数的值的目的
void swap2(int *p1, int *p2) {
    int temp = *p1;
    *p1 = *p2;
    *p2 = temp;
}
//按引用传参，能够达到交换两个数的值的目的。
void swap3(int &r1, int &r2) {
    int temp = r1;
    r1 = r2;
    r2 = temp;
}
```

#### 5. 继承与派生

- ` 基类成员在派生类中的访问权限不得高于继承方式中指定的权限`。继承方式中的 public、protected、private 是用来指明基类成员在派生类中的最高访问权限的。
- 基类的 private 成员是能够被继承的，并且（成员变量）会占用派生类对象的内存，`它只是在派生类中不可见，导致无法使用罢了`。
- using 只能改变基类中 public 和 protected 成员的访问权限，不能改变 private 成员的访问权限，因为基类中 private 成员在派生类中是不可见的，根本不能使用，所以基类中的 private 成员在派生类中无论如何都不能访问。
- `基类构造函数总是被优先调用`，这说明创建派生类对象时，会先调用基类构造函数，再调用派生类构造函数
- 而销毁派生类对象时，析构函数的执行顺序和继承顺序相反，`即先执行派生类析构函数`，再执行基类析构函数。

#### 6. 多态

- 可以通过`基类指针对所有派生类（包括直接派生和间接派生）的成员变量和成员函数`进行 “全方位” 的访问，尤其是成员函数。如果没有多态，我们只能访问成员变量。
- 必须存在继承关系；
- 继承关系中必须有同名的虚函数，并且它们是覆盖关系（函数原型相同）。
- 存在基类的指针，通过该指针调用虚函数。

```c++
#include <iostream>
using namespace std;

//基类Base
class Base{
public:
    virtual void func();
    virtual void func(int);
};
void Base::func(){
    cout<<"void Base::func()"<<endl;
}
void Base::func(int n){
    cout<<"void Base::func(int)"<<endl;
}

//派生类Derived
class Derived: public Base{
public:
    void func();
    void func(char *);
};
void Derived::func(){
    cout<<"void Derived::func()"<<endl;
}
void Derived::func(char *str){
    cout<<"void Derived::func(char *)"<<endl;
}

int main(){
    Base *p = new Derived();
    p -> func();  //输出void Derived::func()
    p -> func(10);  //输出void Base::func(int)
    p -> func("http://c.xinbaoku.com");  //compile error

    return 0;
}
```

- `包含纯虚函数的类称为抽象类`（Abstract Class）。之所以说它抽象，是因为它无法实例化，也就是无法创建对象。原因很明显，`纯虚函数没有函数体，不是完整的函数，无法调用，也无法为其分配内存空间`。

```c++
#include <iostream>
using namespace std;
//线
class Line{
public:
    Line(float len);
    virtual float area() = 0;
    virtual float volume() = 0;
protected:
    float m_len;
};
Line::Line(float len): m_len(len){ }
//矩形
class Rec: public Line{
public:
    Rec(float len, float width);
    float area();
protected:
    float m_width;
};
Rec::Rec(float len, float width): Line(len), m_width(width){ }
float Rec::area(){ return m_len * m_width; }
//长方体
class Cuboid: public Rec{
public:
    Cuboid(float len, float width, float height);
    float area();
    float volume();
protected:
    float m_height;
};
Cuboid::Cuboid(float len, float width, float height): Rec(len, width), m_height(height){ }
float Cuboid::area(){ return 2 * ( m_len*m_width + m_len*m_height + m_width*m_height); }
float Cuboid::volume(){ return m_len * m_width * m_height; }
//正方体
class Cube: public Cuboid{
public:
    Cube(float len);
    float area();
    float volume();
};
Cube::Cube(float len): Cuboid(len, len, len){ }
float Cube::area(){ return 6 * m_len * m_len; }
float Cube::volume(){ return m_len * m_len * m_len; }
int main(){
    Line *p = new Cuboid(10, 20, 30);
    cout<<"The area of Cuboid is "<<p->area()<<endl;
    cout<<"The volume of Cuboid is "<<p->volume()<<endl;
  
    p = new Cube(15);
    cout<<"The area of Cube is "<<p->area()<<endl;
    cout<<"The volume of Cube is "<<p->volume()<<endl;
    return 0;
}
```

#### 7. 运算符重载

- 重载不能改变运算符的优先级和结合性

- 并不是所有的运算符都可以重载。能够重载的运算符包括：

  ```
  \+ - * / % ^ & | ~ ! = < > += -= *= /= %= ^= &= |=  << >> <<= >>= == != <= >= && || ++ -- , ->* -> () []  new new[] delete delete[]
  ```

- 运算符重载函数不能有默认的参数，否则就改变了运算符操作数的个数
- 运算符重载函数既可以作为类的成员函数，也可以作为全局函数。

```c++
#include <iostream>
using namespace std;

class complex{
public:
    complex();
    complex(double real, double imag);
public:
    //声明运算符重载
    complex operator+(const complex &A) const;
    void display() const;
private:
    double m_real;  //实部
    double m_imag;  //虚部
};

complex::complex(): m_real(0.0), m_imag(0.0){ }
complex::complex(double real, double imag): m_real(real), m_imag(imag){ }

//实现运算符重载
complex complex::operator+(const complex &A) const{
    complex B;
    B.m_real = this->m_real + A.m_real;
    B.m_imag = this->m_imag + A.m_imag;
    return B;
}

void complex::display() const{
    cout<<m_real<<" + "<<m_imag<<"i"<<endl;
}

int main(){
    complex c1(4.3, 5.8);
    complex c2(2.4, 3.7);
    complex c3;
    c3 = c1 + c2;
    c3.display();
 
    return 0;
}
```

```c++
//复数类
class Complex{
public:  //构造函数
    Complex(double real = 0.0, double imag = 0.0): m_real(real), m_imag(imag){ }
public:  //运算符重载
    //以全局函数的形式重载
    friend Complex operator+(const Complex &c1, const Complex &c2);
    friend Complex operator-(const Complex &c1, const Complex &c2);
    friend Complex operator*(const Complex &c1, const Complex &c2);
    friend Complex operator/(const Complex &c1, const Complex &c2);
    friend bool operator==(const Complex &c1, const Complex &c2);
    friend bool operator!=(const Complex &c1, const Complex &c2);
    //以成员函数的形式重载
    Complex & operator+=(const Complex &c);
    Complex & operator-=(const Complex &c);
    Complex & operator*=(const Complex &c);
    Complex & operator/=(const Complex &c);
public:  //成员函数
    double real() const{ return m_real; }
    double imag() const{ return m_imag; }
private:
    double m_real;  //实部
    double m_imag;  //虚部
};
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/c-%E5%9F%BA%E7%A1%80/  

