# Flask_BookManageSystem


### 1. JavaScript
####  1.1. 函数
```javascript
var cars = ["Saab", "Volvo", "BMW"];
var txt = "string";
var b2=new Boolean(1);
var x = Math.PI; // 返回PI
var y = Math.sqrt(16); // 返回16的平方根
var num = new Number(value);
try {
    adddlert("Welcome");
}
catch(err) {
    document.getElementById("demo").innerHTML = 
    err.name + "<br>" + err.message;
}
```
#### 1.2. DOM

在 HTML DOM (Document Object Model) 中 , 每一个元素都是 **节点**:

- `文档是一个文档节点`。
- 所有的HTML元素都是`元素节点`。
- 所有 HTML 属性都是`属性节点`。
- 文本插入到 HTML 元素是`文本节点`。
- 注释是注释节点。

//HTMLCollection 是 HTML 元素的集合。
//HTMLCollection 对象类似一个包含 HTML 元素的数组列表。
//getElementsByTagName() 方法返回的就是一个 HTMLCollection 对象。


| [decodeURI()](https://www.runoob.com/jsref/jsref-decodeuri.html) | 解码某个编码的 URI。                                 |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [decodeURIComponent()](https://www.runoob.com/jsref/jsref-decodeuricomponent.html) | 解码一个编码的 URI 组件。                            |
| [encodeURI()](https://www.runoob.com/jsref/jsref-encodeuri.html) | 把字符串编码为 URI。                                 |
| [encodeURIComponent()](https://www.runoob.com/jsref/jsref-encodeuricomponent.html) | 把字符串编码为 URI 组件。                            |
| [escape()](https://www.runoob.com/jsref/jsref-escape.html)   | 对字符串进行编码。                                   |
| [eval()](https://www.runoob.com/jsref/jsref-eval.html)       | `计算 JavaScript 字符串，并把它作为脚本代码`来执行。 |
| [isFinite()](https://www.runoob.com/jsref/jsref-isfinite.html) | 检查某个值是否为有穷大的数。                         |
| [isNaN()](https://www.runoob.com/jsref/jsref-isnan.html)     | 检查某个值是否是数字。                               |
| [Number()](https://www.runoob.com/jsref/jsref-number.html)   | 把对象的值转换为数字。                               |
| [parseFloat()](https://www.runoob.com/jsref/jsref-parsefloat.html) | 解析一个`字符串并返回一个浮点数`。                   |
| [parseInt()](https://www.runoob.com/jsref/jsref-parseint.html) | 解析一个`字符串并返回一个整数`。                     |
| [String()](https://www.runoob.com/jsref/jsref-string.html)   | 把对象的值转换为字符串。                             |
| [unescape()](https://www.runoob.com/jsref/jsref-unescape.html) | 对由 escape() 编码的字符串进行解码。                 |

| 方法                                                         | 描述                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [assert()](https://www.runoob.com/jsref/met-console-assert.html) | 如果断言为 false，则在信息到控制台输出错误信息。             |
| [clear()](https://www.runoob.com/jsref/met-console-clear.html) | 清除控制台上的信息。                                         |
| [count()](https://www.runoob.com/jsref/met-console-count.html) | 记录` count() 调用次数`，一般用于计数。                      |
| [error()](https://www.runoob.com/jsref/met-console-error.html) | 输出错误信息到控制台                                         |
| [group()](https://www.runoob.com/jsref/met-console-group.html) | 在控制台创建一个信息分组。 一个完整的信息分组以 console.group() 开始，console.groupEnd() 结束 |
| [groupCollapsed()](https://www.runoob.com/jsref/met-console-groupcollapsed.html) | 在控制台创建一个信息分组。 类似 console.group() ，但它默认是折叠的。 |
| [groupEnd()](https://www.runoob.com/jsref/met-console-groupend.html) | 设置当前信息分组结束                                         |
| [info()](https://www.runoob.com/jsref/met-console-info.html) | `控制台输出一条信息`                                         |
| [log()](https://www.runoob.com/jsref/met-console-log.html)   | `控制台输出一条信息`                                         |
| [table()](https://www.runoob.com/jsref/met-console-table.html) | 以表格形式显示数据                                           |
| [time()](https://www.runoob.com/jsref/met-console-time.html) | 计时器，开始计时间，与 timeEnd() 联合使用，用于算出一个操作所花费的准确时间。 |
| [timeEnd()](https://www.runoob.com/jsref/met-console-timeend.html) | 计时结束                                                     |
| [trace()](https://www.runoob.com/jsref/met-console-trace.html) | 显示当前执行的代码在堆栈中的调用路径。                       |
| [warn()](https://www.runoob.com/jsref/met-console-warn.html) | 输出警告信息，信息最前面加一个黄色三角，表示警告             |

#### 1.3. CSS 选择器

- 方式一：document. getElementBy

| getElementById         | id     | 兼容性好，推荐使用  | 如果存在多个id相同的元素，只会返回第一个 |
| ---------------------- | ------ | ------------------- | ---------------------------------------- |
| getElementsByTagName   | 标签名 | 不兼容ie8及以下版本 | 返回所有符合条件的元素的集合             |
| getElementsByName      | name   | 不兼容ie8及以下版本 | 返回所有符合条件的元素的集合             |
| getElementsByClassName | class  | 不兼容ie8及以下版本 | 返回所有符合条件的元素的集合             |

- 方式二： doucument.querySelector('#second');

| [.*class*](http://www.runoob.com/cssref/sel-class.html)      | .intro          | 选择所有class="intro"的元素            | 1    |
| ------------------------------------------------------------ | --------------- | -------------------------------------- | ---- |
| [#*id*](http://www.runoob.com/cssref/sel-id.html)            | #firstname      | 选择所有id="firstname"的元素           | 1    |
| [*](http://www.runoob.com/cssref/sel-all.html)               | *               | 选择所有元素                           | 2    |
| *[element](http://www.runoob.com/cssref/sel-element.html)*   | p               | 选择所有<p>元素                        | 1    |
| *[element,element](http://www.runoob.com/cssref/sel-element-comma.html)* | div,p           | 选择所有<div>元素和<p>元素             | 1    |
| [*element* *element*](http://www.runoob.com/cssref/sel-element-element.html) | div p           | 选择<div>元素内的所有<p>元素           | 1    |
| [*element*>*element*](http://www.runoob.com/cssref/sel-element-gt.html) | div>p           | 选择所有父级是 <div> 元素的 <p> 元素   | 2    |
| [*element*+*element*](http://www.runoob.com/cssref/sel-element-pluss.html) | div+p           | 选择所有紧接着<div>元素之后的<p>元素   | 2    |
| [[*attribute*\]](http://www.runoob.com/cssref/sel-attribute.html) | [target]        | 选择所有带有target属性元素             | 2    |
| [[*attribute*=*value*\]](http://www.runoob.com/cssref/sel-attribute-value.html) | [target=-blank] | 选择所有使用target="-blank"的元素      | 2    |
| [[*attribute*~=*value*\]](http://www.runoob.com/cssref/sel-attribute-value-contains.html) | [title~=flower] | 选择标题属性包含单词"flower"的所有元素 | 2    |
| [[*attribute*\|=*language*\]](http://www.runoob.com/cssref/sel-attribute-value-lang.html) | [lang\|=en]     | 选择 lang 属性以 en 为开头的所有元素   | 2    |

#### 1.4. HTML 修改

- `document.write()`直接将内容写入页面的内容流。当文档流执行完毕，会导致页面全部重绘。
- `element.innerHTML`将内容写入某个DOM节点，不会导致页面重绘。
- `document.createElement()`创建多个元素结构清晰。

```javascript
// html 设置元素属性值
element.setAttribute('height', '100px');
element.setAttribute('style', 'height: 100px !important');
element.style.setProperty('height', '300px', 'important');
//第1种方法 ,给元素设置style属性  
$("#hidediv").css("display", "block");  
//第2种方法 ,给元素换class，来实现隐藏div，前提是换的class样式定义好了隐藏属性  
$("#hidediv").attr("class", "blockclass");  
//第3种方法,通过jquery的css方法，设置div隐藏  
$("#blockdiv").css("display", "none");  
$("#hidediv").show();//显示div    
$("#blockdiv").hide();//隐藏div 
```

#### 1.5. js code segment

```javascript
//-------   全屏
function fullscreen(element) {
    if(element.requestFullscreen) {
        element.requestFullscreen();
    } else if(element.mozRequestFullScreen) {
        element.mozRequestFullScreen();
    } else if(element.webkitRequestFullscreen) {
        element.webkitRequestFullscreen();
    } else if(element.msRequestFullscreen) {
        element.msRequestFullscreen();
    }
}
fullscreen(document.documentElement)
//-------   switchcase
let fruit = 'banana';
let drink;
switch (fruit) {
  case 'banana':
    drink = 'banana juice';
    break;
  case 'papaya':
    drink = 'papaya juice';
    break;
  default:
    drink = 'Unknown juice!'
}
console.log(drink) // banana juice
//-------  object entries
const credits = {
  producer: '大迁世界',
  name: '前端小智',
  rating: 9
}
const arr = Object.entries(credits)
console.log(arr)
```

### 2. Flask 

#### 2.1.  参数传递

- **提交表单数据**

> 前端的数据发送与接收
> 1）提交表单数据
> 2）提交JSON数据
>
> 后端的数据接收与响应
> 1）接收GET请求数据
> 2）接收POST请求数据
> 3）响应请求
>
> GET把参数包含在URL中，POST通过request body传递参数。

```javascript
// GET请求
var data = {
    "name": "test",
    "age": 1
};
$.ajax({
    type: 'GET',
    url: /your/url/,
    data: data, // 最终会被转化为查询字符串跟在url后面： /your/url/?name=test&age=1
    dataType: 'json', // 注意：这里是指希望服务端返回json格式的数据
    success: function(data) { // 这里的data就是json格式的数据
    },
    error: function(xhr, type) {
    }
});
```

```javascript
//--- -- POST请求

var data = {}
//如果页面并没有表单，只是input框，请求也只是发送这些值，那么可以直接获取放到data中
data['name'] = $('#name').val()

//如果页面有表单，那么可以利用jquery的serialize()方法获取表单的全部数据
data = $('#form1').serialize();

$.ajax({
    type: 'POST',
    url: /your/url/,
    data: data,
    dataType: 'json', //注意：这里是指希望服务端返回json格式的数据
    success: function(data) { //这里的data就是json格式的数据
    },
    error: function(xhr, type) {
    }
});
```

> A）参数dataType：期望的服务器响应的数据类型，可以是null, xml, script, json
>  B）请求头中的Content-Tpye默认是`Content-Type:application/x-www-form-urlencoded`，所以参数会被编码为 name=xx&age=1 这种格式，提交到后端，后端会当作表单数据处理。

- 提交Json 数据

```javascript
//----        POST一个json数据

var data = {
    “name”: "test",
    "age", 1
}
$.ajax({
    type: 'POST',
    url: /your/url/,
    data: JSON.stringify(data), //转化为字符串
    contentType: 'application/json; charset=UTF-8',
    dataType: 'json', //注意：这里是指希望服务端返回json格式的数据
    success: function(data) { //这里的data就是json格式的数据
    },
    error: function(xhr, type) {
    }
});
```

#### 2.2. 后端接受处理

```python
#---     接受get 请求数据
# get 请求 www.baidu.com/s?name=python&age=12
name = request.args.get('name', '')
age = int(request.args.get('age', '0'))
#---     接受post 请求数据
#---1. 接收表单数据
name = request.form.get('name', '')
age = int(request.form.get('age', '0'))
#---2. 接收json 数据
data = request.get_json()
#---     返回json 数据
from Flask import jsonify
return jsonify({'ok': True})  #return Response(data=json.dumps({'ok': True}), mimetype='application/json')
```

#### 2.3. 案例

```python
# encoding:utf-8
from flask import Flask, render_template, url_for, request, json,jsonify
app = Flask(__name__)
#设置编码
app.config['JSON_AS_ASCII'] = False
#接收参数，并返回json数据
@app.route('/sendDate', methods=['GET', 'POST'])
def form_data():
   #从request中获取表单请求的参数信息
   title = request.form['title']
   datetime = request.form['datetime']
   quiz = request.form['quiz']
      #此处逻辑代码已经省略...................
   return jsonify({'status': '0', 'errmsg': '登录成功！'})
#测试入口
@app.route('/')
def home():
    return render_template("homepage.html")#homepage.html在templates文件夹下
if __name__ == '__main__':
    app.run()
```

```javascript
$.ajax({
          url:'http://127.0.0.1:5000/sendDate',
  data:"title="+data.field['title']+"&datetime="+data.field['datetime']+"&quiz="+data.field['quiz'],
          type:'post',
          dataType:'json',
          success:function(data){ //后端返回的json数据（此处data为json对象）
              alert(data['errmsg']);
          },
          error:function () {
              alert('异常')
          }
      });
```

```JavaScript
<script type="text/javascript">
	function handle(sid){
		var element = document.querySelector("#"+sid+">p");
		console.log(element)
		element.animate({left:'50%',top:'20px'},300)
		setTimeout(function(){
			element.find("a").css({left:'50%',top:0+"px"}).stop(true).animate({left:'50%',top:'20px'},1000)
		},200) 
		//element.css('visibility','hidden');
		//element.classList.add("animated bounce");
		//element.style.setProperty('--animate-duration', '2s');
		console.log("ID:");   //class="animated bounce"
	}

	function myrefresh()
	{
		//window.location.reload();
		$.ajax({
			url:'http://127.0.0.1:5000/getID',
			data:"title=getdata",
			type:'post',
			dataType:'json',
			success:function(data){ //后端返回的json数据（此处data为json对象）  添加动画
				console.log("ID:" + "demo"+data['id']);   //class="animated bounce"
				handle("demo"+data['id'])
			},
			error:function () {
				console.log('异常')
			}
		});
	} //setTimeout()只执行一次，setInterval()可以执行多次。
	setInterval('myrefresh()',2000); //指定1秒刷新一次
</script>
```

### 3. BookManageSystem

#### 3.1. [flask_script](https://www.cnblogs.com/buyisan/p/8270283.html)

> Flask Script扩展提供向Flask插入外部脚本的功能，包括运行一个开发用的服务器，一个定制的Python shell，设置数据库的脚本，cronjobs，及其他运行在web应用之外的命令行任务；使得脚本和系统分开；

```python
from flask_script import Manager  ，Server
from flask_script import Command  
from debug import app  
manager = Manager(app)  
class Hello(Command):  
    'hello world'  
    def run(self):  
        print 'hello world'  
#自定义命令一：
manager.add_command('hello', Hello())  
# 自定义命令二：
manager.add_command("runserver", Server()) #命令是runserver
if __name__ == '__main__':  
    manager.run()  
```

#### 3.2. SQLAlchemy

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201114233802590.png)

| Engine  | 连接         | 驱动引擎             |
| ------- | ------------ | -------------------- |
| Session | 连接池，事务 | 由此开始查询         |
| Model   | 表           | 类定义               |
| Column  | 列           |                      |
| Query   | 若干行       | 可以链式添加多个条件 |

| Integer  | int      | int               | 整形，32位             |
| -------- | -------- | ----------------- | ---------------------- |
| String   | varchar  | string            | 字符串                 |
| Text     | text     | string            | 长字符串               |
| Float    | float    | float             | 浮点型                 |
| Boolean  | tinyint  | bool              | True / False           |
| Date     | date     | datetime.date     | 存储时间年月日         |
| DateTime | datetime | datetime.datetime | 存储年月日时分秒毫秒等 |
| Time     | time     | datetime.datetime | 存储时分秒             |

```python
engine = create_engine("mysql://user:password@hostname/dbname?charset=utf8",
                       echo=True,
                       pool_size=8,
                       pool_recycle=60*30
                       )
from sqlalchemy.ext.declarative import declarative_base
#declarative_base()是sqlalchemy内部封装的一个方法，通过其构造一个基类，这个基类和它的子类，可以将Python类和数据库表关联映射起来。
Base = declarative_base()
class Users(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String(64), unique=True)
    email = Column(String(64))

    def __init__(self, name, email):
        self.name = name
        self.email = email  
#创建表，如果存在则忽略，执行以上代码，就会发现在db中创建了users表。
Base.metadata.create_all(engine)

from sqlalchemy.orm import sessionmaker
# 创建session
DbSession = sessionmaker(bind=engine)
session = DbSession()
#---- 增加
add_user = Users("test", "test123@qq.com")
session.add(add_user)
session.commit()
#---- 查询
users = session.query(Users).filter_by(id=1).all()
for item in users:
    print(item.name)
users = session.query(Users).filter_by(Users.id == 1).all()
#---- 修改
session.query(Users).filter_by(id=1).update({'name': "Jack"})
#--- or
users = session.query(Users).filter_by(name="Jack").first()
users.name = "test"
session.add(users)
#--- 删除
delete_users = session.query(Users).filter(Users.name == "test").first()
if delete_users:
    session.delete(delete_users)
    session.commit()
#--- or
session.query(Users).filter(Users.name == "test").delete()
session.commit()
```

|                                            |                              |
| :----------------------------------------: | :--------------------------: |
|                   filter                   |          filter_by           |
|   支持所有比较运算符，相等比较用比较用==   |   只能使用"="，"!="和"><"    |
|             过滤用类名.属性名              |         过滤用属性名         |
| 不支持组合查询，只能连续调用filter变相实现 | 参数是**kwargs，支持组合查询 |
|             支持and，or和in等              |                              |

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201114233611631.png)

#### 3.3. flask-wtf&&wtf

> flask-wtf和wtf主要是用于`建立html中的元素和Python中的类的对应关系`，通过在Python代码中操作对应的类，对象等从而`控制html中的元素`。我们需要在python代码中使用flask-wtf和wtf来`定义前端页面的表单`（实际是定义一个表单类），再将对应的`表单对象作为render_template函数的参数`，`传递给相应的template`，之后Jinja模板引擎会将相应的template渲染成html文本，再作为http response返回给用户。

```python
# forms.py
from flask_wtf import FlaskForm
from wtforms import StringField, BooleanField, PasswordField
from wtforms.validators import DataRequired

# 定义的表单都需要继承自FlaskForm
class LoginForm(FlaskForm):
    # 域初始化时，第一个参数是设置label属性的
    username = StringField('User Name', validators=[DataRequired()]) #比如StringField代表的是<input type="text">元素
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('remember me', default=False)
# app.py
@app.route('/login')
def login():
    form = LoginForm()
    return render_template('login.html', title="Sign In", form=form)
```

```html
#LoginForm 对应html 模板
{% extends "layout.html" %}
<html>
    <head>
        <title>Login Page</title>
    </head>
    <body>
        <form action="{{ url_for("login") }}" method="POST">
        <p>
            User Name:<br>
            <input type="text" name="username" /><br>
        </p>
        <p>
            Password:</br>
            <input type="password" name="password" /><br>
        </p>
        <p>
            <input type="checkbox" name="remember_me"/>Remember Me
        </p>
            {{ form.csrf_token }} 
        </form>
    </body>
</html>
#----- 对应template
<!-- 模板的语法应当符合Jinja语法 -->
<!-- extend from base layout -->
{% extends "base.html" %}

{% block content %}
  <h1>Sign In</h1>
  <form action="{{ url_for("login") }}" method="post" name="login">
      {{ form.csrf_token }}
      <p>
          {{ form.username.label }}<br>
          {{ form.username(size=80) }}<br>
      </p>
      <p>
          {{ form.password.label }}<br>
          <!-- 我们可以传递input标签的属性，这里传递的是size属性 -->
          {{ form.password(size=80) }}<br>
      </p>
      <p>{{ form.remember_me }} Remember Me</p>
      <p><input type="submit" value="Sign In"></p>
  </form>
{% endblock %}
```

#### 3.4. flask_login

- 首次登录

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201115000918252.png)

```python
def login_user(user, remember=False, force=False, fresh=True):
    if not force and not user.is_active:
        return False
    #获取User对象的get_id method，然后执行，从而获取到用户的ID
    user_id = getattr(user, current_app.login_manager.id_attribute)()
    session['user_id'] = user_id
    session['_fresh'] = fresh
    session['_id'] = current_app.login_manager._session_identifier_generator()

    if remember:
        session['remember'] = 'set'
#user对象存储进当前的request context中，_request_ctx_stack是一个LocalStack对象，top属性指向的就是当前的request context。
    _request_ctx_stack.top.user = user
    #通过send来发射此signal，当注册监听此signal的回调函数收到此signal之后就会执行函数。这里send有两个参数，第一个参数是sender对象，此处通过current_app._get_current_object()来获取当前的app对象，即此signal的sender设为当前的应用；第二个参数是该signal携带的数据，此处将user对象做为signal的数据传递给相应的回调函数。
    user_logged_in.send(current_app._get_current_object(), user=_get_user())
    return True
```

> 注意：Flask的session是以cookie为基础，但是是在Server端使用secret key并使用AES之类的对称加密算法进行加密的，然后将加密后的cookie发送给客户端。由于是加密后的数据，客户端无法篡改数据，也无法获知session中的信息，只能保存该session信息，在之后的请求中携带该session信息。

- 非首次登录

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201115001837082.png)

```python
@app.route('/')
@app.route('/main')
@login_required
def main():
    return render_template(
        'main.html', username=current_user.username)
        
# login_required 会执行一下代码
# flask_login/utils.py
def login_required(func):
    @wraps(func)
    def decorated_view(*args, **kwargs):
        # 如果request method为例外method，即在EXEMPT_METHODS中的method，可以不必鉴权
        if request.method in EXEMPT_METHODS:
            return func(*args, **kwargs)

        # 如果_login_disabled为True则不必鉴权
        elif current_app.login_manager._login_disabled:
            return func(*args, **kwargs)

        # 正常鉴权
        elif not current_user.is_authenticated:
            return current_app.login_manager.unauthorized()
        return func(*args, **kwargs)
    return decorated_view
```



```python
# use login manager to manage session
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.session_protection = 'basic'
login_manager.login_view = 'login'       #  没有权限重定向 页面
login_manager.login_message = u"请先登录。"        #自定义 flash 的消息

##当一个请求过来的时候，如果 ctx.user 没有值，那么 flask-login 就会使用 session 中 session['user_id'] 作为参数，调用 login_manager 中使用 user_loader 装饰器设置的 callback 函数加载用户，需要注意的是，如果指定的 user_id 无效，不应该抛出异常，而是应该返回 None。

@login_manager.user_loader
def load_user(admin_id):
    return Admin.query.get(int(admin_id))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('您已经登出！')
    return redirect(url_for('login'))
```

#### 3.5. url_for()

```JavaScript
<link rel="stylesheet" href="{{url_for('static',filename='css/index.css')}}">
```

```python
# 动态路由， student_id：是可变部分
@app.route('/student_list/<student_id>/')
def student_list(student_id):
    return '学生{}号的信息'.format(student_id)
# 动态路由  过滤 string， int， float, path, uuid
@app.route('/student_list/<int:student_id>/')
def article_detail(student_id):
    return '学生{}号的信息'.format(student_id)
# 动态路由   自定义 url_path
@app.route('/<any(student,class):url_path>/<id>/')
def item(url_path, id):
    if url_path == 'student':
        return '学生{}详情'.format(id)
    else:
        return '班级{}详情'.format(id)
```

- 给指定函数构造 URL

```python
@app.route('/demo3/')
def demo3():
    school_url = url_for('school', school_level='high', name='college') 
    # 具体要拼接的查询参数 以关键字实参的形式写在url_for里
    print(school_url)  #/school/?shool_level=high&name=college
    return school_url
@app.route('/school/')
def school():
    return 'school message'
```

- 访问静态数据

```python
url_for('static', filename='style.css')
#这个文件应该存储在文件系统上的 static/style.css。
```

#### 3.6. Layui 模板学习使用

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201115094825916.png)

#### 3.7. 学习记录 

- 基本代码都有了，学习只关注了flask那登录到主页那部分，其他的处理逻辑没有具体细看。如果不考虑css中 layui 如何使用；
- 以后自己写python的管理系统，基于这个代码修改；
- 登录，数据库连接，请求啥的都齐全了；
- 后续自己编写相关系统的时候可以具体在学习 block 块使用规则。
- 项目效果； 如果做`物联网可以添加卡片式的那种显示效果`；

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20201115095655445.png)

学习链接

- [javascript document](https://www.runoob.com/jsref/jsref-obj-array.html)
- [Fask-login](https://www.jianshu.com/p/5ba6a956d504)



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/flask_bookmanagesystem/  

