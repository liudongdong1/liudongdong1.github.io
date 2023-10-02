# Tornado


> Tornado 是一个`Python web框架和异步网络库` 起初由 FriendFeed 开发. 通过使用非阻塞网络I/O, Tornado 可以支持上万级的连接，处理 长连接, WebSockets, 和其他 需要与每个用户保持长久连接的应用。包括一下四个部分
>
> - web框架 (包括创建web应用的 `RequestHandler 类`，还有很多其他支持的类).
>
> - `HTTP的客户端和服务端`实现 (`HTTPServer and AsyncHTTPClient`).
> - `异步网络库 `(IOLoop and IOStream), 为HTTP组件提供构建模块，也可以用来实现其他协议.
> - `协程库 (tornado.gen) `允许异步代码写的更直接而不用链式回调的方式.

### 1. 应用结构

> 通常一个Tornado web应用包括一个或者多个` RequestHandler 子类`, 一个可以将收到的请求路由到对应`handler的 Application 对象`,和 一个启动服务的 `main() 函数`.

```python
import tornado.ioloop
import tornado.web
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")
def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])
if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()  #开启tornado的事件驱动循环
```

#### .1. Application 对象

- Application 对象是负责`全局配置的`, 包括`映射请求转发给处理程序的路由表`.

> 路由表是 URLSpec 对象(或元组)的列表, 其中每个都包含(至少)一个正则 表达式和一个处理类. 顺序问题; 第一个匹配的规则会被使用. 如果正则表达 式包含捕获组, 这些组会被作为 路径参数 传递给处理函数的HTTP方法. 如果一个字典作为 URLSpec 的第三个参数被传递, 它会作为 初始参数 传递给 RequestHandler.initialize. 最后 URLSpec 可能有一个名字 , 这将允许它被 RequestHandler.reverse_url 使用.

#### .2. RequestHandler 子类

> - Tornado web 应用程序的大部分工作是在 RequestHandler 子类下完成的. `处理子类的主入口点是一个命名为处理HTTP方法的函数`: get(), post(), 等等. 每个处理程序可以定义一个或者多个这种方法来处理不同 的HTTP动作. 如上所述, 这些方法将被匹配路由规则的捕获组对应的参数调用.
> - 在处理程序中, 调用方法如 RequestHandler.render 或者 RequestHandler.write 产生一个响应. render() 通过名字加载一个 Template 并使用给定的参数渲染它. write() 被用于非模板基础的输 出; 它接受字符串, 字节, 和字典(字典会被编码成JSON).
> - 在 RequestHandler 中的很多方法的设计是为了在子类中复写和在整个应用 中使用. 常用的方法是`定义一个 BaseHandler 类`,` 复写一些方法例如 write_error 和 get_current_user `然后子类继承使用你自己的 BaseHandler 而不是 RequestHandler 在你所有具体的处理程序中。

#### .3. 模板UI

```python
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        items = ["Item 1", "Item 2", "Item 3"]
        self.render("template.html", title="My title", items=items)
```

```
Tornado 模版支持 *控制语句 (control statements)* 和 *表达式 (expressions)* .控制语句被 `{%` and `%}` 包裹着, 例如.,`{% if len(items) > 2 %}`. 表达式被 `{{` 和`}}` 围绕, 再例如., `{{ items[0] }}`.
```

#### .4. 全局变量

> tornado的某些代码只希望运行一次，可让目标对象成为全局变量，如果是Handler级别的全局变量，那么可以直接将全局变量申请放在Handler类里面。
>
> 而如果你想某个全部变量多个Handler之间共用，也就是该全局变量是Application级别的

```python
class PageTwoHandler(tornado.web.RequestHandler):
    def initialize(self, configs):
        self.configs = configs

    def get(self):
        self.write(str(self.configs) + "\n")


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [('/pageone', PageOneHandler, {'configs' : configs}),
                ('/pagetwo', PageTwoHandler, {'configs': configs})]
        settings = dict(template_path='/templates',
                    static_path='/static', debug=False)
        tornado.web.Application.__init__(self, handlers, **settings)
```

#### .5. 关闭

```python
async def shutdown():
    periodic_task.stop()
    http_server.stop()
    for client in ws_clients.values():
        client['handler'].close()
    await gen.sleep(1)
    ioloop.IOLoop.current().stop()

def exit_handler(sig, frame):
    ioloop.IOLoop.instance().add_callback_from_signal(shutdown)
    
if __name__ == '__main__':
    signal.signal(signal.SIGTERM, exit_handler)
    signal.signal(signal.SIGINT,  exit_handler)
```

### 2. 相关demo

#### .1. 同步处理

```python
from tornado.httpclient import HTTPClient
def synchronous_fetch(url):
    http_client = HTTPClient()
    response = http_client.fetch(url)
    return response.body
```

#### .2. 异步处理

```python
from tornado.httpclient import AsyncHTTPClient
 
def asynchronous_fetch(url, callback):
    http_client = AsyncHTTPClient()
    def handle_response(response):
        callback(response.body)
    http_client.fetch(url, callback=handle_response)
```

```python
from tornado.concurrent import Future
def async_fetch_future(url):
    http_client = AsyncHTTPClient()
    my_future = Future()
    fetch_future = http_client.fetch(url)
    fetch_future.add_done_callback(
        lambda f: my_future.set_result(f.result()))
    return my_future
```

#### .3. 协程

> Tornado 中推荐用 **协程** 来编写异步代码. 协程使用 Python 中的关键字 `yield`来替代链式回调来实现挂起和继续程序的执行。
>
> 从 Tornado 4.3 开始, 在协程基础上你可以使用这些来代替 `yield`.简单的通过使用 `async def foo()` 来代替 `@gen.coroutine` 装饰器, 用 `await` 来代替 yield。

```python
from tornado import gen
@gen.coroutine
def fetch_coroutine(url):
    http_client = AsyncHTTPClient()
    response = yield http_client.fetch(url)
    # 在 Python 3.3 之前的版本中, 从生成器函数
    # 返回一个值是不允许的,你必须用
    #   raise gen.Return(response.body)
    # 来代替
    return response.body
```

- 生成器案例

> 一个含有 `yield` 的函数时一个 **生成器** . 所有生成器都是异步的;调用它时将会返回一个对象而不是将函数运行完成.`@gen.coroutine` 修饰器通过 `yield` 表达式通过产生一个 [`Future`](https://tornado-zh-cn.readthedocs.io/zh_CN/latest/concurrent.html#tornado.concurrent.Future) 对象和生成器进行通信.

```python
# Simplified inner loop of tornado.gen.Runner
def run(self):
    # send(x) makes the current yield return x.
    # It returns when the next yield is reached
    future = self.gen.send(self.next)
    def callback(f):
        self.next = f.result()
        self.run()
    future.add_done_callback(callback)
```

#### .4. Queue示例

```python
import time
from datetime import timedelta
 
try:
    from HTMLParser import HTMLParser
    from urlparse import urljoin, urldefrag
except ImportError:
    from html.parser import HTMLParser
    from urllib.parse import urljoin, urldefrag
 
from tornado import httpclient, gen, ioloop, queues
 
base_url = 'http://www.tornadoweb.org/en/stable/'
concurrency = 10
 
 
@gen.coroutine
def get_links_from_url(url):
    """Download the page at `url` and parse it for links.
 
    Returned links have had the fragment after `#` removed, and have been made
    absolute so, e.g. the URL 'gen.html#tornado.gen.coroutine' becomes
    'http://www.tornadoweb.org/en/stable/gen.html'.
    """
    try:
        response = yield httpclient.AsyncHTTPClient().fetch(url)
        print('fetched %s' % url)
 
        html = response.body if isinstance(response.body, str) \
            else response.body.decode()
        urls = [urljoin(url, remove_fragment(new_url))
                for new_url in get_links(html)]
    except Exception as e:
        print('Exception: %s%s' % (e, url))
        raise gen.Return([])
 
    raise gen.Return(urls)
 
 
def remove_fragment(url):
    pure_url, frag = urldefrag(url)
    return pure_url
 
 
def get_links(html):
    class URLSeeker(HTMLParser):
        def __init__(self):
            HTMLParser.__init__(self)
            self.urls = []
 
        def handle_starttag(self, tag, attrs):
            href = dict(attrs).get('href')
            if href and tag == 'a':
                self.urls.append(href)
 
    url_seeker = URLSeeker()
    url_seeker.feed(html)
    return url_seeker.urls
 
 
@gen.coroutine
def main():
    q = queues.Queue()
    start = time.time()
    fetching, fetched = set(), set()
 
    @gen.coroutine
    def fetch_url():
        current_url = yield q.get()
        try:
            if current_url in fetching:
                return
 
            print('fetching %s' % current_url)
            fetching.add(current_url)
            urls = yield get_links_from_url(current_url)
            fetched.add(current_url)
 
            for new_url in urls:
                # Only follow links beneath the base URL
                if new_url.startswith(base_url):
                    yield q.put(new_url)
 
        finally:
            q.task_done()
 
    @gen.coroutine
    def worker():
        while True:
            yield fetch_url()
 
    q.put(base_url)
 
    # Start workers, then wait for the work queue to be empty.
    for _ in range(concurrency):
        worker()
    yield q.join(timeout=timedelta(seconds=300))
    assert fetching == fetched
    print('Done in %d seconds, fetched %s URLs.' % (
        time.time() - start, len(fetched)))
 
 
if __name__ == '__main__':
    import logging
    logging.basicConfig()
    io_loop = ioloop.IOLoop.current()
    io_loop.run_sync(main)
```

### 3. Resource

- https://www.bookstack.cn/read/tornado-zh/6.md

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/tornado/  

