# Crawl_heavens


### 1. 电影爬取

- main.py

```python
# -*- encoding:utf-8 -*-
import sys
from PyQt5.QtWidgets import QDialog, QLabel, QPushButton, QLineEdit, QListWidget, QGridLayout, QComboBox, QMessageBox, QApplication, QMenuBar, QAction, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSlot, QThread, QObject
from PyQt5.QtGui import QIcon, QPixmap, QImage
from movieSource.MovieHeaven import MovieHeaven


class ImageWindow(QMainWindow):
    def __init__(self, resources, title):
        super(ImageWindow, self).__init__()
        self.setWindowTitle(title)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        image = QImage(resources)
        pixmap = QPixmap(resources)
        image_label = QLabel(self)
        image_label.setPixmap(pixmap)
        image_label.resize(pixmap.width(), pixmap.height())
        layout.addWidget(image_label)


class LayoutDialog(QMainWindow):
    __slots__ = ['word', 'movie_name_label', 'movie_name_line_edit', 'movie_source_label', 'movie_source_combobox',
                 'search_push_button', 'tip_label', 'search_content_label', 'search_content_text_list']

    def __init__(self):
        super().__init__()
        self.left = 300
        self.top = 300
        self.width = 400
        self.height = 450

        self.work = WorkThread()
        self.init_widgets().init_layout().init_event()

    def init_widgets(self):
        self.setWindowTitle(self.tr("Search Movies"))
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.movie_name_label = QLabel(self.tr("电影名称:"))
        self.movie_name_line_edit = QLineEdit()

        self.movie_source_label = QLabel(self.tr("选择片源:"))
        self.movie_source_combobox = QComboBox()
        self.movie_source_combobox.addItem(self.tr('电影天堂'))

        self.search_push_button = QPushButton(self.tr("查询"))

        self.tip_label = QLabel(self.tr("未开始查询..."))
        self.search_content_label = QLabel(self.tr("查询内容:"))
        self.search_content_text_list = QListWidget()

        self.menu_bar = self.menuBar()

        return self

    def init_layout(self):
        top_layout = QGridLayout()
        top_layout.addWidget(self.movie_name_label, 0, 0)
        top_layout.addWidget(self.movie_name_line_edit, 0, 1)
        top_layout.addWidget(self.movie_source_label, 0, 2)
        top_layout.addWidget(self.movie_source_combobox, 0, 3)
        top_layout.addWidget(self.search_push_button, 0, 4)
        top_layout.addWidget(self.tip_label, 3, 1)
        top_layout.addWidget(self.search_content_label, 3, 0)
        top_layout.addWidget(self.search_content_text_list, 4, 0, 2, 5)

        main_frame = QWidget()
        self.setCentralWidget(main_frame)
        main_frame.setLayout(top_layout)

        self.reward_window = ImageWindow('resources/wechat_reward.jpg', '赞赏')
        self.watch_window = ImageWindow('resources/watch_wechat.jpg', '关注')

        return self

    def init_event(self):
        self.search_push_button.clicked.connect(self.search)
        self.search_content_text_list.itemClicked.connect(self.copy_text)

        reward_action = QAction('赞赏', self)
        reward_action.setIcon(QIcon('resources/reward.png'),)
        reward_action.triggered.connect(self.reward)

        watch_action = QAction('关注', self)
        watch_action.setIcon(QIcon('resources/watch.png'),)
        watch_action.triggered.connect(self.watch_wechat)

        reward_menu = self.menu_bar.addMenu('支持作者')
        reward_menu.addAction(reward_action)
        reward_menu.addAction(watch_action)

    def reward(self):
        self.reward_window.show()

    def watch_wechat(self):
        self.watch_window.show()

    def search(self):
        self.tip_label.setText(self.tr("正在查询请稍后..."))
        movie_name = self.movie_name_line_edit.text()
        if movie_name:
            self.work.render(movie_name, self.movie_source_combobox,
                             self.tip_label, self.search_content_text_list)
        else:
            self.critical("请输入电影名称!")

    def critical(self, message):
        """
        when the movieName is None,
        remind users
        """
        QMessageBox.critical(self, self.tr("致命错误"),
                             self.tr(message))

    def copy_text(self):
        copied_text = self.search_content_text_list.currentItem().text()
        QApplication.clipboard().clear()
        QApplication.clipboard().setText(copied_text)
        self.slot_information()

    def slot_information(self):
        QMessageBox.information(self, "Success!", self.tr("成功将内容复制到剪贴板上!"))


class WorkThread(QThread):     # 爬虫这里用了线程，之前没有想到
    def __init__(self):
        QThread.__init__(self)

    def render(self, movie_name, movie_source_combobox, tip_label, search_content_text_list):
        self.movies_list = []
        self.movie_source_combobox = movie_source_combobox
        self.movie_name = movie_name
        self.tip_label = tip_label
        self.search_content_text_list = search_content_text_list
        self.start()

    def get_select_movie_source(self, movie_name):
        """
        according to the value of the QComboBox,
        generate the right class of movie search
        """
        movies, url, params = None, None, {"typeid": "1"}
        select_source = self.movie_source_combobox.currentText()
        if select_source == self.tr('电影天堂'):
            movies = MovieHeaven()
            url = "http://s.dydytt.net/plus/s0.php"
            params["keyword"] = movie_name.encode('gb2312')
        return movies, url, params

    def run(self):
        search_movies, url, params = self.get_select_movie_source(
            self.movie_name)
        print(search_movies,url, params)
        try:
            self.movies_list = search_movies.get_display_content(url, params)
        except Exception as e:
            self.movies_list.append(self.tr("过于频繁的访问"))
        finally:
            self.search_content_text_list.clear()
            self.search_content_text_list.addItems(self.movies_list)
            self.tip_label.setText(self.tr("查询结束"))


app = QApplication(sys.argv)
dialog = LayoutDialog()
dialog.show()
app.exec_()
```

- 电影天堂爬取代码

```python
# -*- encoding:utf-8 -*-
import requests
import re
import urllib
from movieSource.fake_user_agent import useragent_random
from multiprocessing.dummy import Pool as ThreadPool
import sys


class MovieHeaven:
    __slots__ = ['__pool', '__all_page_details_url_list', '__search_url', '__search_domain', '__download_domain',
                 '__params']

    def __init__(self, parent=None):
        self.__pool = ThreadPool(8)
        self.__all_page_details_url_list = []
        self.__search_url = "http://s.dydytt.net/plus/s0.php"
        self.__search_domain = 'http://s.ygdy8.com'
        self.__download_domain = 'http://www.ygdy8.com'
        self.__params = {"typeid": "1",
                        "keyword": "leetao"}

    def __get_headers(self):
        return {"User-Agent": useragent_random()}

    def __search_movie_results(self, url=None, params=None):
        if url is None:
            url = self.__search_url

        temp_results = requests.get(
            url, params=params, headers=self.__get_headers())
        temp_results.encoding = 'gb2312'
        return temp_results.text

    def __get_movies_detail_page(self, searchResults):
        """
        get the detailPage's url of movies by using regx
        """
        pattern = re.compile(
            r"<td\s+width='\d+%'><b><a\s+href='(.*\.html)'\s*>")
        all_detai_pages = pattern.findall(searchResults)
        return all_detai_pages

    def __get_page_number_total(self, searchResults):
        """
        get the total number of pages
        """
        page_num_total_pattern = re.compile(
            r"<td\s+width='30'><a\s+href='.+PageNo=(\d+)'\s*>")
        page_num_total = page_num_total_pattern.findall(searchResults)
        if len(page_num_total) == 0:
            return -1
        else:
            return int(page_num_total[0])

    def __next_page_detail(self, search_results):
        """
        get the next page'url which lacks the pagenumber
        """
        next_page_pattern = re.compile(
            r"<td\s+width='30'><a href='(.*PageNo=)\d+'>")
        next_page_url = next_page_pattern.findall(search_results)
        return str(next_page_url[0])

    def __get_search_content_by_url(self, next_page_url, page_num_total):
        """
        get remain pages's url
        """
        for page_no in range(2, page_num_total + 1):
            if page_no is not None:
                url = self.__search_domain + next_page_url + str(page_no)
                res = self.__search_movie_results(url)
                return self.__get_movies_detail_page(res)

    def __get_movie_contents_url(self, url, params=None):
        """
        get the first page of searching results
        and  get the remain pages's results
        """
        first_page_results = self.__search_movie_results(url, params)
        first_page_resultsList = self.__get_movies_detail_page(
            first_page_results)

        # get the remain pages's results
        total_page_num = self.__get_page_number_total(first_page_results)
        if total_page_num > 0:
            next_page_url = self.__next_page_detail(first_page_results)
            remain_page_results_list = self.__get_search_content_by_url(
                next_page_url, total_page_num)
            self.__all_page_details_url_list.extend(remain_page_results_list)

        self.__all_page_details_url_list.extend(first_page_resultsList)
        return self.__all_page_details_url_list

    def __get_movie_down_url(self, down_page_url_list):
        results_list = []
        down_page_content_url_list = [
            (self.__download_domain + url) for url in down_page_url_list]
        for result_url_list in self.__pool.map(self.__get_down_page_content_url, self.__pool.map(self.__search_movie_results, down_page_content_url_list)):
            if len(result_url_list) > 0:
                results_list += result_url_list

        self.__pool.close()
        self.__pool.join()
        return results_list

    def __get_down_page_content_url(self, down_page_content):
        download_url_list = []
        ftp_down_pattern = re.compile(r'<td.+><a\s+href="(.+)"\s*>')
        ftp_url_list = ftp_down_pattern.findall(down_page_content)
        if len(ftp_url_list) > 0:
            download_url_list.append(ftp_url_list[0])

        magnet_down_pattern = re.compile(
            r'<a\s+href="(magnet:\?xt=.+)"><strong>')
        magnet_url_list = magnet_down_pattern.findall(down_page_content)
        if len(magnet_url_list) > 0:
            download_url_list.append(magnet_url_list[0].replace("amp;", ""))

        return download_url_list

    def get_display_content(self, url, params=None):
        url_list = self.__get_movie_contents_url(url, params)
        if len(url_list) == 0:
            return ['Not Found']
        else:
            all_download_url_list = self.__get_movie_down_url(url_list)
            movie_list = [
                url for url in all_download_url_list if url is not None and url[-3:] not in ['zip', 'rar', 'exe']]
            return movie_list

```

- face_agent.py

```python
import random

FAKE_USER_AGENT = [
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
    "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
    "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
    "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
    "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
    "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
    "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
    "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.11 TaoBrowser/2.0 Safari/536.11",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; LBBROWSER)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E; LBBROWSER)",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.84 Safari/535.11 LBBROWSER",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; QQBrowser/7.0.3698.400)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SV1; QQDownload 732; .NET4.0C; .NET4.0E; 360SE)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E)",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.89 Safari/537.1",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.89 Safari/537.1",
    "Mozilla/5.0 (iPad; U; CPU OS 4_2_1 like Mac OS X; zh-cn) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8C148 Safari/6533.18.5",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:2.0b13pre) Gecko/20110307 Firefox/4.0b13pre",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:16.0) Gecko/20100101 Firefox/16.0",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
    "Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/53"
]
def useragent_random():
    return random.choice(FAKE_USER_AGENT)
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/crawl_heavens/  

