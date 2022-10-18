---
title: "【Go官方文档】Developing a RESTful API with Go and Gin"
subtitle: ""
date: 2022-10-17T23:10:00+08:00
description: ""
keywords: ""
tags: [”Go-document"]
categories: ["Programming"]
---

{{< admonition quote >}}
本文内容来自：[Tutorial: Developing a RESTful API with Go and Gin](https://golang.google.cn/doc/tutorial/web-service-gin)
{{< /admonition >}}

**环境配置：**
- 系统：*Windows11*
- 编辑器：*vscode*

#### 🍐 创建项目
1. 创建项目文件夹`web-service-gin`并用vscode打开该项目
2. 初始化go mod
```shell
go mod init example/web-service-gin
```
3. 创建文件`main.go`
```go
package main

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

type album struct {
	ID     string  `json:"id"`
	Title  string  `json:"title"`
	Artist string  `json:"artist"`
	Price  float64 `json:"price"`
}

var albums = []album{
	{ID: "1", Title: "Blue Train", Artist: "John Coltrane", Price: 56.99},
	{ID: "2", Title: "Jeru", Artist: "Gerry Mulligan", Price: 17.99},
	{ID: "3", Title: "Sarah Vaughan and Clifford Brown", Artist: "Sarah Vaughan", Price: 39.99},
}

func getAlbums(c *gin.Context) {
	c.IndentedJSON(http.StatusOK,albums)
}

func main() {
	router := gin.Default()
	router.GET("albums",getAlbums)

	router.Run("localhost:8080")
}
```
#### 🥝 运行项目
1. 执行命令：
```shell
go run .
```
2. 打开浏览器，地址栏输入：`http://localhost:8080/albums`,回车确定
3. 浏览器显示内容：
```json
[
    {
        "id": "1",
        "title": "Blue Train",
        "artist": "John Coltrane",
        "price": 56.99
    },
    {
        "id": "2",
        "title": "Jeru",
        "artist": "Gerry Mulligan",
        "price": 17.99
    },
    {
        "id": "3",
        "title": "Sarah Vaughan and Clifford Brown",
        "artist": "Sarah Vaughan",
        "price": 39.99
    }
]
```