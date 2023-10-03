# CSS Learning Record


### 1. Syntax

- **Selector** − A selector is an HTML tag at which a style will be applied. This could be any tag like  or  etc.
- **Property** − A property is a type of attribute of HTML tag. Put simply, all the HTML attributes are converted into CSS properties. They could be *color*, *border* etc.
- **Value** − Values are assigned to properties. For example, *color* property can have value either *red* or *#F1F1F1* etc.

#### 1.1 Selector

```css
# 1. Type selector
h1 {
   color: #36CFFF; 
}

#2. Universal Selector
* { 
   color: #000000; 
}

#3. Descendant Selector
ul em{
    color: #0000;
}

#4. Class Selector
.black {
   color: #000000; 
}

#5. ID selector
#black {
   color: #000000; 
}
```

#### 1.2  Inclusion

```css
<style type = "text/css" media = "all">
         body {
            background-color: linen;
         }
         h1 {
            color: maroon;
            margin-left: 40px;
         }
</style>

# 1. Inline
<h1 style = "color:#36C;"> 
# 2. external
 <link type = "text/css" href = "..." media = "..." />
```

#### 1.3 MeasureMent Units

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200505161028134.png)

### 2. Attribute

#### 2.1 Background

- The **background-color** property is used to set the background color of an element.
- The **background-image** property is used to set the background image of an element.
- The **background-repeat** property is used to control the repetition of an image in the background.
- The **background-position** property is used to control the position of an image in the background.
- The **background-attachment** property is used to control the scrolling of an image in the background.

#### 2.2 font

- The **font-family** property is used to change the face of a font.
- The **font-style** property is used to make a font italic or oblique.
- The **font-variant** property is used to create a small-caps effect.
- The **font-weight** property is used to increase or decrease how bold or light a font appears.
- The **font-size** property is used to increase or decrease the size of a font.

#### 2.4 Text

- The **color** property is used to set the color of a text.
- The **direction** property is used to set the text direction.
- The **letter-spacing** property is used to add or subtract space between the letters that make up a word.
- The **word-spacing** property is used to add or subtract space between the words of a sentence.
- The **text-indent** property is used to indent the text of a paragraph.
- The **text-align** property is used to align the text of a document.
- The **text-decoration** property is used to underline, overline, and strikethrough text.
- The **text-transform** property is used to capitalize text or convert text to uppercase or lowercase letters.
- The **white-space** property is used to control the flow and formatting of text.
- The **text-shadow** property is used to set the text shadow around a text.

#### 2.5 Image

- The **border** property is used to set the width of an image border.
- The **height** property is used to set the height of an image.
- The **width** property is used to set the width of an image.
- The **-moz-opacity** property is used to set the opacity of an image.

#### 2.6 Link

- The **:link** signifies unvisited hyperlinks.
- The **:visited** signifies visited hyperlinks.
- The **:hover** signifies an element that currently has the user's mouse pointer hovering over it.
- The **:active** signifies an element on which the user is currently clicking.

```css
<style type = "text/css">
   a:link {color: #000000}
   a:visited {color: #006600}
   a:hover {color: #FFCC00}
   a:active {color: #FF00CC}
</style>	
```

#### 2.7 Table

> - The **border-collapse** specifies whether the browser should control the appearance of the adjacent borders that touch each other or whether each cell should maintain its style.
> - The **border-spacing** specifies `the width that should appear between table cells`.
> - The **caption-side** captions are presented in the  caption element. By default, these are rendered above the table in the document. You use the  caption-side  property to control the placement of the table caption.
> - The **empty-cells** specifies whether the border should be shown if a cell is empty.
> - The **table-layout** allows browsers to speed up layout of a table by using the first width properties it comes across for the rest of a column rather than having to load the whole table before rendering it.

```css
<html>
   <head>
      <style type = "text/css">
         table.one {border-collapse:collapse;}
         table.two {border-collapse:separate;}
         
         td.a {
            border-style:dotted; 
            border-width:3px; 
            border-color:#000000; 
            padding: 10px;
         }
         td.b {
            border-style:solid; 
            border-width:3px; 
            border-color:#333333; 
            padding:10px;
         }
      </style>
   </head>

   <body>
      <table class = "one">
         <caption>Collapse Border Example</caption>
         <tr><td class = "a"> Cell A Collapse Example</td></tr>
         <tr><td class = "b"> Cell B Collapse Example</td></tr>
      </table>
      <br />
      
      <table class = "two">
         <caption>Separate Border Example</caption>
         <tr><td class = "a"> Cell A Separate Example</td></tr>
         <tr><td class = "b"> Cell B Separate Example</td></tr>
      </table>
   </body>
</html>
```

#### 2.8 Border

> - The **border-color** specifies the color of a border.
> - The **border-style** specifies whether a border should be solid, dashed line, double line, or one of the other possible values.
> - The **border-width** specifies the width of a border.
> - **border-bottom-color** changes the color of bottom border.
> - **border-top-color** changes the color of top border.
> - **border-left-color** changes the color of left border.
> - **border-right-color** changes the color of right border.

#### 2.9 Margin

> - The **margin-bottom** specifies the bottom margin of an element.
> - The **margin-top** specifies the top margin of an element.
> - The **margin-left** specifies the left margin of an element.
> - The **margin-right** specifies the right margin of an element.

#### 2.10 Lists

> - The **list-style-type** allows you to` control the shape or appearance of the marker.`
> - The **list-style-position** specifies whether a long point that wraps to a second line should align with the first line or start underneath the start of the marker.
> - The **list-style-image** specifies an image for the marker rather than a bullet point or number.
> - The **list-style** serves as shorthand for the preceding properties.
> - The **marker-offset** specifies the distance between a marker and the text in the list.

#### 2.11  Paddings

> -  specify how much space should appear between the content of an element and its border 
> -  The **padding-bottom** specifies the bottom padding of an element.
> -  The **padding-top** specifies the top padding of an element.
> -  The **padding-left** specifies the left padding of an element.
> -  The **padding-right** specifies the right padding of an element.

#### 2.12 Outline

> - The **outline-width** property is used to set the width of the outline.
> - The **outline-style** property is used to set the line style for the outline.
> - The **outline-color** property is used to set the color of the outline.
> - The **outline** property is used to set all the above three properties in a single statement.

#### 2.13 Dimension

> - The **height** property is used to set the height of a box.
> - The **width** property is used to set the width of a box.
> - The **line-height** property is used to set the height of a line of text.
> - The **max-height** property is used to set a maximum height that a box can be.
> - The **min-height** property is used to set the minimum height that a box can be.
> - The **max-width** property is used to set the maximum width that a box can be.
> - The **min-width** property is used to set the minimum width that a box can be.

#### 2.14 Scrollbars

> - **visible**：Allows the content to overflow the borders of its containing element.
> - **hidden**：The content of the nested element is simply cut off at the border of the containing element and no scrollbars is visible.
> - **scroll：**The size of the containing element does not change, but the scrollbars are added to allow the user to scroll to see the content.
> - **auto：**The purpose is the same as scroll, but the scrollbar will be shown only if the content does overflow.

#### 2.16 visibility

- `visible， hidden， collapse`

#### 2.17 position

```css
<div style = "position:fixed; left:80px; top:20px; background-color:yellow;">
<div style = "position:absolute; left:80px; top:20px; background-color:yellow;">
<div style = "position:relative; left:80px; top:2px; background-color:yellow;">
```

#### 2.18 layer

- `z-index : 值越大越在上面`

### 3. Effect

#### 3.1  Rounded Corners

```css
#rcorners7 {
   border-radius: 60px/15px;
   background: #FF0000;
   padding: 20px; 
   width: 200px;
   height: 150px; 
}
```

#### 3.2 Border Image

- border-img-source: used to set image path
- border-img-slice: used to slice boarder image
- border-img-width: 
- border-img-repeat:  set boarder image as rounded, repeated, and stretched.

```css
<style>
         #borderimg1 { 
            border: 10px solid transparent;
            padding: 15px;
            border-image-source: url(/css/images/border.png);
            border-image-repeat: round;
            border-image-slice: 30;
            border-image-width: 10px;
         }
</style>
```



### Resource

- css 在线查看： https://www.w3school.com.cn/css/index.asp


---

> 作者: [LiuDongdong](https://liudongdong1.github.io/)  
> URL: https://liudongdong1.github.io/css-learning-record/  

