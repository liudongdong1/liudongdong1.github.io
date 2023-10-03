# Typescript


>  TypeScript is a typed `superset of JavaScript` that compiles to plain JavaScript. TypeScript is `pure object oriented with classes, interfaces and statically typed like C# or Java`. TypeScript supports other JS libraries; and portable across browsers, devices, and operating systems, and don't need a dedicated VM or specific runtime environment to execute;

### 1. Environment

> Node.js is an open source, cross-platform runtime environment for server-side JavaScript. Node.js is required to run JavaScript without a browser support. It uses Google V8 JavaScript engine to execute code. 

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201130202850477.png)

```shell
npm install -g typescript
tsc app.ts
node app.js
```

### 2. code

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201130203446809.png)

```Typescript
// declare a variable
var score1:number = 50;  

//type assertion
var str = '1' 
var str2:number = <number> <any> str   //str is now of type number 
console.log(typeof(str2))

// data scope
var global_num = 12          //global variable 
class Numbers { 
   num_val = 13;             //class variable 
   static sval = 10;         //static field 
   
   storeNum():void { 
      var local_num = 14;    //local variable 
   } 
} 
console.log("Global num: "+global_num)  
console.log(Numbers.sval)   //static variable  
var obj = new Numbers(); 
console.log("Global num: "+obj.num_val) 

//Function
//function function_name (param1[:type], param2[:type], param3[:type])
//function function_name(param1[:type],param2[:type] = default_value) 

//Anonymous function
//var res = function( [arguments] ) { ... }

//lambda function
//( [param1, parma2,…param n] )=>statement;


class Greeting { 
   greet():void { 
      console.log("Hello World!!!") 
   } 
} 
var obj = new Greeting(); 
obj.greet();
```

### 3. Interfaces&Class&Object

> interfaces define properties, methods, and events, which are the members of the interface. Interfaces contain only the declaration of the members. It is the responsibility of the deriving class to define the members. It often helps in providing a standard structure that the deriving classes would follow.

```TypeScript
//interface interface_name { 
//}

interface IPerson { 
   firstName:string, 
   lastName:string, 
   sayHi: ()=>string 
} 

var customer:IPerson = { 
   firstName:"Tom",
   lastName:"Hanks", 
   sayHi: ():string =>{return "Hi there"} 
} 
```

```typescript
interface IParent1 { 
   v1:number 
} 

interface IParent2 { 
   v2:number 
} 

interface Child extends IParent1, IParent2 { } 
var Iobj:Child = { v1:12, v2:23} 
console.log("value 1: "+this.v1+" value 2: "+this.v2)
```

```typescript
class Shape { 
   Area:number 
   constructor(a:number) { 
      this.Area = a 
   } 
} 
class Circle extends Shape { 
   disp():void { 
      console.log("Area of the circle:  "+this.Area) 
   } 
}
var obj = new Circle(223); 
obj.disp()
```

```typescript
class StaticMem {  
   static num:number; 
   static disp():void { 
      console.log("The value of num is"+ StaticMem.num) 
   } 
}
StaticMem.num = 12     // initialize the static variable 
StaticMem.disp()      // invoke the static methodtypescript
```

- Object

```typescript
var object_name = { 
   key1: “value1”, //scalar value 
   key2: “value”,  
   key3: function() {
      //functions 
   }, 
   key4:[“content1”, “content2”] //collection  
};
```

### 4. Namespace

```typescript
FileName :IShape.ts 
---------- 
namespace Drawing { 
   export interface IShape { 
      draw(); 
   }
}  

FileName :Circle.ts 
---------- 
/// <reference path = "IShape.ts" /> 
namespace Drawing { 
   export class Circle implements IShape { 
      public draw() { 
         console.log("Circle is drawn"); 
      }  
      
      FileName :Triangle.ts 
      ---------- 
      /// <reference path = "IShape.ts" /> 
      namespace Drawing { 
         export class Triangle implements IShape { 
            public draw() { 
               console.log("Triangle is drawn"); 
            } 
         } 
         
         FileName : TestShape.ts 
         /// <reference path = "IShape.ts" />   
         /// <reference path = "Circle.ts" /> 
         /// <reference path = "Triangle.ts" />  
         function drawAllShapes(shape:Drawing.IShape) { 
            shape.draw(); 
         } 
         drawAllShapes(new Drawing.Circle());
         drawAllShapes(new Drawing.Triangle());
      }
   }
}    
```

### 5. Module

#### 5.1. Internal Module

```typescript
// old syntax
module TutorialPoint { 
   export function add(x, y) {  
      console.log(x+y); 
   } 
}

//new syntax
namespace TutorialPoint { 
   export function add(x, y) { console.log(x + y);} 
}
```

#### 5.2.  External Module

> Traditionally dependency management between JavaScript files was done using `browser script tags (<script></script>)`. But that’s `not extendable`, as its very linear while loading modules. That means instead of loading files one after other there is no asynchronous option to load modules. When you are programming js for the server for example NodeJs you don’t even have script tags.
>
> - Module Loader: `RequireJS`, an implementation of asynchronous module definition specification, and load all module separately, even when they dependent on each other;
> - Defining External Module: `each file is considered as a module`, 

```typescript
//Syntax
//FileName : SomeInterface.ts 
export interface SomeInterface { 
   //code declarations 
}
import someInterfaceRef = require(“./SomeInterface”);
```

```typescript
// IShape.ts 
export interface IShape { 
   draw(); 
}

// Circle.ts 
import shape = require("./IShape"); 
export class Circle implements shape.IShape { 
   public draw() { 
      console.log("Cirlce is drawn (external module)"); 
   } 
} 

// Triangle.ts 
import shape = require("./IShape"); 
export class Triangle implements shape.IShape { 
   public draw() { 
      console.log("Triangle is drawn (external module)"); 
   } 
}
   
// TestShape.ts 
import shape = require("./IShape"); 
import circle = require("./Circle"); 
import triangle = require("./Triangle");  

function drawAllShapes(shapeToDraw: shape.IShape) {
   shapeToDraw.draw(); 
} 

drawAllShapes(new circle.Circle()); 
drawAllShapes(new triangle.Triangle()); 
```

### 6. Ambients

>  telling the TypeScript compiler that the actual source code exists elsewhere. When you are consuming a bunch of third party **js** libraries like jquery/angularjs/nodejs you can’t rewrite it in TypeScript. Ensuring typesafety and intellisense while using these libraries will be challenging for a TypeScript programmer. `Ambient declarations help to seamlessly integrate other js libraries into TypeScript.`

```typescript
// declare  in Sample.d.ts
declare module Module_Name {
}
/// <reference path = " Sample.d.ts" />
```

```javascript
FileName: CalcThirdPartyJsLib.js 
var TutorialPoint;  
(function (TutorialPoint) {  
   var Calc = (function () { 
      function Calc() { 
      } 
      Calc.prototype.doSum = function (limit) {
         var sum = 0; 
         
         for (var i = 0; i <= limit; i++) { 
            Calc.prototype.doSum = function (limit) {
               var sum = 0; 
               
               for (var i = 0; i <= limit; i++) { 
                  sum = sum + i; 
                  return sum; 
                  return Calc; 
                  TutorialPoint.Calc = Calc; 
               })(TutorialPoint || (TutorialPoint = {})); 
               var test = new TutorialPoint.Calc();
            }
         }
      }
   }
}   
 
FileName: Calc.d.ts 
declare module TutorialPoint { 
   export class Calc { 
      doSum(limit:number) : number; 
   }
}

FileName : CalcTest.ts  
/// <reference path = "Calc.d.ts" /> 
var obj = new TutorialPoint.Calc(); 
obj.doSum("Hello"); // compiler error 
console.log(obj.doSum(10));
```

### 7. Learning Tutorial

- https://www.tutorialspoint.com/typescript/typescript_quick_guide.htm
- https://www.typescriptlang.org/docs

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/typescript/  

