# SpringException


> Spring MVC 通过 **HandlerExceptionResolver** 处理程序的异常，包括 Handler映射、数据绑定以及目标方法执行 时发生的异常。SpringMVC 提供的 HandlerExceptionResolver 的实现类

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210628135210911.png)

### 1. 全局异常捕获处理

- **@ControllerAdvice**注解： 开启全局异常捕获
- **@ExceptionHandle**r注解： 说明捕获哪些异常

```java
@ControllerAdvice
public class MyExceptionHandler {
    @ExceptionHandler(value =Exception.class)
	public String exceptionHandler(Exception e){
		System.out.println("发生了一个异常"+e);
       	return e.getMessage();
    }
}
```

### 2. 统一结果返回&异常

```java
public class Result<T> {
    //是否成功
    private Boolean success;
    //状态码
    private Integer code;
    //提示信息
    private String msg;
    //数据
    private T data;
    public Result() {

    }
    //自定义返回结果的构造方法
    public Result(Boolean success,Integer code, String msg,T data) {
        this.success = success;
        this.code = code;
        this.msg = msg;
        this.data = data;
    }
    //自定义异常返回的结果
    public static Result defineError(DefinitionException de){
        Result result = new Result();
        result.setSuccess(false);
        result.setCode(de.getErrorCode());
        result.setMsg(de.getErrorMsg());
        result.setData(null);
        return result;
    }
    //其他异常处理方法返回的结果
    public static Result otherError(ErrorEnum errorEnum){
        Result result = new Result();
        result.setMsg(errorEnum.getErrorMsg());
        result.setCode(errorEnum.getErrorCode());
        result.setSuccess(false);
        result.setData(null);
        return result;
    }
}
```

```java
public enum ErrorEnum {
	// 数据操作错误定义
	SUCCESS(200, "nice"),
	NO_PERMISSION(403,"你没得权限"),
	NO_AUTH(401,"你能不能先登录一下"),
	NOT_FOUND(404, "未找到该资源!"),
	INTERNAL_SERVER_ERROR(500, "服务器跑路了"),
	;

	/** 错误码 */
	private Integer errorCode;

	/** 错误信息 */
	private String errorMsg;

	ErrorEnum(Integer errorCode, String errorMsg) {
		this.errorCode = errorCode;
		this.errorMsg = errorMsg;
	}

    public Integer getErrorCode() {
        return errorCode;
    }

    public String getErrorMsg() {
        return errorMsg;
    }
}
```

- 自定义异常处理类,spring 对于 RuntimeException 异常才会进行事务回滚。

```java
public class DefinitionException extends RuntimeException{

    protected Integer errorCode;
    protected String errorMsg;

    public DefinitionException(){

    }
    public DefinitionException(Integer errorCode, String errorMsg) {
        this.errorCode = errorCode;
        this.errorMsg = errorMsg;
    }

    public Integer getErrorCode() {
        return errorCode;
    }

    public void setErrorCode(Integer errorCode) {
        this.errorCode = errorCode;
    }

    public String getErrorMsg() {
        return errorMsg;
    }

    public void setErrorMsg(String errorMsg) {
        this.errorMsg = errorMsg;
    }
}
```

- 自定义一个全局异常处理类

```java
@ControllerAdvice
public class GlobalExceptionHandler {

    /**
     * 处理自定义异常
     *
     */
    @ExceptionHandler(value = DefinitionException.class)
    @ResponseBody
    public Result bizExceptionHandler(DefinitionException e) {
        return Result.defineError(e);
    }

    /**
     * 处理其他异常
     *
     */
    @ExceptionHandler(value = Exception.class)
    @ResponseBody
    public Result exceptionHandler( Exception e) {
        return Result.otherError(ErrorEnum.INTERNAL_SERVER_ERROR);
    }
}
```

- controller

```java
@RestController
@RequestMapping("/result")
public class ResultController {
    @GetMapping("/getStudent")
    public Result getStudent(){
        Student student = new Student();
        student.setAge(21);
        student.setId(111);
        student.setName("学习笔记");
        Result result = new Result();
        result.setCode(200);
        result.setSuccess(true);
        result.setData(student);
        result.setMsg("学生列表信息");
        return result;
    }
    @RequestMapping("/getDeException")
    public Result DeException(){
        throw new DefinitionException(400,"我出错了");
    }
    @RequestMapping("/getException")
    public Result Exception(){
        Result result = new Result();
        int a=1/0;
        return result;
    }
}
```

### 3. 404异常

```java
@Configuration
public class ResourceConfig implements WebMvcConfigurer {
    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        //可以访问localhost:8095/static/images/image.jpg
        registry.addResourceHandler("/static/**").addResourceLocations("classpath:/static/");
    }
}
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/springexception/  

