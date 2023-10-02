# CommonMath


> The [Apache Commons Math](https://mvnrepository.com/artifact/org.apache.commons/commons-math3/3.6.1) project is a library of lightweight, self-contained mathematics and statistics components addressing the most common practical problems not immediately available in the Java programming language or commons-lang.  [官方案例文档](https://commons.apache.org/proper/commons-math/userguide/index.html)
>
> - [org.apache.commons.math4.stat](https://commons.apache.org/proper/commons-math/userguide/stat.html) - statistics, statistical tests
> - [org.apache.commons.math4.analysis](https://commons.apache.org/proper/commons-math/userguide/analysis.html) - rootfinding, integration, interpolation, polynomials
> - [org.apache.commons.math4.random](https://commons.apache.org/proper/commons-math/userguide/random.html) - random numbers, strings and data generation
> - [org.apache.commons.math4.special](https://commons.apache.org/proper/commons-math/userguide/special.html) - special functions (Gamma, Beta)
> - [org.apache.commons.math4.linear](https://commons.apache.org/proper/commons-math/userguide/linear.html) - matrices, solving linear systems
> - [org.apache.commons.math4.util](https://commons.apache.org/proper/commons-math/userguide/utilities.html) - common math/stat functions extending java.lang.Math
> - [org.apache.commons.math4.complex](https://commons.apache.org/proper/commons-math/userguide/complex.html) - complex numbers
> - [org.apache.commons.math4.distribution](https://commons.apache.org/proper/commons-math/userguide/distribution.html) - probability distributions
> - [org.apache.commons.math4.fraction](https://commons.apache.org/proper/commons-math/userguide/fraction.html) - rational numbers
> - [org.apache.commons.math4.transform](https://commons.apache.org/proper/commons-math/userguide/transform.html) - transform methods (Fast Fourier)
> - [org.apache.commons.math4.geometry](https://commons.apache.org/proper/commons-math/userguide/geometry.html) - geometry (Euclidean spaces and Binary Space Partitioning)
> - [org.apache.commons.math4.optim](https://commons.apache.org/proper/commons-math/userguide/optimization.html) - function maximization or minimization
> - [org.apache.commons.math4.ode](https://commons.apache.org/proper/commons-math/userguide/ode.html) - Ordinary Differential Equations integration
> - [org.apache.commons.math4.genetics](https://commons.apache.org/proper/commons-math/userguide/genetics.html) - Genetic Algorithms
> - [org.apache.commons.math4.fitting](https://commons.apache.org/proper/commons-math/userguide/fitting.html) - Curve Fitting
> - [org.apache.commons.math4.ml](https://commons.apache.org/proper/commons-math/userguide/ml.html) - Machine Learning

>- Computing `means, variances` and other `summary statistics` for a list of numbers
>- Fitting a line to a set of data points using `linear regression`
>- Fitting a` curve` to a set of data points
>- Finding a` smooth curve` that passes through a collection of points (`interpolation`)
>- Fitting a `parametric model` to a set of measurements using least-squares methods
>- Solving equations involving real-valued functions (i.e. root-finding)
>- Solving systems of linear equations
>- Solving Ordinary Differential Equations
>- Minimizing multi-dimensional functions
>- `Generating random numbers` with more restrictions (e.g distribution, range) than what is possible using the JDK
>- Generating random samples and/or datasets that are "like" the data in an input file
>- Performing statistical significance tests
>- Miscellaneous mathematical functions such as factorials, binomial coefficients and "special functions" (e.g. gamma, beta functions)

### 1. 多项式拟合

```java
double[] x = new double[]{1, 2, 3, 4, 5};
double[] y = new double[]{19,33,53,79,111};
		  
WeightedObservedPoints points = new WeightedObservedPoints(); 	 
for(int i = 0; i < x.length; i++) { //把数据点加入观察的序列
	points.add(x[i], y[i]);
}		
PolynomialCurveFitter fitter = PolynomialCurveFitter.create(2);  //指定多项式阶数 
double[] result = fitter.fit(points.toList());  // 曲线拟合，结果保存于数组[c,b,a]  a*x*x+b*x+c=y	 
for(int i = 0; i < result.length; i++) {
	System.out.println(result[i]);
}
```

```java
/** 
* 计算指定对象的运行时间开销。 
* 
* @param curveFitting 指定被测对象。 
* @return 返回sub.run的时间开销，单位为s。 
* @throws Exception 
*/  
public double calcTimeCost(CurveFitting curveFitting) throws Exception {  
    List<Object> params = curveFitting.getParams();  
    long startTime = System.nanoTime();  
    Object result = curveFitting.run(params);  
    long stopTime = System.nanoTime();  
    curveFitting.printResult(result);  
    System.out.println("start: " + startTime + " / stop: " + stopTime);  
    return 1.0e-9 * (stopTime - startTime);  
}  
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/commonmath/  

