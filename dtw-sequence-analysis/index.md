# DTW&Sequence Analysis


**level**: SIGKDD  ACM
**author**:Thanawin Rakthanmanon
**date**:  August 12–16, 2012
**keyword**:

- Sequence data matching

------

## Paper: DTW

<div align=center>
<br/>
<b>Searching and Mining Trillions of Time Series Subsequences under Dynamic Time Warping  </b>
</div>


#### Research Objective

- **Application Area**:  time series motif discovery [25] [26], anomaly detection [35] [31], time series summarization, shapelet extraction [39], clustering, and classification [6], gestures/brainwaves/musical patterns/anomalous heartbeats in real-time
- **Purpose**:  <font color=red>fast sequential search  instead of approximately search</font>

#### Proble Statement

> Time Series Subsequences must be Normalized ，or tiny changes we made are completely dwarfed by changes we might expect to see in a real world deployment. ,but  it is not sufficient to normalize the entire dataset.  

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200526152414354.png)

> Arbitrary Query Lengths cannot be Indexed.

#### Methods

- **Problem Formulation**:

  - **Definition1:**  Time serias to search :  $T=t_1,t_2,...,t_m$

  - **Definition 2:**  subsequence to query:  $Q = T_{i,k}=t_i,t_{i+1},...,t_{i+k-1},  i\epsilon [1,m-k+1]$

  - **Definition 3:**  the Euclidean distance(ED) between Q and C, where |Q|=|C|, the distance is :
    $$
    ED(Q,C)=\sqrt{\sum_{i=1}^n(q_i-c_i)^2}
    $$

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200526153507470.png)

- **ED&&DTW**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200228095803776.png)

**【Opinion 1】 Using the Squared Distance** 
$$
ED(Q,C)=\sqrt{\sum_{i=1}^n(q_i-c_i)^2} --->    ED(Q,C)=\sum_{i=1}^n(q_i-c_i)^2
$$


**【Opinion 2】 Using Lower Bounding** 

- **a）LB_Kim**

  通过计算两个序列的1,2,3,4四个点对应的ED距离，来计算两个序列的相似性，其中1,2点为首尾点，3,4点表示函数的最小点和最大点。其时间复杂度为O(n)。还有些改进版的，再在这四个点的基础上多取一些点来进行计算。公式：$LB_{Kim}(Q,C)=Max_{(i=1,2,3,4)}d(f_i^Q,f_i^C)$ , 经过Z-normalization 后，影响不大

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200526154124354.png)

  **b）LB_Yi**

  在LB_Kim的基础上做了改进，通过定义被比较序列C的最大和最小值的范围，来进行相似性的比较，公式如下：

$$
LBY_I(Q,C)=\sum_{q_i>max(C)}d(q_i,max(C))+\sum_{q_i<min(C)}d(q_i,min(C))
$$

  如下图所示：

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200526154326770.png)

  **c) LB_Keogh**

  LB_Keogh 下界函数，相比于LB_Kim以及LB_Yi具有更好的效果。Keogh使用了上下包络线，该下界距离更为紧凑, 不容易产生漏报。U 和 L 指的是上下包络函数。
$$
  LBFKeogh(Q,C)=\sum_{i=1}^n\begin {cases} (q_i-u_i)^2,q_i>u_i\\(q_i-l_i)^2,q_i<l_i\\0 \end{cases}
$$
  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200526154828729.png)

**【Opinion 3】 Using Early Abandoning of ED and LB_Keogh  **

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201026101815567.png)

**【Opinion 3】 Using Early Abandoning of DTW**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200526155211203.png)

对于右图，从左边开始DTW匹配，右边使用之前计算地LB_keogh 代码如下：

```c#
  /// Calculate Dynamic Time Wrapping distance
        /// A,B: data and query, respectively
        /// cb : cummulative bound used for early abandoning
        /// r  : size of Sakoe-Chiba warpping band
        private static double dtw(double[] A, double[] B, double[] cb, int m, int r,
                                  double bsf = double.PositiveInfinity)
        {

            double[] cost;
            double[] cost_prev;
            double[] cost_tmp;
            int i, j, k;
            double x, y, z, min_cost;

            /// Instead of using matrix of size O(m^2) or O(mr), we will reuse two array of size O(r).
            cost = new double[2 * r + 1]; //(double*)malloc(sizeof(double)*(2*r+1));
            for (k = 0; k < 2 * r + 1; k++) cost[k] = double.PositiveInfinity;

            cost_prev = new double[2 * r + 1]; //(double*)malloc(sizeof(double)*(2*r+1));
            for (k = 0; k < 2 * r + 1; k++) cost_prev[k] = double.PositiveInfinity;

            for (i = 0; i < m; i++)
            {
                k = max(0, r - i);
                min_cost = double.PositiveInfinity;

                for (j = max(0, i - r); j <= min(m - 1, i + r); j++, k++)
                {
                    // Initialize all row and column
                    if ((i == 0) && (j == 0))
                    {
                        cost[k] = dist(A[0], B[0]);
                        min_cost = cost[k];
                        continue;
                    }

                    if ((j - 1 < 0) || (k - 1 < 0)) y = double.PositiveInfinity;
                    else y = cost[k - 1];
                    if ((i - 1 < 0) || (k + 1 > 2 * r)) x = double.PositiveInfinity;
                    else x = cost_prev[k + 1];
                    if ((i - 1 < 0) || (j - 1 < 0)) z = double.PositiveInfinity;
                    else z = cost_prev[k];

                    // Classic DTW calculation
                    cost[k] = min(min(x, y), z) + dist(A[i], B[j]);

                    // Find minimum cost in row for early abandoning (possibly to use column instead of row).
                    if (cost[k] < min_cost)
                    {
                        min_cost = cost[k];
                    }
                }

                // We can abandon early if the current cummulative distace with lower bound together are larger than bsf
                if (i + r < m - 1 && min_cost + cb[i + r + 1] >= bsf)
                {
                    return min_cost + cb[i + r + 1];
                }

                // Move current array to previous array.
                cost_tmp = cost;
                cost = cost_prev;
                cost_prev = cost_tmp;
            }
            k--;
            // the DTW distance is in the last cell in the matrix of size O(m^2) or at the middle of our array.
            double final_dtw = cost_prev[k];

            return final_dtw;
        }
```

**【Opinion 4】 The UCR Suite**   

- **Early Abandoning Z-Normalization** 

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200526155315097.png)

-  **Reordering Early Abandoning**

> in the picture below, different orderings produce different speedups, an the author use an universal optimal ordering that to sort the index based on the absolute values of the Z-normalized Q, based on the intuition that the value at $Q_I$ will be compared to many $C_i$ during a search;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200228100422959.png)

- **Reversing the Query/Data Role in LB_Keogh** 

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200228100547809.png)

对应代码：

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200526202120154.png)



- **Cascading Lower Bounds**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200228100626971.png)

#### Experiment

- Random works of length 20 million with increasing long query

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200526155630301.png)

- **EEG Query:**

  ![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20201026105256644.png)

- **Supporting Very Long queries: DNA**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200526155729346.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200526155755980.png)

- Application:  Online Time serial motifs,  classification of historical musical scores, classification of ancient coins, clustering of star light curves

> We believe that oversampled data can be searched more quickly by exploiting a provisional search in a downsampled version of the data that can quickly provide a low best-so-far, which, when projected back into the original space can be used to “prime” the search by setting a low best-so-far at the beginning of the search, thus allowing the early abandoning techniques to be more efficient. 
>
> We simply caution the reader that oversampled (i.e., “smooth”) data may allow more speedup than a direct application of the UCR suite may initially suggest.

#### Examples

- **Time Series Shapelets** have garnered significant interest since their introduction in 2009 [39]. We obtained the original code and tested it on the Face (four) dataset, finding it took 18.9 minutes to finish. After replacing the similarity search routine with UCR suite, it took 12.5 minutes to finish. 
- **Online Time Series Motifs** generalize the idea of mining repeated patterns in a batch time series to the streaming case [25]. We obtained the original code and tested it on the EEG dataset used in the original paper. The fastest running time for the code assuming linear space is 436 seconds. After replacing the distance function with UCR suite, it took just 156 seconds.
-  **Classification of Historical Musical Scores** [10]. This dataset has 4,027 images of musical notes converted to time series. We used the UCR suite to compute the rotation-invariant DTW leaveone-out classification. It took 720.6 minutes. SOTA takes 142.4 hours. Thus, we have a speedup factor of 11.8.
-  **Classification of Ancient Coins** [15]. 2,400 irregularly shaped coins are converted to time series of length 256, and rotationinvariant DTW is used to search the database, taking 12.8 seconds per query. Using the UCR suite, this takes 0.8 seconds per query. 
- **Clustering of Star Light Curves** is an important problem in astronomy [20], as it can be a preprocessing step in outlier detection [31]. We consider a dataset with 1,000 (purportedly) phase-aligned light curves of length 1,024, whose class has been determined by an expert [31]. Doing spectral clustering on this data with DTW (R = 5%) takes about 23 minutes for all algorithms, and averaged over 100 runs we find the Rand-Index is 0.62. While this time may seem slow, recall that we must do 499,500 DTW calculations with relatively long sequences. As we do not trust the original claim of phase alignment, we further do rotation-invariant DTW that dramatically increases Rand-Index to 0.76. Using SOTA, this takes 16.57 days, but if we use the UCR suite, this time falls by an order of magnitude, to just 1.47 days on a single core.

#### Notes

- Code available:     http://www.cs.ucr.edu/~eamonn/UCRsuite.html

#### Code Explain:

- 包络线绘制

```c#
/// Finding the envelop of min and max value for LB_Keogh
        /// Implementation idea is intoruduced by Danial Lemire in his paper
        /// "Faster Retrieval with a Two-Pass Dynamic-Time-Warping Lower Bound", Pattern Recognition 42(9), 2009.
        public static void lower_upper_lemire(double[] t, int len, int r, double[] l, double[] u)
        {
            Deque du = new Deque();
            Deque dl = new Deque();

            init(ref du, 2 * r + 2);
            init(ref dl, 2 * r + 2);

            push_back(ref du, 0);
            push_back(ref dl, 0);

            for (int i = 1; i < len; i++)
            {
                if (i > r)
                {
                    u[i - r - 1] = t[front(ref du)];
                    l[i - r - 1] = t[front(ref dl)];
                }
                if (t[i] > t[i - 1])
                {
                    pop_back(ref du);
                    while (!du.Empty && t[i] > t[back(ref du)])
                        pop_back(ref du);
                }
                else
                {
                    pop_back(ref dl);
                    while (!dl.Empty && t[i] < t[back(ref dl)])
                        pop_back(ref dl);
                }
                push_back(ref du, i);
                push_back(ref dl, i);
                if (i == 2 * r + 1 + front(ref du))
                    pop_front(ref du);
                else if (i == 2 * r + 1 + front(ref dl))
                    pop_front(ref dl);
            }
            for (int i = len; i < len + r + 1; i++)
            {
                u[i - r - 1] = t[front(ref du)];
                l[i - r - 1] = t[front(ref dl)];
                if (i - front(ref du) >= 2 * r + 1)
                    pop_front(ref du);
                if (i - front(ref dl) >= 2 * r + 1)
                    pop_front(ref dl);
            }
        }
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/Figure_1.png)

```c#
 /// LB_Keogh 1: Create Envelop for the query
        /// Note that because the query is known, envelop can be created once at the begenining.
        ///
        /// Variable Explanation,
        /// order : sorted indices for the query.
        /// uo, lo: upper and lower envelops for the query, which already sorted.
        /// t     : a circular array keeping the current data.
        /// j     : index of the starting location in t
        /// cb    : (output) current bound at each position. It will be used later for early abandoning in DTW.
        private static double lb_keogh_cumulative(long[] order, double[] t, double[] uo, double[] lo, double[] cb,
                                                  long j, int len, double mean, double std,
                                                  double best_so_far = double.PositiveInfinity)
        {
            double lb = 0;
            double x, d;

            for (int i = 0; i < len && lb < best_so_far; i++)
            {
                x = (t[(order[i] + j)] - mean) / std;
                d = 0;
                if (x > uo[i])
                    d = dist(x, uo[i]);
                else if (x < lo[i])
                    d = dist(x, lo[i]);
                lb += d;
                cb[order[i]] = d;
            }
            return lb;
        }
```

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200526191916114.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200526203936526.png)

**level**: 
**author**:  François Petitjean   Faculty of IT, Monash University
**date**: 2014 
**keyword**:

- DTW  Sequence Analyze

------

## Paper:  DBA

<div align=center>
<br/>
<b>Dynamic Time Warping Averaging of Time Series  allows Faster and more Accurate Classification </b>
</div>

#### Summary

1. exploit  a recent result to allow meaningful averaging of warped times series, to allows us to create ultra-efficient Nearest "Centroid " classifiers that at least as accurate as their more lethargic Nearest Neighbor cousins.
2. application area:  reducing the data cardinality,   reducing the data dimensionality(the idea works well when the raw data is oversampled),       reducing the number of objects the nearest neighbor algorithm must see.

#### Research Objective

- **Application Area**: sequence analyse
- **Purpose**:  using DBA method to represent  a category

#### Proble Statement

previous work:

- NN-DTW algorithm are competitive or superior in domains as diverse as gesture recognition, robotics and ECG classification[1]. 
- DBA can be used to speed up NN-DTW by constructing the most representative time series of each class and using only those for training.
- sometiems NCC and NN can have approximately the same accuracy, in such cases we prefer NCC because it is faster and requires less memory.
- sometiems NCC can be more accurate than NN, in such cases we prefer NCC because of the accuracy gains, and the reduced computational requirements come for free.

#### Methods

- **Problem Formulation**:

  - Definitions: Dataset $D=\{ T_1,...,T_N\}$ , $T=(t_1,t_2,...,t_L)$,  L is the length.
  - Averaging under time warping : Finding the multiple alignment of  a set of sequences, or its average sequence(often called consesus sequence in biology) is a typical chicken-and-egg problem: knowig the average sequence provides a multiple alignment and vice versa, Finding the solution to the multiple alignment problem( and thus finding of an average sequence) has been shown to be NP-complete with the exact solution requiring$O(L^N)$ operations for N sequences of length L.
  - Average object: given a set of objects $O=\{O_1,...,O_N)\}$  in a space E indeced by a measure d, the average object $\vec{o} is the object that minimizes the sum of the squares to the set:

  $$
  argmin_{\vec{o}\epsilon E} \sum_{i=1}^Nd^2(\vec{o},O_i)
  $$


  - Average time series for DTW: 

$$
  argmin_{\vec{T}\epsilon E}\sum_{i=1}^N DTW^2(\vec{T},T_i)
$$

  - DBA:  the best-so-far method to average time series for Dynamic Time Warping:  DBA iteratively refines an average sequence $\vec{T}$  and folows an expectation-maximization scheme:
    - Consider  $\vec{T}$   fixed and find the best multiple alignment  M   of the set of sequences D   consistently with  $\vec{T}$
    - consider the M fixed and update $\vec{T}$ as the best average sequence consistent with M

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200305122636067.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200305122617085.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200305123050110.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200305123133018.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200305123625681.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200305123304559.png)

#### Evaluation

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200305120253517.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200526222446448.png)

- Left) NN has error-rate of 12.60%, while the Nearest Centroid classifier (right) with the same instances achieves an error-rate of just 5.22%

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200526222631759.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200526222804005.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200526225000764.png)

#### Notes 

- Nearest centroid classifier [1]

- code  available: 23   Matlab and java source code for DBA

  



**level**: 
**author**: Germain Forestier,University of Haute-Alsace, Mulhouse, France
**date**: 
**keyword**:

- DTW, Series generate

------

## Paper: Generating Synthetic series

<div align=center>
<br/>
<b>Generating synthetic time series to augment sparse datasets
</b>
</div>



#### Summary

1. extend DBA to calculate a weighted average of time series under DTW.
2. enlarge training sets by generating synthetic (or artificial) examples.
3. can generate an unlimited number of synthetic time series and tailor the weights distribution to achieve diversity.
4. deal with cold start problem, and use synthetic time series to double the trainning sets size regardless of their original sizes.

#### Research Objective

- **Application Area**:  Sequence Analyse
- **Purpose**:   cold start problem, synthetic time series to double the training sets size regardless of their original sizes.

#### Proble Statement

based information

- in some cases, it can be easier to experss our knowledge of the problem by generating synthetic data than by modifying the classifier itself. For instance, images containing street numbers on houses can be slightly rotated without changing what number they actually are. Voice can be slightly accelerated or slowed down without modifying the meaning. we can replace some words in a sentence by a close synonym without completely altering its meaning.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200305110536197.png)

previous work:

- Le Guennec et al. proposed to stretch or shrink randomly selected slices of a time series in order to create synthetic examples.

#### Methods

- **Problem Formulation**:

  - Definition 1:  A dataset $D=\{T_1,T_2,...,T_N\}$, for $T_1$:  $T_1=<t_1,t_2,...,t_L>$, L is the length.
  - Definition2: Average time series for DTW: 

  $$
  argmin\vec{T}\epsilon E\sum_{i=1}^NDTW^2(\vec{T},T_i)
  $$

  

  - Definition 3:  weighted average of time series under DTW, given a weighted set of time series $D=\{(T_1,w_1),...,(T_N,w_N)\} $in a space E induced by DTW, $\vec{T}$ the average time series is the time series that minimizes :

$$
argmin\vec{T}\epsilon E\sum_{i=1}^Nw_i*DTW^2(\vec{T},T_i)
$$



-  DBA uses expectation-maximization scheme and iteratively refines a starting average $\vec{T}$  by:
   - Expectation: Considering $\vec{T}$  fixed and finding the best multiple alignment M of the set of sequence D consistenly with $\vec{T}$ 
   - Maximization: considering M fixed and updating $\vec{T} $ as the best average sequence consistent with M.

**【Problem Define】**

- how to compute a weighted average consistently with dynamic time warping
- how to decide upon the weights to give to each times series.

**[Weighted average of time series for DTW]**

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200305111507465.png)

DTW_alignment:   找到DTW匹配时的path所对应的序列对，  将序列对D中对应元素值相加，然后统计个数取平均,其中代码片段如下：

```java
while (pathMatrix[i][j] != DBA.NIL) {
    updatedMean[i] += T[j];
    nElementsForMean[i]++;
    move = pathMatrix[i][j];
    i += moveI[move];
    j += moveJ[move];
}
assert (i != 0 || j != 0);
updatedMean[i] += T[j];
nElementsForMean[i]++;
```

medoid 方法java代码如下：

```java
private static int approximateMedoidIndex(double[][] sequences, double[][] mat) {
		/*
		 * we are finding the medoid, as this can take a bit of time, if
		 * there is more than 50 time series, we sample 50 as possible
		 * medoid candidates
		 */
		ArrayList<Integer> allIndices = new ArrayList<>();
		for (int i = 0; i < sequences.length; i++) {
			allIndices.add(i);
		}
		Collections.shuffle(allIndices);
		ArrayList<Integer> medianIndices = new ArrayList<>();
		for (int i = 0; i < sequences.length && i < 50; i++) {
			medianIndices.add(allIndices.get(i));
		}

		int indexMedoid = -1;
		double lowestSoS = Double.MAX_VALUE;

		for (int medianCandidateIndex : medianIndices) {
			double[] possibleMedoid = sequences[medianCandidateIndex];
			double tmpSoS = sumOfSquares(possibleMedoid, sequences, mat);
			if (tmpSoS < lowestSoS) {
				indexMedoid = medianCandidateIndex;
				lowestSoS = tmpSoS;
			}
		}
		return indexMedoid;
	}

	private static double sumOfSquares(double[] sequence, double[][] sequences, double[][] mat) {
		double sos = 0.0;
		for (int i = 0; i < sequences.length; i++) {
			double dist = DTW(sequence, sequences[i], mat);
			sos += dist * dist;
		}
		return sos;
	}
```



**[Average All]**

> We ﬁrst propose to sample the weights vector following a ﬂat Dirichlet distribution with unit concentration parameter w ⇠Dir(1). We used a low value for the shape parameter (0.2 in this paper) of the Gamma-distributed random variable used for the Dirichlet distribution in order to give more weight to a time series that is then used as the initial object to update in Weighted DBA algorithm
>
> the following two methods ﬁrst select a subset of the time series to average

[ Average Selected (AS) ]

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200305112112422.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200305112129474.png)

[ Average Selected with Distance (ASD)]

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200305112407656.png)

#### Notes 

- code available： https://github.com/fpetitjean/DBA

- ##### Dirichlet distribution

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200305113855885.png)

**level**: 2017 PMLR  , Sydney Australia
**author**:  Macro Cuturi, Mathieu Blondel
**date**:   2017
**keyword**:

- DTW, Sequence prediction

------

## Paper: Soft-DTW

<div align=center>
<br/>
<b>Soft-DTW: a Differentiable Loss Function for Time-Series</b>
</div>

#### Summary

1. propose a differentiable learning loss between time series, building upon the DTW discrepancy.
2. computes the soft-minimum of all alignment costs, that both its value and gradient can be computed with quadratic time complexity.
3. propose to use soft-DTW as a fitting term to compare the output of a machine synthesizing a time series segment with a ground truth observation.

#### Research Objective

  - **Application Area**:   Sequence Analyse
- **Purpose**:    the compare data is not a vector, but a sequence

#### Proble Statement

- the gradients of soft-DTW to all of its variables can be computed as a by-product of the computation of the discrepancy itself, with an added quadratic storage cost.

#### Methods

**【Opinion 1】DTW and soft-DTW loss function**

![image-20200527092022176](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200527092022176.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200527091952121.png)

【Wait to read clearly】don't understand the follows.

#### Evaluation

  - **Environment**:   
    - Dataset: 
- Average with soft-DTW loss

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200527092731729.png)

- Clustering with the soft-DTW geometry

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200527092811040.png)

- Multi-step-ahead prediction

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20200527091559497.png)

#### Notes <font color=orange>去加强了解</font>

  - coda available:  https: //github.com/mblondel/soft-dtw. 

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/dtw-sequence-analysis/  

