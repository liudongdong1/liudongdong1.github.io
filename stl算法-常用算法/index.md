# STL算法-常用算法


> STL 算法部分主要由头文件 <algorithm>,<numeric>,<functional> 组成。要使用 STL 中的算法函数必须包含头文件 <algorithm>，对于数值算法须包含 <numeric>，<functional> 中则定义了一些模板类，用来声明函数对象。
>
> - 非可变序列算法：指不直接修改其所操作的容器内容的算法。
> - 可变序列算法：指可以修改它们所操作的容器内容的算法。
> - 排序算法：包括对序列进行排序和合并的算法、搜索算法以及有序序列上的集合操作。
> - 数值算法：对容器内容进行数值计算。

### 1 查找算法

查找算法共 13 个，包含在 <algorithm> 头文件中，用来提供元素排序策略，这里只列出一部分算法：

- adjacent_find: 在 iterator 对标识元素范围内，查找一对相邻重复元素，找到则返回指向这对元素的第一个元素的 ForwardIterator。否则返回 last。重载版本使用输入的二元操作符代替相等的判断。
- count: 利用等于操作符，把标志范围内的元素与输入值比较，返回相等元素个数。
- count_if: 利用输入的操作符，对标志范围内的元素进行操作，返回结果为 true 的个数。
- binary_search: 在有序序列中查找 value，找到返回 true。重载的版本实用指定的比较函数对象或函数指针来判断相等。
- equal_range: 功能类似 equal，返回一对 iterator，第一个表示 lower_bound，第二个表示 upper_bound。
- find: 利用底层元素的等于操作符，对指定范围内的元素与输入值进行比较。当匹配时，结束搜索，返回指向该元素的 Iterator。
- find_if: 使用输入的函数代替等于操作符执行 find。
- search: 给出两个范围，返回一个 ForwardIterator，查找成功指向第一个范围内第一次出现子序列 (第二个范围) 的位置，查找失败指向 last1。重载版本使用自定义的比较操作。
- search_n: 在指定范围内查找 val 出现 n 次的子序列。重载版本使用自定义的比较操作。

```c++
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>  

using namespace std;

int main(int argc, char* argv[])
{
	int iarr[] = { 0, 1, 2, 3, 4, 5, 6, 6, 6, 7, 8 };
	vector<int> iv(iarr, iarr + sizeof(iarr) / sizeof(int));

	/*** adjacent_find: 在iterator对标识元素范围内，查找一对相邻重复元素 ***/
	// 原型： _FwdIt adjacent_find(_FwdIt _First, _FwdIt _Last)
	cout << "adjacent_find: ";
	cout << *adjacent_find(iv.begin(), iv.end()) << endl;

	/*** count: 利用等于操作符，把标志范围内的元素与输入值比较，返回相等元素个数。 ***/
	// 原型： count(_InIt _First, _InIt _Last, const _Ty& _Val)
	cout << "count(==7): ";
	cout << count(iv.begin(), iv.end(), 6) << endl;// 统计6的个数

	/*** count_if: 利用输入的操作符，对标志范围内的元素进行操作，返回结果为true的个数。 ***/
	// 原型： count_if(_InIt _First, _InIt _Last, _Pr _Pred)
	// 统计小于7的元素的个数 :9个
	cout << "count_if(<7): ";
	cout << count_if(iv.begin(), iv.end(), bind2nd(less<int>(), 7)) << endl;

	/*** binary_search: 在有序序列中查找value，找到返回true。 ***/
	// 原型： bool binary_search(_FwdIt _First, _FwdIt _Last, const _Ty& _Val)
	cout << "binary_search: ";
	cout << binary_search(iv.begin(), iv.end(), 4) << endl; // 找到返回true

	/*** equal_range: 功能类似equal，返回一对iterator，第一个表示lower_bound，第二个表示upper_bound。 ***/
	// 原型： equal_range(_FwdIt _First, _FwdIt _Last, const _Ty& _Val)
	pair<vector<int>::iterator, vector<int>::iterator> pairIte;  
	pairIte = equal_range(iv.begin(), iv.end(), 3);
	cout << "pairIte.first:" << *(pairIte.first) << endl;// lowerbound 3   
	cout << "pairIte.second:" << *(pairIte.second) << endl; // upperbound 4

	/*** find: 利用底层元素的等于操作符，对指定范围内的元素与输入值进行比较。 ***/
	// 原型： _InIt find(_InIt _First, _InIt _Last, const _Ty& _Val)
	cout << "find: ";
	cout << *find(iv.begin(), iv.end(), 4) << endl; // 返回元素为4的元素的下标位置

	/*** find_if: 使用输入的函数代替等于操作符执行find。 ***/
	// 原型： _InIt find_if(_InIt _First, _InIt _Last, _Pr _Pred)
	cout << "find_if: " << *find_if(iv.begin(), iv.end(), bind2nd(greater<int>(), 2)) << endl; // 返回大于2的第一个元素的位置：3 

	/*** search: 给出两个范围，返回一个ForwardIterator，查找成功指向第一个范围内第一次出现子序列的位置。 ***/
	// 原型： _FwdIt1 search(_FwdIt1 _First1, _FwdIt1 _Last1, _FwdIt2 _First2, _FwdIt2 _Last2)
	// 在iv中查找 子序列 2 3 第一次出现的位置的元素   
	int iarr3[3] = { 2, 3 };
	vector<int> iv3(iarr3, iarr3 + 2);
	cout << "search: " << *search(iv.begin(), iv.end(), iv3.begin(), iv3.end()) << endl;

	/*** search_n: 在指定范围内查找val出现n次的子序列。 ***/
	// 原型： _FwdIt1 search_n(_FwdIt1 _First1, _FwdIt1 _Last1, _Diff2 _Count, const _Ty& _Val)
	// 在iv中查找 2个6 出现的第一个位置的元素   
	cout << "search_n: " << *search_n(iv.begin(), iv.end(), 2, 6) << endl;  

	return 0;
}

/*
adjacent_find: 6
count(==7): 3
count_if(<7): 9
binary_search: 1
pairIte.first:3
pairIte.second:4
find: 4
find_if: 3
search: 2
search_n: 6
*/
```

### 2 排序和通用算法

排序算法共 14 个，包含在 <algorithm> 头文件中，用来判断容器中是否包含某个值，这里只列出一部分算法：

- merge: 合并两个有序序列，存放到另一个序列。重载版本使用自定义的比较。
- random_shuffle: 对指定范围内的元素随机调整次序。重载版本输入一个随机数产生操作。
- nth_element: 将范围内的序列重新排序，使所有小于第 n 个元素的元素都出现在它前面，而大于它的都出现在后面。重载版本使用自定义的比较操作。
- reverse: 将指定范围内元素重新反序排序。
- sort: 以升序重新排列指定范围内的元素。重载版本使用自定义的比较操作。
- stable_sort: 与 sort 类似，不过保留相等元素之间的顺序关系。

```c++
#include "stdafx.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional> // 定义了greater<int>()

using namespace std;

// 要注意的技巧
template <class T>
struct display
{
	void operator()(const T&x) const
	{
		cout << x << " ";
	}
};

// 如果想从大到小排序，可以采用先排序后反转的方式，也可以采用下面方法:
// 自定义从大到小的比较器，用来改变排序方式
bool Comp(const int& a, const int& b) {
	return a > b;
}

int main(int argc, char* argv[])
{
	int iarr1[] = { 0, 1, 2, 3, 4, 5, 6, 6, 6, 7, 8 };
	vector<int> iv1(iarr1, iarr1 + sizeof(iarr1) / sizeof(int));
	vector<int> iv2(iarr1 + 4, iarr1 + 8); // 4 5 6 6
	vector<int> iv3(15);

	/*** merge: 合并两个有序序列，存放到另一个序列 ***/
	// iv1和iv2合并到iv3中（合并后会自动排序）
	merge(iv1.begin(), iv1.end(), iv2.begin(), iv2.end(), iv3.begin());
	cout << "merge合并后: ";
	for_each(iv3.begin(), iv3.end(), display<int>());
	cout << endl;

	/*** random_shuffle: 对指定范围内的元素随机调整次序。 ***/
	int iarr2[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	vector<int> iv4(iarr2, iarr2 + sizeof(iarr2) / sizeof(int));
	// 打乱顺序  
	random_shuffle(iv4.begin(), iv4.end());
	cout << "random_shuffle打乱后: ";
	for_each(iv4.begin(), iv4.end(), display<int>());
	cout << endl;

	/*** nth_element: 将范围内的序列重新排序。 ***/
	// 将小于iv.begin+5的放到左边   
	nth_element(iv4.begin(), iv4.begin() + 5, iv4.end());
	cout << "nth_element重新排序后: ";
	for_each(iv4.begin(), iv4.end(), display<int>());
	cout << endl;

	/*** reverse: 将指定范围内元素重新反序排序。 ***/
	reverse(iv4.begin(), iv4.begin());
	cout << "reverse翻转后: ";
	for_each(iv4.begin(), iv4.end(), display<int>());
	cout << endl;

	/*** sort: 以升序重新排列指定范围内的元素。 ***/
	// sort(iv4.begin(), iv4.end(), Comp); // 也可以使用自定义Comp()函数
	sort(iv4.begin(), iv4.end(), greater<int>());
	cout << "sort排序（倒序）: ";
	for_each(iv4.begin(), iv4.end(), display<int>());
	cout << endl;

	/*** stable_sort: 与sort类似，不过保留相等元素之间的顺序关系。 ***/
	int iarr3[] = { 0, 1, 2, 3, 3, 4, 4, 5, 6 };
	vector<int> iv5(iarr3, iarr3 + sizeof(iarr3) / sizeof(int));
	stable_sort(iv5.begin(), iv5.end(), greater<int>());
	cout << "stable_sort排序（倒序）: ";
	for_each(iv5.begin(), iv5.end(), display<int>());
	cout << endl;

	return 0;
}

/*
merge合并后: 0 1 2 3 4 4 5 5 6 6 6 6 6 7 8
random_shuffle打乱后: 8 1 6 2 0 5 7 3 4
nth_element重新排序后: 0 1 2 3 4 5 6 7 8
reverse翻转后: 0 1 2 3 4 5 6 7 8
sort排序（倒序）: 8 7 6 5 4 3 2 1 0
stable_sort排序（倒序）: 6 5 4 4 3 3 2 1 0
*/
```

### 3 删除和替换算法

删除和替换算法共 15 个，包含在 <numeric> 头文件中，这里只列出一部分算法：

- copy: 复制序列。
- copy_backward: 与 copy 相同，不过元素是以相反顺序被拷贝。
- remove: 删除指定范围内所有等于指定元素的元素。注意，该函数不是真正删除函数。内置函数不适合使用 remove 和 remove_if 函数。
- remove_copy: 将所有不匹配元素复制到一个制定容器，返回 OutputIterator 指向被拷贝的末元素的下一个位置。
- remove_if: 删除指定范围内输入操作结果为 true 的所有元素。
- remove_copy_if: 将所有不匹配元素拷贝到一个指定容器。

```c++
#include "stdafx.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional> // 定义了greater<int>()

using namespace std;

template <class T>
struct display
{
	void operator()(const T&x) const
	{
		cout << x << " ";
	}
};

int main(int argc, char* argv[])
{
	int iarr1[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	vector<int> iv1(iarr1, iarr1 + sizeof(iarr1) / sizeof(int));
	vector<int> iv2(9);

	/*** copy: 复制序列 ***/
	//  原型： _OutIt copy(_InIt _First, _InIt _Last,_OutIt _Dest)
	copy(iv1.begin(), iv1.end(), iv2.begin());
	cout << "copy(iv2): ";
	for_each(iv2.begin(), iv2.end(), display<int>());
	cout << endl;

	/*** copy_backward: 与copy相同，不过元素是以相反顺序被拷贝。 ***/
	//  原型： _BidIt2 copy_backward(_BidIt1 _First, _BidIt1 _Last,_BidIt2 _Dest)
	copy_backward(iv1.begin(), iv1.end(), iv2.rend());
	cout << "copy_backward(iv2): ";
	for_each(iv2.begin(), iv2.end(), display<int>());
	cout << endl;

	/*** remove: 删除指定范围内所有等于指定元素的元素。 ***/
	//  原型： _FwdIt remove(_FwdIt _First, _FwdIt _Last, const _Ty& _Val)
	remove(iv1.begin(), iv1.end(), 5); // 删除元素5
	cout << "remove(iv1): ";
	for_each(iv1.begin(), iv1.end(), display<int>());
	cout << endl;

	/*** remove_copy: 将所有不匹配元素复制到一个制定容器，返回OutputIterator指向被拷贝的末元素的下一个位置。 ***/
	//  原型： 	_OutIt remove_copy(_InIt _First, _InIt _Last,_OutIt _Dest, const _Ty& _Val)
	vector<int> iv3(8);
	remove_copy(iv1.begin(), iv1.end(), iv3.begin(), 4); // 去除4 然后将一个容器的元素复制到另一个容器
	cout << "remove_copy(iv3): ";
	for_each(iv3.begin(), iv3.end(), display<int>());
	cout << endl;

	/*** remove_if: 删除指定范围内输入操作结果为true的所有元素。 ***/
	//  原型： _FwdIt remove_if(_FwdIt _First, _FwdIt _Last, _Pr _Pred)
	remove_if(iv3.begin(), iv3.end(), bind2nd(less<int>(), 6)); //  将小于6的元素 "删除"
	cout << "remove_if(iv3): ";
	for_each(iv3.begin(), iv3.end(), display<int>());
	cout << endl;

	/*** remove_copy_if: 将所有不匹配元素拷贝到一个指定容器。 ***/
	// 原型： _OutIt remove_copy_if(_InIt _First, _InIt _Last,_OutIt _Dest, _Pr _Pred)
	//  将iv1中小于6的元素 "删除"后，剩下的元素再复制给iv3
	remove_copy_if(iv1.begin(), iv1.end(), iv2.begin(), bind2nd(less<int>(), 4));
	cout << "remove_if(iv2): ";
	for_each(iv2.begin(), iv2.end(), display<int>());
	cout << endl;

	return 0;
}

/*
copy(iv2): 0 1 2 3 4 5 6 7 8
copy_backward(iv2): 8 7 6 5 4 3 2 1 0
remove(iv1): 0 1 2 3 4 6 7 8 8
remove_copy(iv3): 0 1 2 3 6 7 8 8
remove_if(iv3): 6 7 8 8 6 7 8 8
remove_if(iv2): 4 6 7 8 8 3 2 1 0
*/
```

- replace: 将指定范围内所有等于 vold 的元素都用 vnew 代替。
- replace_copy: 与 replace 类似，不过将结果写入另一个容器。
- replace_if: 将指定范围内所有操作结果为 true 的元素用新值代替。
- replace_copy_if: 与 replace_if，不过将结果写入另一个容器。
- swap: 交换存储在两个对象中的值。
- swap_range: 将指定范围内的元素与另一个序列元素值进行交换。
- unique: 清除序列中重复元素，和 remove 类似，它也不能真正删除元素。重载版本使用自定义比较操作。
- unique_copy: 与 unique 类似，不过把结果输出到另一个容器。

```c++
#include "stdafx.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional> // 定义了greater<int>()

using namespace std;

template <class T>
struct display
{
	void operator()(const T&x) const
	{
		cout << x << " ";
	}
};

int main(int argc, char* argv[])
{
	int iarr[] = { 8, 10, 7, 8, 6, 6, 7, 8, 6, 7, 8 };
	vector<int> iv(iarr, iarr + sizeof(iarr) / sizeof(int));

	/*** replace: 将指定范围内所有等于vold的元素都用vnew代替。 ***/
	//  原型： void replace(_FwdIt _First, _FwdIt _Last, const _Ty& _Oldval, const _Ty& _Newval)
	// 将容器中6 替换为 3   
	replace(iv.begin(), iv.end(), 6, 3);
	cout << "replace(iv): ";
	for_each(iv.begin(), iv.end(), display<int>()); // 由于_X是static 所以接着 增长
	cout << endl; // iv:8 10 7 8 3 3 7 8 3 7 8   

	/*** replace_copy: 与replace类似，不过将结果写入另一个容器。 ***/
	//  原型： _OutIt replace_copy(_InIt _First, _InIt _Last, _OutIt _Dest, const _Ty& _Oldval, const _Ty& _Newval)
	vector<int> iv2(12);
	// 将容器中3 替换为 5，并将结果写入另一个容器。  
	replace_copy(iv.begin(), iv.end(), iv2.begin(), 3, 5);
	cout << "replace_copy(iv2): ";
	for_each(iv2.begin(), iv2.end(), display<int>());  
	cout << endl; // iv2:8 10 7 8 5 5 7 8 5 7 8 0（最后y一个残留元素）   

	/*** replace_if: 将指定范围内所有操作结果为true的元素用新值代替。 ***/
	//  原型： void replace_if(_FwdIt _First, _FwdIt _Last, _Pr _Pred, const _Ty& _Val)
	// 将容器中小于 5 替换为 2   
	replace_if(iv.begin(), iv.end(), bind2nd(less<int>(), 5), 2);
	cout << "replace_copy(iv): ";
	for_each(iv.begin(), iv.end(), display<int>());   
	cout << endl; // iv:8 10 7 8 2 5 7 8 2 7 8   

	/*** replace_copy_if: 与replace_if，不过将结果写入另一个容器。 ***/
	//  原型： _OutIt replace_copy_if(_InIt _First, _InIt _Last, _OutIt _Dest, _Pr _Pred, const _Ty& _Val)
	// 将容器中小于 5 替换为 2，并将结果写入另一个容器。  
	replace_copy_if(iv.begin(), iv.end(), iv2.begin(), bind2nd(equal_to<int>(), 8), 9);
	cout << "replace_copy_if(iv2): ";
	for_each(iv2.begin(), iv2.end(), display<int>()); 
	cout << endl; // iv2:9 10 7 8 2 5 7 9 2 7 8 0(最后一个残留元素)

	int iarr3[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, };
	vector<int> iv3(iarr3, iarr3 + sizeof(iarr3) / sizeof(int));
	int iarr4[] = { 8, 10, 7, 8, 6, 6, 7, 8, 6, };
	vector<int> iv4(iarr4, iarr4 + sizeof(iarr4) / sizeof(int));

	/*** swap: 交换存储在两个对象中的值。 ***/
	//  原型： _OutIt replace_copy_if(_InIt _First, _InIt _Last, _OutIt _Dest, _Pr _Pred, const _Ty& _Val)
	// 将两个容器中的第一个元素交换  
	swap(*iv3.begin(), *iv4.begin());
	cout << "swap(iv3): ";
	for_each(iv3.begin(), iv3.end(), display<int>());  
	cout << endl;

	/*** swap_range: 将指定范围内的元素与另一个序列元素值进行交换。 ***/
	//  原型： _FwdIt2 swap_ranges(_FwdIt1 _First1, _FwdIt1 _Last1, _FwdIt2 _Dest)
	// 将两个容器中的全部元素进行交换  
	swap_ranges(iv4.begin(), iv4.end(), iv3.begin());
	cout << "swap_range(iv3): ";
	for_each(iv3.begin(), iv3.end(), display<int>());
	cout << endl;

	/*** unique: 清除序列中相邻的重复元素，和remove类似，它也不能真正删除元素。 ***/
	//  原型： _FwdIt unique(_FwdIt _First, _FwdIt _Last, _Pr _Pred) 
	unique(iv3.begin(), iv3.end());
	cout << "unique(iv3): ";
	for_each(iv3.begin(), iv3.end(), display<int>());
	cout << endl;

	/*** unique_copy: 与unique类似，不过把结果输出到另一个容器。 ***/
	//  原型： _OutIt unique_copy(_InIt _First, _InIt _Last, _OutIt _Dest, _Pr _Pred)
	unique_copy(iv3.begin(), iv3.end(), iv4.begin());
	cout << "unique_copy(iv4): ";
	for_each(iv4.begin(), iv4.end(), display<int>());
	cout << endl;

	return 0;
}

/*
replace(iv): 8 10 7 8 3 3 7 8 3 7 8
replace_copy(iv2): 8 10 7 8 5 5 7 8 5 7 8 0
replace_copy(iv): 8 10 7 8 2 2 7 8 2 7 8
replace_copy_if(iv2): 9 10 7 9 2 2 7 9 2 7 9 0
swap(iv3): 8 1 2 3 4 5 6 7 8
swap_range(iv3): 0 10 7 8 6 6 7 8 6
unique(iv3): 0 10 7 8 6 7 8 6 6
unique_copy(iv4): 0 10 7 8 6 7 8 6 8
*/
```

### 4 排列组合算法

排列组合算法共 2 个，包含在 <algorithm> 头文件中，用来提供计算给定集合按一定顺序的所有可能排列组合，这里全部列出：

- next_permutation: 取出当前范围内的排列，并重新排序为下一个字典序排列。重载版本使用自定义的比较操作。
- prev_permutation: 取出指定范围内的序列并将它重新排序为上一个字典序排列。如果不存在上一个序列则返回 false。重载版本使用自定义的比较操作。

```c++
#include "stdafx.h"
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

template <class T>
struct display
{
	void operator()(const T&x) const
	{
		cout << x << " ";
	}
};

int main(int argc, char* argv[])
{
	int iarr[] = { 12, 17, 20, 22, 23, 30, 33, 40 };
	vector<int> iv(iarr, iarr + sizeof(iarr) / sizeof(int));

	/*** next_permutation: 取出当前范围内的排列，并重新排序为下一个字典序排列。***/
	//  原型： bool next_permutation(_BidIt _First, _BidIt _Last)
	// 生成下一个排列组合（字典序）   
	next_permutation(iv.begin(), iv.end());
	for_each(iv.begin(), iv.end(), display<int>());
	cout << endl;

	/*** prev_permutation: 取出指定范围内的序列并将它重新排序为上一个字典序排列。 ***/
	//  原型： bool prev_permutation(_BidIt _First, _BidIt _Last)
	prev_permutation(iv.begin(), iv.end());
	for_each(iv.begin(), iv.end(), display<int>());
	cout << endl;

	return 0;
}

/*
12 17 20 22 23 30 40 33
12 17 20 22 23 30 33 40
*/
```

### 5 数值算法

数值算法共 4 个，包含在 <numeric> 头文件中，分别是：

- accumulate: iterator 对标识的序列段元素之和，加到一个由 val 指定的初始值上。重载版本不再做加法，而是传进来的二元操作符被应用到元素上。
- partial_sum: 创建一个新序列，其中每个元素值代表指定范围内该位置前所有元素之和。重载版本使用自定义操作代替加法。
- inner_product: 对两个序列做内积 (对应元素相乘，再求和) 并将内积加到一个输入的初始值上。重载版本使用用户定义的操作。
- adjacent_difference: 创建一个新序列，新序列中每个新值代表当前元素与上一个元素的差。重载版本用指定二元操作计算相邻元素的差。

```c++
#include "stdafx.h"
#include <iostream>
#include <vector>
#include <numeric> // 数值算法
#include <iterator> // 定义了ostream_iterator

using namespace std;

int main(int argc, char* argv[])
{
	int arr[] = { 1, 2, 3, 4, 5 };
	vector<int> vec(arr, arr + 5);
	vector<int> vec2(arr, arr + 5);

	//  accumulate: iterator对标识的序列段元素之和，加到一个由val指定的初始值上。
	int temp;
	int val = 0;
	temp = accumulate(vec.begin(), vec.end(), val);
	cout << "accumulate(val = 0): " << temp << endl;
	val = 1;
	temp = accumulate(vec.begin(), vec.end(), val);
	cout << "accumulate(val = 1): " << temp << endl;

	// inner_product: 对两个序列做内积(对应元素相乘，再求和)并将内积加到一个输入的初始值上。
	// 这里是：1*1 + 2*2 + 3*3 + 4*4 + 5*5
	val = 0;
	temp = inner_product(vec.begin(), vec.end(), vec2.begin(), val);
	cout << "inner_product(val = 0): " << temp << endl;

	// partial_sum: 创建一个新序列，其中每个元素值代表指定范围内该位置前所有元素之和。
	// 第一次，1   第二次，1+2  第三次，1+2+3  第四次，1+2+3+4
	ostream_iterator<int> oit(cout, " "); // 迭代器绑定到cout上作为输出使用
	cout << "ostream_iterator: ";
	partial_sum(vec.begin(), vec.end(), oit);// 依次输出前n个数的和
	cout << endl;
	// 第一次，1   第二次，1-2  第三次，1-2-3  第四次，1-2-3-4
	cout << "ostream_iterator(minus): ";
	partial_sum(vec.begin(), vec.end(), oit, minus<int>());// 依次输出第一个数减去（除第一个数外到当前数的和）
	cout << endl;

	// adjacent_difference: 创建一个新序列，新序列中每个新值代表当前元素与上一个元素的差。
	// 第一次，1-0   第二次，2-1  第三次，3-2  第四次，4-3
	cout << "adjacent_difference: ";
	adjacent_difference(vec.begin(), vec.end(), oit); // 输出相邻元素差值 后面-前面
	cout << endl;
	// 第一次，1+0   第二次，2+1  第三次，3+2  第四次，4+3
	cout << "adjacent_difference(plus): ";
	adjacent_difference(vec.begin(), vec.end(), oit, plus<int>()); // 输出相邻元素差值 后面-前面 
	cout << endl;

	return 0;
}

/*
accumulate(val = 0): 15
accumulate(val = 1): 16
inner_product(val = 0): 55
ostream_iterator: 1 3 6 10 15
ostream_iterator(minus): 1 -1 -4 -8 -13
adjacent_difference: 1 1 1 1 1
adjacent_difference(plus): 1 3 5 7 9
*/
```



### 6 生成和异变算法

生成和异变算法共 6 个，包含在 <algorithm> 头文件中，这里只列出一部分算法：

- fill: 将输入值赋给标志范围内的所有元素。
- fill_n: 将输入值赋给 first 到 first+n 范围内的所有元素。
- for_each: 用指定函数依次对指定范围内所有元素进行迭代访问，返回所指定的函数类型。该函数不得修改序列中的元素。
- generate: 连续调用输入的函数来填充指定的范围。
- generate_n: 与 generate 函数类似，填充从指定 iterator 开始的 n 个元素。
- transform: 将输入的操作作用与指定范围内的每个元素，并产生一个新的序列。重载版本将操作作用在一对元素上，另外一个元素来自输入的另外一个序列。结果输出到指定容器。

```c++
#include "stdafx.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

using namespace std;

template <class T>
struct display
{
	void operator()(const T&x) const
	{
		cout << x << " ";
	}
};
//  作用类似于上面结构体，只不过只能显示int类型的数据
void printElem(int& elem)
{
	cout << elem << " ";
}

template<class T>
struct plus2
{
	void operator()(T&x)const
	{
		x += 2;
	}

};

class even_by_two
{
private:
	static int _x; //  注意静态变量   
public:
	int operator()()const
	{
		return _x += 2;
	}
};
int even_by_two::_x = 0; //  初始化静态变量

int main(int argc, char* argv[])
{
	int iarr[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	vector<int> iv(iarr, iarr + sizeof(iarr) / sizeof(int));

	/*** fill: 将输入值赋给标志范围内的所有元素。 ***/
	//  原型： void fill(_FwdIt _First, _FwdIt _Last, const _Ty& _Val)  
	fill(iv.begin(), iv.end(),5);
	cout << "fill: ";
	for_each(iv.begin(), iv.end(), display<int>());
	cout << endl;

	/*** fill_n: 将输入值赋给first到first+n范围内的所有元素。 ***/
	//  原型： _OutIt fill_n(_OutIt _Dest, _Diff _Count, const _Ty& _Val)
	fill_n(iv.begin(), 4, 3); //  赋4个3给iv 
	cout << "fill_n: ";
	for_each(iv.begin(), iv.end(), display<int>());
	cout << endl;

	/*** for_each: 用指定函数依次对指定范围内所有元素进行迭代访问，返回所指定的函数类型。 ***/
	//  原型： _Fn1 for_each(_InIt _First, _InIt _Last, _Fn1 _Func)
	for_each(iv.begin(), iv.end(), plus2<int>()); //  每个元素+2
	cout << "for_each: ";
	for_each(iv.begin(), iv.end(), printElem); //  输出
	cout << endl;

	/*** generate: 连续调用输入的函数来填充指定的范围。 ***/
	//  原型： void generate(_FwdIt _First, _FwdIt _Last, _Fn0 _Func)
	//  使用even_by_two()函数返回的值，来填充容器iv
	generate(iv.begin(), iv.end(), even_by_two());
	cout << "generate: ";
	for_each(iv.begin(), iv.end(), display<int>());
	cout << endl;

	/*** generate_n: 与generate函数类似，填充从指定iterator开始的n个元素。 ***/
	//  原型： _OutIt generate_n(_OutIt _Dest, _Diff _Count, _Fn0 _Func)
	//  使用even_by_two()函数返回的值，来填充容器iv的前三个值
	generate_n(iv.begin(), 3, even_by_two());
	cout << "generate_n: ";
	for_each(iv.begin(), iv.end(), display<int>()); //  由于_X是static 所以接着 增长
	cout << endl;

	/*** transform: 将输入的操作作用与指定范围内的每个元素，并产生一个新的序列。 ***/
	//  原型： _OutIt transform(_InIt _First, _InIt _Last, _OutIt _Dest, _Fn1 _Func)
	// 容器的所有值全部减2
	transform(iv.begin(), iv.end(), iv.begin(), bind2nd(minus<int>(), 2));
	cout << "transform: ";
	for_each(iv.begin(), iv.end(), display<int>()); //  由于_X是static 所以接着 增长
	cout << endl;

	return 0;
}

/*
fill: 5 5 5 5 5 5 5 5 5
fill_n: 3 3 3 3 5 5 5 5 5
for_each: 5 5 5 5 7 7 7 7 7
generate: 2 4 6 8 10 12 14 16 18
generate_n: 20 22 24 8 10 12 14 16 18
transform: 18 20 22 6 8 10 12 14 16
*/
```



### 7 关系算法

关系算法共 8 个，包含在 <algorithm> 头文件中，这里只列出一部分算法：

- equal: 如果两个序列在标志范围内元素都相等，返回 true。重载版本使用输入的操作符代替默认的等于操作符。
- includes: 判断第一个指定范围内的所有元素是否都被第二个范围包含，使用底层元素的 < 操作符，成功返回 true。重载版本使用用户输入的函数。
- max: 返回两个元素中较大一个。重载版本使用自定义比较操作。
- min: 返回两个元素中较小一个。重载版本使用自定义比较操作。
- max_element: 返回一个 ForwardIterator，指出序列中最大的元素。重载版本使用自定义比较操作。
- min_element: 返回一个 ForwardIterator，指出序列中最小的元素。重载版本使用自定义比较操作。
- mismatch: 并行比较两个序列，指出第一个不匹配的位置，返回一对 iterator，标志第一个不匹配元素位置。如果都匹配，返回每个容器的 last。重载版本使用自定义的比较操作。

```c++
#include "stdafx.h"
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main(int argc, char* argv[])
{
	int iarr[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	vector<int> iv1(iarr, iarr + 5);
	vector<int> iv2(iarr, iarr + 9);

	//  equal: 如果两个序列在标志范围内元素都相等，返回true。
	cout <<"equal: " << equal(iv1.begin(), iv1.end(), iv2.begin()) << endl;//  1 表示相等，因为只比较跟 iv1长度大小的数组      

	// includes: 判断第一个指定范围内的所有元素是否都被第二个范围包含，使用底层元素的<操作符，成功返回true。
	// 判断判断iv2中元素是否都出现在iv1中
	cout << "includes: " << includes(iv1.begin(), iv1.end(), iv2.begin(), iv2.end()) << endl;

	// max: 返回两个元素中较大一个。
	cout << "max: " << max(iv1[0],iv1[1]) << endl;
	// min: 返回两个元素中较小一个。
	cout << "min: " << min(iv1[0], iv1[1]) << endl;

	// max_element: 返回一个ForwardIterator，指出序列中最大的元素。
	cout << "max_element: " << *max_element(iv1.begin(), iv1.end()) << endl;
	// min_element: 返回一个ForwardIterator，指出序列中最小的元素。
	cout << "min_element: " << *min_element(iv1.begin(), iv1.end()) << endl;

	//  mismatch: 并行比较两个序列，指出第一个不匹配的位置，返回一对iterator，标志第一个不匹配元素位置。如果都匹配，返回每个容器的last。
	pair<vector<int>::iterator, vector<int>::iterator> pa;
	pa = mismatch(iv1.begin(), iv1.end(), iv2.begin());
	if (pa.first == iv1.end()) //  true 表示相等，因为只比较跟iv1长度大小的数组 
		cout << "第一个向量与第二个向量匹配" << endl;
	else
	{
		cout << "两个向量不同点--第一个向量点:" << *(pa.first) << endl; // 这样写很危险，应该判断是否到达end   
		cout << "两个向量不同点--第二个向量点:" << *(pa.second) << endl;
	}

	return 0;
}

/*
equal: 1
includes: 0
max: 2
min: 1
max_element: 5
min_element: 1
第一个向量与第二个向量匹配
*/
```



### 8 **集合算法**

集合算法共 4 个，包含在 <algorithm> 头文件中，这里全部列出：

- set_union: 构造一个有序序列，包含两个序列中所有的不重复元素。重载版本使用自定义的比较操作。
- set_intersection: 构造一个有序序列，其中元素在两个序列中都存在。重载版本使用自定义的比较操作。
- set_difference: 构造一个有序序列，该序列仅保留第一个序列中存在的而第二个中不存在的元素。重载版本使用自定义的比较操作。
- set_symmetric_difference: 构造一个有序序列，该序列取两个序列的对称差集 (并集 - 交集)。

```c++
#include "stdafx.h"
#include <iostream>
#include <set>
#include <algorithm>
#include <iterator> 

using namespace std;

template <class T>
struct display
{
	void operator()(const T&x) const
	{
		cout << x << " ";
	}
};

int main(int argc, char* argv[])
{
	int iarr1[] = { 1, 3, 5, 7, 9, 11 };
	int iarr2[] = { 1, 1, 2, 3, 5, 8, 13 };

	multiset<int> s1(iarr1, iarr1 + 6);
	multiset<int> s2(iarr2, iarr2 + 7);
	cout << "s1: ";
	for_each(s1.begin(), s1.end(), display<int>());
	cout << endl;
	cout << "s2: ";
	for_each(s2.begin(), s2.end(), display<int>());
	cout << endl;

	/*** set_union: 构造一个有序序列，包含两个序列中所有的不重复元素。 ***/
	//  原型： _OutIt set_union(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _InIt2 _Last2, _OutIt _Dest)
	cout << "union of s1 and s2: ";
	// 两个集合合并，相同元素个数取 max(m,n)。   
	set_union(s1.begin(), s1.end(), s2.begin(), s2.end(), ostream_iterator<int>(cout, " "));
	cout << endl;

	/*** set_intersection: 构造一个有序序列，其中元素在两个序列中都存在。 ***/
	//  原型： _OutIt set_union(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _InIt2 _Last2, _OutIt _Dest)
	cout << "Intersection of s1 and s2: ";
	// 两个集合交集，相同元素个数取 min(m,n).  
	set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(), ostream_iterator<int>(cout, " "));
	cout << endl;

	/*** set_difference: 构造一个有序序列，该序列仅保留第一个序列中存在的而第二个中不存在的元素。 ***/
	//  原型： _OutIt set_union(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _InIt2 _Last2, _OutIt _Dest)
	cout << "Intersection of s1 and s2: ";
	// 两个集合差集 就是去掉S1中 的s2   
	set_difference(s1.begin(), s1.end(), s2.begin(), s2.end(), ostream_iterator<int>(cout, " "));
	cout << endl;

	/*** set_symmetric_difference: 构造一个有序序列，该序列取两个序列的对称差集(并集-交集)。 ***/
	//  原型： _OutIt set_union(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _InIt2 _Last2, _OutIt _Dest)
	cout << "Intersection of s1 and s2: ";
	// 两个集合对称差集：就是取两个集合互相没有的元素 。两个排序区间，元素相等指针后移，不等输出小的并前进   
	// 相同元素的个数 abs(m-n)   
	set_symmetric_difference(s1.begin(), s1.end(), s2.begin(), s2.end(), ostream_iterator<int>(cout, " "));
	cout << endl;

	return 0;
}

/*
s1: 1 3 5 7 9 11
s2: 1 1 2 3 5 8 13
union of s1 and s2: 1 1 2 3 5 7 8 9 11 13
Intersection of s1 and s2: 1 3 5
Intersection of s1 and s2: 7 9 11
Intersection of s1 and s2: 1 2 7 8 9 11 13
*/
```

### 9 堆算法

集合算法共 4 个，包含在 <algorithm> 头文件中，这里只列出一部分算法：

- make_heap: 把指定范围内的元素生成一个堆。重载版本使用自定义比较操作。
- pop_heap: 并不真正把最大元素从堆中弹出，而是重新排序堆。它把 first 和 last-1 交换，然后重新生成一个堆。可使用容器的 back 来访问被 "弹出" 的元素或者使用 pop_back 进行真正的删除。重载版本使用自定义的比较操作。
- push_heap: 假设 first 到 last-1 是一个有效堆，要被加入到堆的元素存放在位置 last-1，重新生成堆。在指向该函数前，必须先把元素插入容器后。重载版本使用指定的比较操作。

```c++
#include "stdafx.h"
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

template <class T>
struct display
{
	void operator()(const T&x) const
	{
		cout << x << " ";
	}
};

int main(int argc, char* argv[])
{
	int iarr[] = { 4, 5, 1, 3, 2 };
	vector<int> iv(iarr, iarr + sizeof(iarr) / sizeof(int));

	/*** make_heap: 把指定范围内的元素生成一个堆。 ***/
	//  原型： void make_heap(_RanIt _First, _RanIt _Last)
	make_heap(iv.begin(), iv.end());
	cout << "make_heap: ";
	for_each(iv.begin(), iv.end(), display<int>());
	cout << endl;

	/*** pop_heap: 并不真正把最大元素从堆中弹出，而是重新排序堆。 ***/
	//  原型： void pop_heap(_RanIt _First, _RanIt _Last)
	pop_heap(iv.begin(), iv.end());
	iv.pop_back();
	cout << "pop_heap: ";
	for_each(iv.begin(), iv.end(), display<int>());
	cout << endl;

	/*** push_heap: 假设first到last-1是一个有效堆，要被加入到堆的元素存放在位置last-1，重新生成堆。 ***/
	//  原型： void push_heap(_RanIt _First, _RanIt _Last)
	iv.push_back(6);
	push_heap(iv.begin(), iv.end());
	cout << "push_heap: ";
	for_each(iv.begin(), iv.end(), display<int>());
	cout << endl;

	return 0;
}

/*
make_heap: 5 4 1 3 2
pop_heap: 4 3 1 2
push_heap: 6 4 1 2 3
*/
```

### Resouce

- https://www.cnblogs.com/linuxAndMcu/p/10264339.html

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/stl%E7%AE%97%E6%B3%95-%E5%B8%B8%E7%94%A8%E7%AE%97%E6%B3%95/  

