# byteIntRelative


>1byte = 8 bit； 

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210419170144.png)

### 1. 强制转化截取和 精确转化

>`int强转成byte`，而不是本文所说的int转byte[]。通过前一个问题我们知道int占32位，而byte占8位。比如``int a10 = 261对应的二进制就是100000101`【左边剩下的23个0就省略了】，想转成byte也很简单，就是`直接截取最右边8位``即可：00000101【十进制的5】，也就是261强转byte后等于5。

```python
public byte[] toBytes(int number){
	        byte[] bytes = new byte[4];
	        bytes[3] = (byte)number;
	        bytes[2] = (byte) ((number >> 8) & 0xFF);
	        bytes[1] = (byte) ((number >> 16) & 0xFF);
	        bytes[0] = (byte) ((number >> 24) & 0xFF);
	        return bytes;
	 }
# 这俩个代码 功能是相同的
 public byte[] toBytes(int number){
	        byte[] bytes = new byte[4];
	        bytes[3] = (byte)number;
	        bytes[2] = (byte) (number >> 8);
	        bytes[1] = (byte) (number >> 16);
	        bytes[0] = (byte) (number >> 24);
	        return bytes;
	 }
```

- 大端模式&小端模式

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20210419170518.png)

### 2. 进制转化

```python
#  十六进制 到 十进制 注意这里 16进制有引号
int('0xf',16) 
#	二进制 到 十进制  注意这里 2进制有引号
int('10100111110',2)      
#	八进制 到 十进制  注意这里 8进制有引号
int('17',8)    
#	十进制 转 十六进制，
hex(1033)  #输出：'0x409'
#	二进制 转 十六进制  进制先转成 十进制， 再转成 十六进制，其他转同这个；
hex(int('101010',2))
#	十进制转二进制
bin(10)
#	十六进制转 二进制
bin(int('ff',16))
#	二进制 到 八进制
oct(0b1010)
```

### 3. 字节转化

>bytearray和bytes不一样的地方在于，`bytearray`是可变的。`Bytes 代表的是（二进制）数字的序列`，只不过在是通过 `ASCII` 编码之后才是我们看到的字符形式，如果我们单独取出一个字节，它仍然是一个数字：

```python
#将二进制byte序列转化为 bytes 类型二进制序列
type(b'xxxxx')
```

- str《-》bytes

```python
#str --》 bytes
str = '人生苦短，我用Python!'
bytes = str.encode()
print(bytes)
#bytes --》 str
bytes = b'\xe4\xba\xba\xe7\x94\x9f\xe8\x8b\xa6\xe7\x9f\xad\xef\xbc\x8c\xe6\x88\x91\xe7\x94\xa8Python!'
str = bytes.decode()
print(str)
```

- 字符串转字节串

```python
字符串编码为字节码: '12abc'.encode('ascii')  ==>  b'12abc'
数字或字符数组: bytes([1,2, ord('1'),ord('2')])  ==>  b'\x01\x0212'
16进制字符串: bytes().fromhex('010210')  ==>  b'\x01\x02\x10'
16进制字符串: bytes(map(ord, '\x01\x02\x31\x32'))  ==>  b'\x01\x0212'
16进制数组: bytes([0x01,0x02,0x31,0x32])  ==>  b'\x01\x0212'
```

- 字节串转字符串

```python
字节码解码为字符串: bytes(b'\x31\x32\x61\x62').decode('ascii')  ==>  12ab
字节串转16进制表示,夹带ascii: str(bytes(b'\x01\x0212'))[2:-1]  ==>  \x01\x0212
字节串转16进制表示,固定两个字符表示: str(binascii.b2a_hex(b'\x01\x0212'))[2:-1]  ==>  01023132
字节串转16进制数组: [hex(x) for x in bytes(b'\x01\x0212')]  ==>  ['0x1', '0x2', '0x31', '0x32']
```

- bytearray 转化

```python
#str --》 bytearray 
str = '人生苦短，我用Python!'
bytes = bytearray(str.encode())
bytes = bytearray(b'\xe4\xba\xba\xe7\x94\x9f\xe8\x8b\xa6\xe7\x9f\xad\xef\xbc\x8c\xe6\x88\x91\xe7\x94\xa8Python!')
str = bytes.decode()
print(str)
#bytearray  --》 str
bytes[:6] = bytearray('生命'.encode())
bytes = bytearray(b'\xe7\x94\x9f\xe5\x91\xbd\xe8\x8b\xa6\xe7\x9f\xad\xef\xbc\x8c\xe6\x88\x91\xe7\x94\xa8Python!')
str = bytes.decode()
print(str)

c = bytes([2,3,6,8])  
print(c)
> b'\x02\x03\x06\x08'
```

### 4. 文件读写

```python
# Read the entire file as a single byte string
with open('somefile.bin', 'rb') as f:
    data = f.read(16)
    text = data.decode('utf-8')

# Write binary data to a file
with open('somefile.bin', 'wb') as f:
    text = 'Hello World'
    f.write(text.encode('utf-8'))
```

>二进制I/O还有一个鲜为人知的特性就是数组和C结构体类型能直接被写入，而不需要中间转换为自己对象

```python
import array
nums = array.array('i', [1, 2, 3, 4])
with open('data.bin','wb') as f:
    f.write(nums)
```

### 7. 移位&反码

- <<左移： 按二进制形式把所有的数字向左移动对应的位数，`高位移出（舍弃）`，`低位的空位补零`。
- \>>右移:    按二进制形式把所有的数字向右移动对应的位数，`低位移出（舍弃）`，`高位的空位补符号位`,即正数补0，负数补1。
- \>>>: 按二进制形式把所有的数字向右移动对应位数，低位移出（舍弃），高位的空位补零。对于正数来说和带符号右移相同，对于负数来说不同。
- x & y # 且操作，返回结果的每一位是 x 和 y 中对应位做 and 运算的结果，``只有 1 and 1 = 1，其他情况位0`
- x | y # 或操作，返回结果的每一位是 x 和 y 中对应位做 or 运算的结果，只有 `0 or 0 = 0，其他情况位1`
- ~x # 反转操作，对 x 求的每一位求补，只需记住结果是 -x - 1
- x ^ y # 或非运算，如果 y 对应位是0，那么结果位取 x 的对应位，如果 y 对应位是1，取 x 对应位的补

>java中的数字都是有符号的,即有正有负
>有符号数字表示的二进制中左边第一位是符号位，0表示正数，1表示负数
>原码：8位有符号的二级制的原始表达方式
>反码：原码符号位不变，其他位取反（就是0变1，1变0）
>补码：反码+1
>计算机中的数字使用补码表示
>使用规则： 正数的原码和补码一样（也就不存在反码），负数的反码根据上面的规则计算，即反码+1

### 8. 学习资源

- [java 进制转化操作](https://www.cnblogs.com/Marydon20170307/p/9167339.html)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/byteintrelative/  

