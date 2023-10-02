# Java_CMD


> 自动化的过程也已经成熟。 无需手动运行所有内容。 使用Java，我们可以运行单个或多个Shell命令，执行Shell脚本，运行终端/命令提示符，设置工作目录以及通过核心类操作环境变量。

### 1.Runtime.getRuntime().exec()

```java
Process exec(String command) 
在单独的进程中执行指定的字符串命令。 
 
Process exec(String[] cmdarray) 
在单独的进程中执行指定命令和变量。 
 
Process exec(String[] cmdarray, String[] envp) 
在指定环境的独立进程中执行指定命令和变量。 
 
Process exec(String[] cmdarray, String[] envp, File dir) 
在指定环境和工作目录的独立进程中执行指定的命令和变量。 
 
Process exec(String command, String[] envp) 
在指定环境的单独进程中执行指定的字符串命令。 
 
Process exec(String command, String[] envp, File dir) 
在有指定环境和工作目录的独立进程中执行指定的字符串命令。
```

```java
package util.model;

import java.io.*;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

public class CnnModelUtilv2 {
    private static final String CMD = "./venv/bin/python model_predict.py ";
    private static final String MEDICINE_LABLE_FILE_NAME = "/medicine_name-label.txt";

    private static final Map<Integer, String> MEDICINE_NAME_MAP = new HashMap<>();

    static {
        BufferedReader bufferedReader =
                new BufferedReader(new InputStreamReader(CnnModelUtilv2.class.getResourceAsStream(MEDICINE_LABLE_FILE_NAME)));

        bufferedReader.lines().forEach(v -> {
            String[] split = v.split(",");
            MEDICINE_NAME_MAP.put(Integer.valueOf(split[1]), split[0]);
        });
    }
    public static Map<String, Float> medicineNamePredict(File file) throws IOException {
        Process process = Runtime.getRuntime().exec(CMD + file.getAbsolutePath(), null,
                new File("./medicine-runtime-data/script"));
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        String line = reader.readLine();
        System.out.println(line);
        Map<String, Float> medicinePredict = new LinkedHashMap<>();
        if (line != null && !"null".equals(line)) {
            for (String s : line.split(",")) {
                String[] split = s.split(":");
                medicinePredict.put(MEDICINE_NAME_MAP.get(Integer.valueOf(split[0])), Float.valueOf(split[1]));
            }
            return medicinePredict;
        }
        return null;
    }
}
```

### 2. 调用shell命令

```java
private void callCMD(String tarName, String fileName, String... workspace){
	try {
		String cmd = "tar -cf" + tarName + " " + fileName;
//      String[] cmd = {"tar", "-cf", tarName, fileName};
		File dir = null;
		if(workspace[0] != null){
			dir = new File(workspace[0]);
			System.out.println(workspace[0]);
		}
		process = Runtime.getRuntime().exec(cmd, null, dir);
//      process = Runtime.getRuntime().exec(cmd);
		int status = process.waitFor();
		if(status != 0){
			System.err.println("Failed to call shell's command and the return status's is: " + status);
		}
	}
	catch (Exception e){
		e.printStackTrace();
	}
}
```

### 3. 调用shell脚本

```shell
#!/usr/bin/env bash
args=1
if [ $# -eq 1 ];then
	args=$1
	echo "The argument is: $args"
fi
echo "This is a $call"
start=`date +%s`
sleep 3s
end=`date +%s`
cost=$((($end - $start) * $args * $val))
echo "Cost Time: $cost"
```

```java
private void callScript(String script, String args, String... workspace){
	try {
		String cmd = "sh " + script + " " + args;
//        	String[] cmd = {"sh", script, "4"};
		File dir = null;
		if(workspace[0] != null){
			dir = new File(workspace[0]);
			System.out.println(workspace[0]);
		}
		String[] evnp = {"val=2", "call=Bash Shell"};
		process = Runtime.getRuntime().exec(cmd, evnp, dir);
//            process = Runtime.getRuntime().exec(cmd);
		BufferedReader input = new BufferedReader(new InputStreamReader(process.getInputStream()));
		String line = "";
		while ((line = input.readLine()) != null) {
			System.out.println(line);
		}
		input.close();
	}
	catch (Exception e){
		e.printStackTrace();
	}
}
 
public static void main(String[] args) {
	// TODO Auto-generated method stub
	CallShell call = new CallShell();
	call.callScript("test.sh", "4", "/root/experiment/");
}
```



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/java_cmd/  

