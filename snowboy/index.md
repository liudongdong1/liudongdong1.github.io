# SnowBoy


> snowboy 是一个`开源的、轻量级语音唤醒引擎`，可以通过它很轻松地创建属于自己的类似“hey, Siri” 的唤醒词。它的主要特性如下：
>
> - 高度可定制性。可自由创建和训练属于自己的唤醒词 始终倾听。
> - 可离线使用，无需联网，保护隐私。精确度高，低延迟 轻量可嵌入。
> - 耗费资源非常低（单核700MHz 树莓派只占用 10% CPU）
> - 开源跨平台。开放源代码，支持多种操作系统和硬件平台，可绑定多种编程语言

- Demo Code

```python
import snowboydecoder
import sys
import signal
interrupted = False
def signal_handler(signal, frame):
    global interrupted
    interrupted = True
def interrupt_callback():
    global interrupted
    return interrupted
if len(sys.argv) == 1:
    print("Error: need to specify model name")
    print("Usage: python demo.py your.model")
    sys.exit(-1)
model = sys.argv[1]
# capture SIGINT signal, e.g., Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

detector = snowboydecoder.HotwordDetector(model, sensitivity=0.5)
print('Listening... Press Ctrl+C to exit')

# main loop
detector.start(detected_callback=snowboydecoder.play_audio_file,
               interrupt_check=interrupt_callback,
               sleep_time=0.03)

detector.terminate()
```



### Resource

- 开源代码： https://github.com/seasalt-ai/snowboy
- 使用教程：
  -  https://www.wandianshenme.com/play/smart-speaker-hotword-detection-engine-snowboy-setup-guide/
  - https://blog.csdn.net/qq_38113006/article/details/105745564

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/snowboy/  

