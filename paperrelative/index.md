# PaperRelative


> Ahsan, H., Prabhu, A., Deeksha, S. D., Domanal, S. G., Ashwin, T. S., & Reddy, G. R. M. (2014, August). Vision based laser controlled keyboard system for the disabled. In *Proceedings of the 7th International Symposium on Visual Information Communication and Interaction* (pp. 200-203).

------

# Paper: Keyboard

<div align=center>
<br/>
<b>Vision based laser controlled keyboard system for the disabled
</b>
</div>

#### Summary

1. design one such unistroke keyboard system with optimized character placement based on frequenctly used digraphs and trigraphs.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210119195510106.png)

#### Module

- **Calibration:** 
  - purpose: the angle and position of the keyboard can change every time it's set up, so its necessary to know how the exact position and size of the captured keyboard;
  - Methods: 
    - Input: a captured frame;
    - set of points representing the corners of the keyboard;
      - convert captured frame to hsv;
      - define the range for blue color detection;
      - get the contours that lie within the range;
      - filter the contours based on area;
      - store the center of each of these contours;
      - calibrate the keyboard angle;
  - **Calibrate keyboard angle:** 
    - input:  set of points representing the corners of the keyboard;
    - output: the calibrated keyboard;
      - sort all the points according to x distance;
      - divide sorted points into 3 parts as: left(0-6), mid(7-14), right(15-21)
      - each part: sort based on y distance; sort consecutive pairs of points based x distance;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210119200203556.png)

- **Tracker the laser:** using the average of a fixed number of frames to represent the position;

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210119200742617.png)



> *CamK:* *Camera-based* *Keystroke* *Detection* *and* *Localization* *for* *Small* *Mobile* Devices. IEEE Transaction on *Mobile* Computing, vol. 17, no. 10, pp. 2236-2251.

------

# Paper: CamK

<div align=center>
<br/>
<b>CamK: Camera-Based Keystroke Detection and Localization for Small Mobile Devices</b>
</div>

#### Phenomenon&Challenge:

1. location deviation,the inner key distance is only about two centimeters,  a position deviation between the real fingertip and the detected fingertip   (initial traininng to get the optimal parameters for image processing ,then using an extended region to represented the detected fingertip  the fingertip is located in the key for a certain duration)
2. false positives (for non-keystroke,combine keystroke detection with keystroke localization,and introduce online calibration,using the movemnet features of the fingertip after a keystroke)
3. processing latency ,using adaptively changing images sizes ,focusing on the target area in the large-size images,adopting multiple threads and removing the operations of writing/reading images.

#### RelatedWork:

1. virtual keyboards including wearable keyboards like rings and gloves    on-screen keyboards    Projector keyboard   audio signal and camera based virtual keyboards

#### Chart&Analyse:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1571817295530.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1571817340622.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1571817351729.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/1571817368504.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1571817386271.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/1571817396275.png)

> Carver, Charles J., et al. "AmphiLight: Direct Air-Water Communication with Laser Light." *17th {USENIX} Symposium on Networked Systems Design and Implementation ({NSDI} 20)*. 2020.

------

# Paper: AmphiLight

<div align=center>
<br/>
<b>AmphiLight: Direct Air-Water Communication with Laser Light</b>
</div>
#### Summary

1. propose a bidirectional, direct air-water wireless communication link based on laser light, capable of:
   - adapting to water dynamics with ultrasonic sensing;
   - steering within a full 3D hemisphere using only MEMS mirror and passive optical elements.
2. judiciously design the basic communication link to overcome issues of existing laser hardware and improve its portability for communication.
3. to handle strong ambient light interference, we exploit the narrow spectral power distribution of laser light by placing a narrow optical filter in front of an ultra-sensitive receiver to filter out ambient light and maintain sufficient signal-to-noise ratios,
4. to adapt to environmental dynamics, propose a new optical system to enable precise, full-hemisphere laser steering using low-cost, portable hardware.
5. address water dynamics by augmenting the link with ultrasonic sensing and a forecasting method.

#### Research Objective

  - **Application Area**: underwater operations, environmental monitoring, surveying, or coordinating of heterogeneous aerial and underwater systems.

#### Proble Statement

- Existing wireless techniques mostly focus on a single physical medium and full short in achieving high-bandwidth bidirectional communication across the air-water interface.
  - let the underwater vehicles surface to share data;
  - deploy an infrastructure at the water surface, connected to both the underwater assets( via acoustic transducers, completely in the water) and ground station(via tethering or wifi)

previous work:

- Compare to acoustic, the light communication supports much shorter communication latency with faster propagation speeds
- Compared to RF, it endures much lower attenuation in the water.

#### Methods

- **Problem Formulation**:
  - limited communication between assets underwater and in the air.
- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210119205131576.png)

> Sami S, Tan S R X, Sun B, et al. LAPD: Hidden Spy Camera Detection using Smartphone Time-of-Flight Sensors[C]//Proceedings of the 19th **ACM Conference on Embedded Networked Sensor Systems**. **2021**: 288-301. CCF_B

------

# Paper:LAPD

<div align=center>
<br/>
<b>LAPD: Hidden Spy Camera Detection using Smartphone Time-of-Flight Sensors</b>
</div>

#### Summary

1. Present LAPD, a novel hidden camera detection and localization system that leverages the time-of-flight sensor on commodity smartphones.

#### Research Objective

  - **Application Area**: privacy preserve,  detect hidden spy cameras concealed in sensitive locations
- **Purpose**:  use smartphone to automatically detect and lcoalize hidden cameras.
- **Basic phenomenon:** the hidden camera embedded in the object reflects the incoming laser pulses at a higher intensity than its surroundings due to an effect called lens-sensor-retro-reflection.( `when almost all light energy impacting an object is reflected directly back to the source`.)

#### Proble Statement

- hidden cameras are difficult to find with naked eyes.
- **Goal**: detect the presence and location of hidden cameras using only the information available from smartphones with ToF sensors.
  - accessibility: already-existing comodity smartphones with ToF sensors
  - accuracy: correctly identify the presence and location of hidden cameras
  - usability: automatically detect these hidden cameras with minimal user intervention.

previous work:

- `specialized equipment` like commercial "hidden camera detectors" but` yield low detection rates`![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220118223130718.png)
- analyze the wireless traffic that the cameras generate can `only detect the presence of the hidden cameras, except their locations.`
- failed when the video record in local memory card.

#### Methods

- **system overview**:

> The user first uses LAPD to select a suspicious object. LAPD then guides them to determine the ideal scan distance, and scan the object for hidden cameras. During the scan, LAPD uses a computer vision and machine learning processing pipeline to detect reflections from hidden cameras while rejecting false positives



![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220118220426147.png)

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220121191628644.png)

> - **Depth map:** a ToF sensor captures depth information by emitting light from a laser, receiving its reflections, and storing the data in a two-dimensional depth map;
> - **Depth map:** a confidence measure of the estimated depth to indicate its accuracy and validity.
>
> LAPD utilizes both depth and intensity maps to identify high-intensity retro-reflections from cameras.

- **3D localization:**
  - using the smartphone's camera and inertial sensors to perform Simultaneous Localization and Mapping of the environment, and tracks the smartphone's position and orientation within it.
- **Suspicious Object Selection:**
  - user draws a 2D bounding box with a center point around the object in the phone, and the phone using the transform function to tracking  the object in 3D space.
- **Scan Distance Computation:**
  - first guides the user to move to within 20 cm of the object to saturate the ToF sensor。
  - then asks them to gradually move further from the object

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220121191708778.png)

【Challenge 1】**different objects that embed the hidden cameras may cause varying reflectivity**

> - To find ideal location
>   - close: the ToF sensor will oversaturate
>   - faraway: insufficient light will reach the ToF sensor ahd hinder the obervation.
> - utilizes augmented reality to guide the user to move closer and further away from the object, calculating the ideal distance by determining the object's reflectivity at various distance.

【Challenge 2】**ToF sensor hardware limitation (spatial, bit resolutions) limiting the amout of information for distinguishing reflectiosn from the hidden camera lens as opposed to other reflections from the surroundings**

> - difficult to discern the shape, size and precise intensity of the reflections.
> - design and implement a chain of filters including deep-learning-based filters that incorporate multi-modal information ( depth and reflection intensities) to eliminate false positives.

【Challenge 3】**the reflections are limited by camera optical properties (20 FoV)**

- implementing an FoV filter that eliminates remaining reflections or candidate hidden cameras, which appear highly reflective outside the constrained angle.

#### Evaluation

![](https://gitee.com/github-25970295/blogimgv2022/raw/master/image-20220121192018342.png)

- camera housings; hidden camera modules, lighting conditions, smartphone types.


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/paperrelative/  

