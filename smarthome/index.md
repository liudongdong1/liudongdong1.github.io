# SmartHome


![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223154456937.png)
**keyword**:

- Human-centered computing , LoRa

------

# Paper: WIDESEE

<div align=center>
<br/>
<b>WIDESEE: Towards Wide-Area Contactless Wireless Sensing</b>
</div>
#### Summary

1. WIDESEE presents solutions across software and hardware to overcome two aspects of challenges for wide-range contactless sensing:
   1. the interference brought by the device mobility and LoRa’s high sensitvity
   2. the ambiguous target information such as location when employing just a single pair of transceivers

#### Research Objective

- **Application Area**:
- **Purpose**:  

#### Proble Statement

-  **limited sensing range**, which hinders its applications in wide-area sensing such as disaster rescue.the signals reflected from the target, which contain information related to the context of the target, are much weaker than the direct path signals between the transmitter and receiver.WiFi-based systems are only capable of performing sensing in a room-level range (i.e. approximately 3-6 m),whereas RFID or mmWave-based systems show an even smaller sensing range of 1-3 m
-  LoRa offers a long propagation distance and strong penetration capability through obstacles.

challenges:

-  the larger sensing range of LoRa also means the interference range is also larger due to the higher signal receiving sensitivity

<font color=red> redesign the antenna system and the sensing algorithm, employ a compact reconfigurable directional antenna at the receiver to narrow down the target sensing region,to stay focus on the area of interests</font>

-  a transceiver pair equipped with a single antenna does not provide us sufficient information regarding the target location since the number of unknown variables is greater than that of the constrained equations for localization
-  although employing a drone can increase the sensing coverage, the vibration introduced by the drone during its operation (i.e., flying) affects the resultant signals and accordingly the target sensing performance.

**level**:  Mobile Data Management
**author**: Nirmalya Roy ,School of Information Systems, Singapore Management University
**date**: 2015
**keyword**:

-  energy demand estimation, ADLS

------

# Paper: Activity-Aware Room-level Power Analytics (AARPA)

<div align=center>
<br/>
<b>AARPA: Combining Mobile and Power-line
Sensing for Fine-grained Appliance Usage and
Energy Monitoring</b>
</div>


#### Summary

1. applies correlation over both macroscopic and microscopic power consumption features, to identify the total usage duration, and the total energy consumption, of individual devices, from such circuit-breaker level aggregated data.
2. helps capture the energy consumption characteristics of low-load,
   commonly-used domestic appliances
3. provides useful additional context about the lifestyle habits and context of
   an individual

#### Related Work

- Green Building Energy Management using Plug Load Meters
  - EnergyHub [1] and Greenbox [2]
  - employs machine learning on data collected from infrastructure sensors,such as magnetic sensors, has been proposed to infer fine grained power usage in home [8]
- Smartphone and Sensor based Energy Prediction:
  - Beware [3] provides the user information on energy consumption of entire home. Detect the electricity consumption of different devices and notify the user if the devices use more energy than expected
  - Energy Lens [11] provides deeper real time visibility of plug-load
    energy consumption in buildings. It uses the mobile phone to provide a consumer with real-time energy analytics

#### Research Objective

- the ability to precisely capture the usage profile of everyday consumer appliances also provides insight into an individual’s context
- Fine-grained monitoring of everyday appliances (such as toasters and coffee makers) can not only promote energy-efficient building operations, but also provide unique insights into the context and activities of individuals
- develop a novel correlation-based approach called CBPA to identify individual
  appliances based on both their unique transient and steady state power signatures.
- uses mobile sensing to first infer high-level activities of daily living (ADLs), and
  then uses knowledge of such ADLs to effectively reduce the set of candidate appliances that potentially contribute to the aggregate readings at any point

------

# Paper: Co-locating services

<div align=center>
<br/>
<b>Co-locating services in IoT systems to minimize
the communication energy cost</b>
</div>


#### Research Objective

- **Purpose**:  
  - .An issue for perpetually running and managing these IoT devices is the energy cost. One energy saving strategy is to co-locate several services on one device in order to reduce the computing and communication energy. 
  - propose a service merging strategy for mapping and co-locating multiple services on devices.

#### Proble Statement

-  various device sleep scheduling algorithms [3] to keep some devices power off
-  running at a low-power mode. Another approach is to reduce
   network communication traffic to conserve energy

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223135344822.png)

**level**: PACM Interact. Mob
**keyword**:

- Human-centered computing, Networks(wireless access points, base stations)

------

# Paper: RFID Light Bulb

<div align=center>
<br/>
<b>RFID Light Bulb: Enabling Ubiquitous Deployment of Interactive
RFID Systems</b>
</div>


#### Summary

1. leverage advances in semiconductor optics, RF antenna design and system integration to create a hybrid RFID reader and smart LED lamp, in the form factor of a standard light bulb
2. handle the difficult of deployment RFID system

#### Research Objective

- **Application Area**:
- **Purpose**:  

#### Proble Statement

- RFID :inexpensive, wireless, batery-free connectivity and interactivity for objects that are traditionally not instrumented
- complexity of installing bulky RFID readers, antennas, and their supporting power and network infrastructure

- IOT devices:door locks, security cameras, thermostats, voice-based personal assistants, and even simple butons that automate internet retail purchases.
  Bluetooth Low Energy, Zigbee, and Wi-Fi are all examples of networking technologies that connect these diferent classes of devices.
- RFID application:
  -  tracking the currently open page of a book
  -  sensing liquid level in container
  -  device free action recognition
  -  touch sensitive buttons
  -  thermometer
  -  contact switch

#### Methods

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223140127333.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223140615510.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223140639969.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223140701477.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223140719264.png)

#### Application

- Light-Assisted Navigation![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223140957214.png)
- Infrastructure Monitoring![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223140957214.png)
- Ambient Contextual Timers: a timer in conjunction with a
  machine to inform a user that a particular activity has completed
- Prepackaged Content:![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223141341090.png)

#### Conclusion

- designed an RFID Light Bulb (Figure 1): a Wi-Fi-connected smart LED bulb
  that contains an integrated RFID reader and antenna
- rototype whole-house interactive RFID applications that demonstrate the efectiveness of our bulbs

**level**: Journal of Intelligent & Fuzzy Systems
**author**: School of Engineering, Cardiff University, Cardiff, CF24 3AA, United Kingdom
**date**: 2019
**keyword**:

- A ctivity recognition, knowledge-driven approaches, data-driven approaches, activity model, hybrid reasoning

------

# Paper: Hybrid knowledge-data-driven

<div align=center>
<br/>
<b>A hybrid approach of knowledge-driven
and data-driven reasoning for activity
recognition in smart homes</b>
</div>


#### Summary

1. presents an alternative approach by combining knowledge-driven with data-driven reasoning to allow activity models to evolve and adapt automatically based on users’ particularities
2. a knowledge-driven reasoning is presented for inferring an initial activity model. The model is then trained using data-driven techniques to produce a dynamic activity model that learns users’ varying action

#### Research Objective

- **Application Area**: support and assistance for elderly, disabled and cognitively impaired people

#### Proble Statement

- activity recognition has become a primary indicator to measure physical and mental health of elderly individuals based on their ability to perform basic activities such as bathing, eating ,cooking  (这是想问题的一个切入点)
- there are different types of activities. The activity can be broken down into multiple levels of actions
- there is no strict constraint on the sequence of actions to perform the activities,depending on user’s preference and particularities
- the actions in which the activity is performed can be dynamic evolved.
- the model should be adapt to different environment and user’s behaviours

previous work:

- smart home:
  -  identify activities and patterns of daily routines <font color=red>context : location, time,object used ..</font>
  -  monitor environmental changes using sensors installed in different locations and deployed on various objects [4]
  -  activity recognition [5], predicting human behaviour [6] and detecting
     early diseases [7, 8]
- data-driven: machine learning or deep learning, code-start problem that require large sensor data, difficult to adapt in different environment
- knowledge-driven reasoning: priori knowledge about the world to build activities models using knowledge representation. <font color=red>the inference model is static and general ,difficult to recognize every type of human activities in home setting  </font>
  - rule based systems
  - case based systems
  - ontological reasoning

#### Methods

- **Problem Formulation**:

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223144151381.png)

![image-20191223144225224](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20191223144225224.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223144244631.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223144356678.png)

- The common-sense knowledge base contains a collection of semantic concepts and their relationships that are related to the basic understanding of the
  environment.
- the domain-specific knowledge base is used to represent concepts that are specifically described with respect to a certain domain in order to improve the principal understanding of the environment
- scenario: ![image-20191223145302558](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20191223145302558.png)
- infer users’ activities through a description logic rule-based inference system:![image-20191223145426841](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20191223145426841.png)

**level**: CVPR  
**author**:Fl´avia Dias Casagrande and Evi Zouganeli
**date**: 2019
**keyword**:

- Smart home Sequence prediction Time prediction Binary sensors · Recurrent neural network · Probabilistic Methods

------

# Paper: Activity Recognition Prediction

<div align=center>
<br/>
<b>Activity Recognition and Prediction in Real
Homes</b>
</div>


#### Summary

1. using probabilistic methods and Long Short-Term Memory (LSTM) networks, include the time information to improve prediction accuracy, as well as predict both the next sensor event and its time of occurrence using one LSTM model

#### Research Objective

- **Application Area**:assisting functions with reminders or encouragement, diagnosis tools, alarm creation, prediction ,anticipation and prevention of hazardous situations
- **Purpose**:  activity recognition and prediction in real homes using either binary sensor data or
  depth video data   classify four activities –no movement, standing up, sitting down, and TV interaction

- **system overview**:

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223150443175.png)

#### Notes 

- 是一家公司的文章介绍：https://www.roommate.no/en/home/

------

# Paper: Kinectcs datasets

<div align=center>
<br/>
<b>A Short Note on the Kinetics-700 Human Action Dataset</b>
</div>


#### Summary

1. http://deepmind.com/kinetics.
2. 1)action class sourcing, 2) candidate video matching, 3) candidate clip selection, 4) human verification, 5) quality analysis and filtering

**keyword**:

- smarthome

------

# Paper: Unified Frame ADL RecPre

<div align=center>
<br/>
<b>A Unified Framework for Activity Recognition-Based Behavior Analysis and
Action Prediction in Smart Homes</b>
</div>


#### Summary

- **Application Area**: lifestyle analysis, security and surveillance, and  interaction monitoring

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223151459042.png)

**keyword**:

- activity prediction,Location-based social networks(LBSNs)

------

# Paper: What’s Your Next Move

<div align=center>
<br/>
<b>What’s Your Next Move: User Activity Prediction in Location-based
Social Networks</b>
</div>


#### Summary

- exploit the check-in category information to model the underlying user movement pattern
- uses a mixed hidden Markov model to predict the category of user activity at the nect step and then predict the most likely location given the estimated category distribution
- **Difficult**： data sparseness  , the semantic meaning
- 一种基于签到的思想

#### Steps:

1. predicting the category of user activity at the next step
2. predicting a location given the estimated category distribution

**keyword**:

- activity prediction,Location-based social networks(LBSNs)

------

# Paper: SmrtFridge

<div align=center>
<br/>
<b>SmrtFridge: IoT-based, User Interaction-Driven Food Item &
!antity Sensing</b>
</div>


#### Summary

- identify the individual food items that users place in or remove from a fridge
- estimate the residual quantity of food items inside a refrigerated container (opaque or transparent)
- **Previous Works**:  RFID, camera, weight sensors

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223153020345.png)

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20191223153126433.png)

# Paper: Ambient-Assisted Living Tools 

<div align=center>
<br/>
<b>A Survey on Ambient-Assisted Living Tools
for Older Adults</b>
</div>


#### Summary

- 相关背景介绍，在写论文时候可以参考

## 智慧家庭中定位和行为预测

下图网格表示一个房间，网格中每一个点摆放RFID标签，rfid标签存储物体或者位置相关信息，可以根据相位或者RSSI分为几个状态，并将值量化。

- 想法一： 同一时候，会得到一张图，图中每个一像素点位置代表房间内rfid的位置，其值表示rfid对应的量化状态。  可以时间基本的定位，或者人物交互活动
- 想法二： 类比图像光流算法，根据状态点的变化移动，根据状态变化轨迹，可以识别粗粒度行为，进行相应的预测。



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/smarthome/  

