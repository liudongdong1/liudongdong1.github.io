# ROSRelative


http://wiki.ros.org/ROS/Tutorials

### Section one

![List of Distributions](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210210152119465.png)

- Every ROS release will be supported on exactly one Ubuntu LTS.
- LTS releases will not share a common Ubuntu release with any previous releases.
- ROS releases will not add support for new Ubuntu distributions after their release date.

![](https://gitee.com/github-25970295/blogImage/raw/master/img/image-20210210152106253.png)

###### FileSystem

```bash
rospack find [package_name]  #get information about package
roscd [locationname[/subdir]]  #equal to cd
echo $ROS_PACKAGE_PATH  
roscd log # take you to the folder where the log file locate
rosls [locationname[/subdir]]  #  ls the directory by name
```

###### ROS package

1. consist of a package

   a catkin compliant package.xml file , a CMakeLists.txt , and every package have ists own directory

   ```
   workspace_folder/        -- WORKSPACE
     src/                   -- SOURCE SPACE
       CMakeLists.txt       -- 'Toplevel' CMake file, provided by catkin
       package_1/
         CMakeLists.txt     -- CMakeLists.txt file for package_1
         package.xml        -- Package manifest for package_1
       ...
       package_n/
         CMakeLists.txt     -- CMakeLists.txt file for package_n
         package.xml        -- Package manifest for package_n
   ```

2. creating a package

   ```shell
   # You should have created this in the Creating a Workspace Tutorial
   $ cd ~/catkin_ws/src
   #create a new package called 'beginner_tutorials' which depends on std_msgs, roscpp, and rospy
   catkin_create_pkg beginner_tutorials std_msgs rospy roscpp
   # catkin_create_pkg <package_name> [depend1] [depend2] [depend3]
   #build the package in the catkin workspace
   cd ~/catkin_ws
   catkin_make
   #after the above op the file is  /opt/ros/$ROSDISTRO_NAME.
   #add the workspace to the ROS environment
   . ~/catkin_ws/devel/setup.bash
   #find the dependency
   rospack depends1 [packagename]
   ```

   

###### Element  

- [Nodes](http://wiki.ros.org/Nodes): A node is an executable that uses ROS to communicate with other nodes.
- [Messages](http://wiki.ros.org/Messages): ROS data type used when subscribing or publishing to a topic.
- [Topics](http://wiki.ros.org/Topics): Nodes can *publish* messages to a topic as well as *subscribe* to a topic to receive messages.
- [Master](http://wiki.ros.org/Master): Name service for ROS (i.e. helps nodes find each other)
- [rosout](http://wiki.ros.org/rosout): ROS equivalent of stdout/stderr
- [roscore](http://wiki.ros.org/roscore): Master + rosout + parameter server (parameter server will be introduced later)

1. node: A *node* is a process that performs computation. Nodes are combined together into a graph and communicate with one another using streaming [topics](http://wiki.ros.org/Topics), RPC [services](http://wiki.ros.org/Services), and the [Parameter Server](http://wiki.ros.org/Parameter Server). These nodes are meant to operate at a fine-grained scale; a robot control system will usually comprise many nodes. For example, one node controls a laser range-finder, one Node controls the robot's wheel motors, one node performs localization, one node performs path planning, one node provides a graphical view of the system, and so on.

   ```shell
   roscore # start by making sure that we have roscore running 
   #onfigure the talker node to publish to /wg/chatter instead of chatter:
   rosrun rospy_tutorials talker chatter:=/wg/chatter
   #Node parameter assignment
   rosrun rospy_tutorials talker _param:=1.0
   # client library support  rospy for python  roscpp for c++
   rosnode list  #displays the node that currently running 
   rosnode machine  #list nodes running on a particular machine or list machines
   rosnode info /rosout  # information about a node named /rosout
   rosndoe kill [nodename]
   rosnode cleanup     #purge registration information of unreachable nodes
   rosrun [package_name] [node_name]  #run a node within a package
   rosrun turtlesim turtlesim_node __name:=my_turtle # name the node while running
   rosnode ping my_turtle  #check whether the node is connected
   #roscore = ros+core : master (provides name service for ROS) + rosout (stdout/stderr) + parameter server (parameter server will be introduced later)
   ```

   | **Node Namespace** | **Remapping Argument** | **Matching Names** | **Final Resolved Name** |
   | ------------------ | ---------------------- | ------------------ | ----------------------- |
   | `/`                | `foo:=bar`             | `foo`, `/foo`      | `/bar`                  |
   | `/baz`             | `foo:=bar`             | `foo`, `/baz/foo`  | `/baz/bar`              |
   | `/`                | `/foo:=bar`            | `foo`, `/foo`      | `/bar`                  |
   | `/baz`             | `/foo:=bar`            | `/foo`             | `/baz/bar`              |
   | `/baz`             | `/foo:=/a/b/c/bar`     | `/foo`             | `/a/b/c/bar`            |

   - `__name` is a special reserved keyword for "the name of the node." It lets you remap the node name without having to know its actual name. It can only be used if the program that is being launched contains one node.

- `__log` is a reserved keyword that designates the location that the node's log file should be written. Use of this keyword is generally not encouraged -- it is mainly provided for use by ROS tools like [roslaunch](http://wiki.ros.org/roslaunch).
  - `__ip` and `__hostname` are substitutes for `ROS_IP` and `ROS_HOSTNAME`. Use of this keyword is generally not encouraged as it is provided for special cases where environment variables cannot be set.
- `__master` is a substitute for `ROS_MASTER_URI`. Use of this keyword is generally not encouraged as it is provided for special cases where environment variables cannot be set.
  - `__ns` is a substitute for `ROS_NAMESPACE`. Use of this keyword is generally not encouraged as it is provided for special cases where environment variables cannot be set.

2. Ros Topic   sing the [rostopic](http://wiki.ros.org/rostopic) and [rqt_plot](http://wiki.ros.org/rqt_plot) commandline tools.

   ```shell
   sudo apt install ros-<distro>-rqt
   sudo apt install ros-<distro>-rqt-commom-plugins
   rosrun rqt_graph rqt_graph   #to show the graph relations
   rostopic -h # help page
   rostopic bw #display bandwidth used by topic
   rostopic echo [topic] # print message to screen
   rostopic hz #display publishing rate of topic
   rostopic pub #publish data to topic
   rostopic type [topic] # print topic type
   #eg rostopic type /turtle1/cmd_vel  return  geometry_msgs/Twist
   #rosmsg show geometry_msgs/Twist for more detail information
   rostopic list -h # list help page
   #Ros Messages  : publisher and subscriber must send and receive the same type of message
   rostopic pub [topic] [msg_type] [args]
   #args need to accordance with msg_type eg:geometry_msgs/Vector3 linear
     float64 x
     float64 y
     float64 z
   geometry_msgs/Vector3 angular
     float64 x
     float64 y
     float64 z
   ```

3. Service : another way for node communicating with each other,service allow nodes to send request and get response

   ```shell
   rosservice list  # print information about active services
   rosservice call [service] [args] #call the service with the provided args
   rosservice type [service] #print service type
   rosservice find  #find services by service type
   rosservice uri  #print service POSRPC uri
   #rosparam allows you to store and manipulate data on the ROS parameter server
   rosparam set [param_name] args #set parameter  take effect after run call method
   rosparam get  # get parameter
   rosparam get /  #get all parameters with args
   rosparam load [file_name] [namespace] #load parameters from file
   rosparam dump [file_name] [namespace]  #dump parameters to file
   rosparam delete  #delete parameter
   rosparam list  #list parameter names
   ```

4. rqt_console && roslaunch   : for debugging and roslaunch for starting many nodes at once

   ```shell
   sudo apt install ros-<distro>-rqt-common-plugins ros-<distro>-turtlesim
   rosrun rqt_console rqt_console
   rosrun rqt_logger_level rqt_logger_level
   roslaunch [package] [filename.launch] #starts nodes as defined in a launch file
   ```

   a simple launch file:

   ```shell
   <launch>
   	<group ns="turtlesim1">
   		<node pkg="turtlesim" name="sim" type="turlesim_node"/>
   	</group>
   	<group ns=trutlesims2>  #namespace
   		<node pkg="turtlesim" name="sim" type="turtlesim_node"/>
   	</group>
   	<node pkg="turtlesim" name="mimic" type="mimic">
   		<remap from="input" to="turtlesim1/turtle1"/>
   		<remap from="output" to="turtlesim2/turtle2"/>
   	</node>
   </launch>
   ```

   ![mimiclaunch.jpg](http://wiki.ros.org/ROS/Tutorials/UsingRqtconsoleRoslaunch?action=AttachFile&do=get&target=mimiclaunch.jpg)

5. rosed  : directly edit a file within a package by using the package name rather than having to type the entire path to package

   ```shell
   rosed [package_name] [filename]
   #msg file are simple text file that describe the fields of a ROS message, used to generate source code for message in different languages
   #srb: describes a service ,composed of two parts,a request and a response
   rosmsg show [message_type]  #
   rossrv show <service type>
   catkin_make  # compiles a ROS package
   ```

###### Usage

1. python interface 

   - publish code

   ```python
   #!/usr/bin/env python
   import rospy
   from std_msgs.msg import String  #reuse the std_msgs/String message type
   def talker():
       pub=rospy.Publisher('chatter',String,queue_size=10) #publishing to the chatter topic using the message type string .
       rospy.init_node('talker',anonymous=True)#init a nodewith name talker,anonymous=True ensures that your node has a unique name by adding random numbers to the end of Name
       rate=rospy.Rate(10) #10hz
       while not rospy.is_shutdown():
           hello_str="hello world %s " % rospy.get_time()
           rospy.loginfo(hello_str)
           pub.publish(hello_str)
           rate.sleep()
   if __name__ =='__main__':
       try:
           talker()
       except rospy.ROSInterrupException:
           pass
   ```

   - subscribe code  roscd beginner_turials/scripts

   ```python
   !#/usr/bin/env python
   import rospy
   from std_msgs.msg import String
   def callback(data):
       rospy.loginfo(rospy.get_caller_id()+ "i heard %s",data.data)
   def lister():
       #in Ros nodes are uniquely named,if two nodes with the same name are launched ,the previous one is kicked off.
       rospy.init_node('listener',anonymous=True)
       rospy.Subscriber("chatter",String,callback)
       #spin() simply keeps python from exiting until this node is stopped 
       rospy.spin()
   if __name__=='__main':
       listener()
   ```

   - running

     ```shell
     roscore #make sure that a roscore is up and running 
     #cd to you catkin workspace
     cd ~/catkin_we
     catkin_make
     source ./devel/setup.bash
     rosrun beginner_tutorials talker (c++)
     rosrun beginner_tutorials talker.py (python)
     ```

2. cpp code

   ```c++
   #include "ros/ros.h"
   #include "beginner_tutorials/AddTwoInts.h"
   bool add(beginner_tutorials::AddTwoInts::Request &req,
           beginner_tutorials::AddTwoInts::Respense &res){
       res.sum=req.a+req.b;
       ROS_INFO("request: x=%1d, y=%1d",(long int)req.a,(long int )req.b);
       ROS_INFO("sending back response: [%1d]",(long int )res.sum);
       return true;
   }
   int main(int argc,char **argv){
       ros::init(argc,argv,"add_tow_ints_server");
       ros::NodeHandle n;
       /**
         * The advertise() function is how you tell ROS that you want to
         * publish on a given topic name. This invokes a call to the ROS
         * master node, which keeps a registry of who is publishing and who
         * is subscribing. After this advertise() call is made, the master
         * node will notify anyone who is trying to subscribe to this topic name,
         * and they will in turn negotiate a peer-to-peer connection with this
         * node.  advertise() returns a Publisher object which allows you to
         * publish messages on that topic through a call to publish().  Once
         * all copies of the returned Publisher object are destroyed, the topic
         * will be automatically unadvertised.
         *
         * The second parameter to advertise() is the size of the message queue
         * used for publishing messages.  If messages are published more quickly
         * than we can send them, the number here specifies how many messages to
         * buffer up before throwing some away.
         */
       ros::ServicesServer service=n.advertiseSerivice("add_two_ints",add); //the service is created and advertised over ROS
       ros::spin();
       return 0;
   }
   ```

   - client code

   ```c++
   #include "ros/ros.h"
   #include "beginner_tutorials/AddTwoInts.h"
   #include <cstdlib>
   int main(){
       ros::init(argc,argv,"add_two_ints_client");
       if(argc!=3){
           ROS_INFO("usage: add_two_ints_client X Y");
       }
       ros::NodeHandle n;
       //ros::Subscriber sub = n.subscribe("chatter", 1000, chatterCallback);
       ros::ServiceClient client=n.serviceClient<beginner_tutorial::AddTwoInts>("add_two_ints");  //equal to the name of servernode
       beginner_tutorials::AddTwoInts srv;
       srv.request.a=atoll(argv[1]);
       srv.request.b=atoll(argb[2]);
       if(client.call(srv)){  //call the service
           ROS_INFO("sum:%d",(long int)srv.response.sum);
       }
       else{
           ROS_ERROR("failed to call service and_two_ints");
           return 1;
       }
       return 0;
   }
   ```

   CMakeLists.txt   cd ~/catkin_ws  catkin_make

   ```cmake
   add_executable(add_two_ints_server src/add_two_ints_server.cpp)
   target_link_libraries(add_two_ints_server ${catkin_LIBRARIES})
   add_dependencies(add_two_ints_server_beginner_tutorials_gencpp)
   add_executable(add_two_ints_client src/add_two_ints_client.cpp)
   target_link_libraries(add_two_ints_client ${catkin_LIBRARIES})
   add_dependencies(add_two_ints_client beginner_tutorals_gencpp)
   ```

###### BackupData

- using rosbag to record the published data

  ```shell
  mkdir ~/bagfiles
  cd ~/bagfiles
  rosbag record -a
  
  rosbag infor <your bagfile>  #checks the contents of the bag file without plying it back
  rosbag play -r n <your bagfile>  #from the nth line to play 
  ```


###### roswtf   

```shell
#examines your system to try and find problems
roscd rosmaster
roswtf
```

#### Section two

###### Creating ROS package

```shell
mkdir -p src/foobar
cd src/foobar
vim package.xml
rospack find foobar
vim CMakeLists.xml
```

​	the package.xml file

```xml
<package format="2">
  <name>foobar</name>
  <version>1.2.4</version>
  <description>
  This package provides foo capability.
  </description>
  <maintainer email="foobar@foo.bar.willowgarage.com">PR-foobar</maintainer>
  <license>BSD</license>
  <buildtool_depend>catkin</buildtool_depend>
  <build_depend>roscpp</build_depend>
  <build_depend>std_msgs</build_depend>
  <exec_depend>roscpp</exec_depend>
  <exec_depend>std_msgs</exec_depend>
</package>
```

​	CMakeLists.xml

```cmake
cmake_mininum_required(VERSION 2.8.3)
project(foobar)
find_package(catkin REQUIED roscpp std_msgs)
catkin_package()
```

###### SystemDependency

```shell
rosdep install [package] #download the system dependencies
#error installaction has not been init
sudo rosdep init
rosdep update
rosdep resolve my_dependency_name #to see if the dependency is resolved
```

###### ROS across multiple machine

ros is designed with distributed computing in mind ,a well-written node makes no assumptions about where in the network it runs ,allowing computation to be relocated at run-time to match the available resources

- only need one master
- all nodes must be configured to use the same master,via ROS_MASTER_URI
- there must  be complete bi-direction connectivity between all pairs of machine ,on all ports
- each machine must advertise itself by a name that all other machines can resolve.

```shell
#start the master
roscore
#start the listener
export POS_MATER_URI=http://mastername:port
rosrun rospy_tutorials listener.py
#start the listener
export ROS_MASTER_URI=http://matername:port
rosrun rospy_tutorials talker.py
#varaition connectivity
#rostopic
```

http://wiki.ros.org/ROS/Tutorials  4,5 节内容学习

http://moorerobots.com/blog/post/1  偏向视频讲解

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/rosrelative/  

