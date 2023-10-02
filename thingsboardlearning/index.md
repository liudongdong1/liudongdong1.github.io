# ThingsBoardLearning


> ThingsBoard is an open-source IoT platform that enables `rapid development, management, and scaling `of IoT projects. Our goal is to provide the out-of-the-box IoT cloud or on-premises solution that will enable server-side infrastructure for your IoT applications.
>
> - **scalable**: the horizontally scalable platform, built using leading open-source technologies.
> - **fault-tolerant**: no single-point-of-failure, every node in the cluster is identical.
> - **robust and efficient**: a single server node can handle tens or even hundreds of thousands of devices, depending on the use-case. ThingsBoard cluster can handle millions of devices.
> - **customizable**: adding new functionality is easy with customizable widgets and rule engine nodes.
> - **durable**: never lose your data.

### 0. Architecture

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211104085205535.png)

- ThingsBoard transports: provides [MQTT](https://thingsboard.io/docs/reference/mqtt-api/), [HTTP](https://thingsboard.io/docs/reference/http-api/), [CoAP](https://thingsboard.io/docs/reference/coap-api/) and [LwM2M](https://thingsboard.io/docs/reference/lwm2m-api/) based APIs 
- ThingsBoard Core: handling [REST API](https://thingsboard.io/docs/reference/rest-api/) calls and WebSocket [subscriptions](https://thingsboard.io/docs/user-guide/telemetry/#websocket-api). storing up to date information about active device sessions and monitoring device [connectivity state](https://thingsboard.io/docs/user-guide/device-connectivity-status/). 
- ThingsBoard Rule Engine
- ThingsBoard Web UI

#### .1. Monolithic architecture

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211104144533429.png)

#### .2. Microservice architecture

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211104144631072.png)

### 1. Device Connection

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211104085828570.png)

### 2. Key Concepts

#### .1. Entities and relations

- **[Tenants](https://thingsboard.io/docs/user-guide/ui/tenants/)** - you can treat the tenant as a separate `business-entity`: it’s an individual or an organization who `owns or produce devices and assets`; Tenant may have multiple tenant administrator users and millions of customers, devices and assets;
- **[Customers](https://thingsboard.io/docs/user-guide/ui/customers/)** - the customer is also a separate business-entity: individual or organization who `purchase or uses tenant devices and/or assets`; Customer may have multiple users and millions of devices and/or assets;
- **[Users](https://thingsboard.io/docs/user-guide/ui/users/)** - users are able to` browse dashboards and manage entities`;
- **[Devices](https://thingsboard.io/docs/user-guide/ui/devices/)** - basic IoT entities that may `produce telemetry data and handle RPC commands`. For example,` sensors, actuators, switches`;
- **[Assets](https://thingsboard.io/docs/user-guide/ui/assets/)** - abstract IoT entities that may be related to other devices and assets. For example` factory, field, vehicle`;
- **[Entity Views](https://thingsboard.io/docs/user-guide/entity-views/)** - useful if you like to share only part of device or asset data to the customers;
- **[Alarms](https://thingsboard.io/docs/user-guide/alarms/)** - `events` that` identify issues` with your assets, devices, or other entities;
- **[Dashboards](https://thingsboard.io/docs/user-guide/dashboards/)** - visualization of your IoT data and ability to control particular devices through the user interface;
- **Rule Node** - `processing units for incoming messages`, entity lifecycle events, etc;
- **Rule Chain** - defines the` flow of the processing` in the [Rule Engine](https://thingsboard.io/docs/user-guide/rule-engine-2-0/re-getting-started/). May contain many rule nodes and links to other rule chains;

Each entity supports:

- **[Attributes](https://thingsboard.io/docs/user-guide/attributes/)** -` static and semi-static key-value pairs associated with entities`. For example serial number, model, firmware version;
- **[Time-series data](https://thingsboard.io/docs/user-guide/telemetry/)** - `time-series data points` available for storage, querying and visualization. For example temperature, humidity, battery level;
- **[Relations](https://thingsboard.io/docs/user-guide/entities-and-relations/#relations)** - `directed connections to other entities`. For example contains, manages, owns, produces.

![Relation](https://gitee.com/github-25970295/blogpictureV2/raw/master/entities-and-relations.svg)

#### .2. Tenant Profiles

> a System Administrator is able to configure common settings for multiple tenants using Tenant Profiles. Each Tenant has the one and only profile at a single point in time.

- **entity limits**:  to configure a maximum number of entities that each Tenant is able to create.
- **API limits:** to configure a maximum number of messages, API calls, etc., per month that each Tenant would like to perform.
  - **Transport Messages** means any message that your device sends to the server. This may be telemetry, attribute update, RPC call, etc.
  - **Transport Data Points** means a number of the Key-Value pairs that your telemetry or attribute messages contain. 

#### .3. Device Profiles

> the Tenant administrator is able to configure common settings for multiple devices using Device Profiles. Each Device has one and only profile at a single point in time.

- the [Root Rule Chain](https://thingsboard.io/docs/user-guide/rule-engine-2-0/overview/#rule-chain) processes all incoming messages and events for any device.
- **Queue Name**:
  -  the [Main](https://thingsboard.io/docs/user-guide/rule-engine-2-0/overview/#rule-engine-queue) queue will be used to store all incoming messages and events from any device. The transport layer will submit messages to this queue and Rule Engine will poll the queue for new messages.
  - Separation of the queues also allows you to customize different [submit](https://thingsboard.io/docs/user-guide/rule-engine-2-0/overview/#queue-submit-strategy) and [processing](https://thingsboard.io/docs/user-guide/rule-engine-2-0/overview/#queue-processing-strategy) strategies. `configure it in the thingsboard.yml`
- Transport configuration:
  - default transport type: the platform’s default [MQTT](https://thingsboard.io/docs/reference/mqtt-api/), [HTTP](https://thingsboard.io/docs/reference/http-api/), [CoAP](https://thingsboard.io/docs/reference/mqtt-api/) and [LwM2M](https://thingsboard.io/docs/reference/lwm2m-api/) APIs
  - MQTT transport type: specify custom MQTT topics filters for time-series data and attribute updates that correspond to the [telemetry upload API](https://thingsboard.io/docs/reference/mqtt-api/#telemetry-upload-api) and [attribute update API](https://thingsboard.io/docs/reference/mqtt-api/#publish-attribute-update-to-the-server), respectively
  - CoAP default:  supports basic [CoAP API](https://thingsboard.io/docs/reference/coap-api/) same as for [Default transport type](https://thingsboard.io/docs/user-guide/device-profiles/#default-transport-type). However, it is also possible to send data via [Protocol Buffers](https://developers.google.com/protocol-buffers) by changing the parameter CoAP device payload to Protobuf.
  - CoAP device type: Efento NB-IoT: 
- Alarm rules: 
  - **Alarm Type** - a type of Alarm. Alarm type must be unique within the device profile alarm rules;
  - **Clear condition** - defines `criteria` when the Alarm will be cleared;
  - **Advanced settings** - defines alarm` propagation to related assets, customers, tenant, or other entities`.
  - **Create Conditions** - defines the `criteria` when the Alarm will be created/updated.  `Severiy, Key Filters, Condition Type, Schedule, Details`

#### .4. Attributes

- **server-side attributes:** 

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211104093620186.png)

- **shared attributes**

>The device firmware/application may request the value of the shared attribute(s) or subscribe to the updates of the attribute(s).

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211104093741430.png)

- **client-side attributes**

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211104094318380.png)

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211104094514206.png)

#### .5. Time-series Data

- **Collect** data from devices using various [protocols and integrations](https://thingsboard.io/docs/getting-started-guides/connectivity/);
- **Store** time series data in SQL (PostgreSQL) or NoSQL (Cassandra or Timescale) databases;
- **Query** the latest time series data values or all data within the specified time range with flexible aggregation;
- **Subscribe** to data updates using [WebSockets](https://thingsboard.io/docs/user-guide/telemetry/#websocket-api) for visualization or real-time analytics;
- **Visualize** time series data using configurable and highly customizable widgets and [dashboards](https://thingsboard.io/docs/user-guide/dashboards/);
- **Filter and analyze** data using flexible [Rule Engine](https://thingsboard.io/docs/user-guide/rule-engine-2-0/re-getting-started/);
- **Generate [alarms](https://thingsboard.io/docs/user-guide/alarms/)** based on collected data;
- **Forward** data to external systems using [External Rule Nodes](https://thingsboard.io/docs/user-guide/rule-engine-2-0/external-nodes/) (e.g. Kafka or RabbitMQ Rule Nodes).

> ThingsBoard internally treats time-series data as timestamped key-value pairs. We call single timestamped key-value pair a **data point**.

#### .6. Remote Command to devices

- **Client-side RPC**

> send the request **from the device to the platform** and get the response back to the device.
>
> - Irrigation system gets the weather forecast from the online service through the platform.
> - Constrained device without system clock requests the current timestamp from the platform.
> - Access Control card reader sends the request to third-party security system to make a decision to open the door and log access.

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211104095739006.png)

- **Server-side RPC**

>send the request **from the platform to the device** and optionally get the response back to the platform.
>
>- remote control: reboot, turn the engine on/off, change state of the gpio/actuators, change configuration parameters, etc.

![One-way server-side RPC](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211104101226828.png)

![Two-way server-side RPC](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211104101248198.png)

### 3. Rule Engine

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211104101611044.png)

### 4. Gateway

![](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20211104144953580.png)

- [**MQTT** connector](https://thingsboard.io/docs/iot-gateway/config/mqtt/) to control, configure and collect data from IoT devices that are connected to external MQTT brokers using existing protocols.
- [**OPC-UA** connector](https://thingsboard.io/docs/iot-gateway/config/opc-ua/) to collect data from IoT devices that are connected to OPC-UA servers.
- [**Modbus** connector](https://thingsboard.io/docs/iot-gateway/config/modbus/) to collect data from IoT devices that are connected through Modbus protocol.
- [**BLE** connector](https://thingsboard.io/docs/iot-gateway/config/ble/) to collect data from IoT devices that are connected using Bluetooth Low Energy.
- [**Request** connector](https://thingsboard.io/docs/iot-gateway/config/request/) to collect data from IoT devices that are have HTTP(S) API endpoints.
- [**CAN** connector](https://thingsboard.io/docs/iot-gateway/config/can/) to collect data from IoT devices that are connected through CAN protocol.
- [**BACnet** connector](https://thingsboard.io/docs/iot-gateway/config/bacnet/) to collect data from IoT devices that are connected throughBACnet protocol.
- [**ODBC** connector](https://thingsboard.io/docs/iot-gateway/config/odbc/) to collect data from ODBC databases.
- [**REST** connector](https://thingsboard.io/docs/iot-gateway/config/rest/) to create endpoints and collect data from incoming HTTP requests.
- [**SNMP** connector](https://thingsboard.io/docs/iot-gateway/config/rest/) to collect data from SNMP managers.
- [**FTP** connector](https://thingsboard.io/docs/iot-gateway/config/ftp/) to collect data from FTP server
- [**Custom** connector](https://thingsboard.io/docs/iot-gateway/custom/) to collect data from IoT devices that are connected by different protocols. (You can create your own connector for the requires protocol).
- **Persistence** of collected data to guarantee data delivery in case of network or hardware failures.
- **Automatic reconnect** to ThingsBoard cluster.
- Simple yet powerful **mapping** of incoming data and messages **to unified format**.

---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/thingsboardlearning/  

