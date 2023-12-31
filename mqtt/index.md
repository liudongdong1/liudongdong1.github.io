# MQTT


> ThingsBoard is an open-source IoT platform that enables rapid development, management and scaling of IoT projects. Our goal is to provide the out-of-the-box IoT cloud or on-premises solution that will enable server-side infrastructure for your IoT applications.

## 1. [ThingsBoard](https://github.com/thingsboard/thingsboard-gateway)

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200829204245.png)

### 1.1. ConnectProtocol

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200829204707.png)

## 2. Features

### 2.1. Entities&&Relations

【Entities】

- **Tenants:** individual or organization who owns or produce devices and assets; have multiple tenant administrator users and customers.
- **Customers:** who purchase or use tenant devices or assets; have multiple users and devices or assets;
- **Users:** browse dashboards and manage entities;
- **Devices:** basic IoT entities,like sensors, actuators, switches;
- **Assets:** abstract IoT entities may be related to other devices and assets,like factory, field, vehicle;
- **Alarms:** events
- **Dashboards:** visualization of data and user control interface;
- **Rule Chain:** logic unit of related Rule Nodes; 

> each entity supports:
>
> - **Attributes:** like serial number, model, firmware version
> - **Telemetry Data:** time-series data points available for storage,quering and visualization, like temperature,humidity
> - **Relations:** contains, manages, owns, produces;

[Examples:](https://thingsboard.io/docs/user-guide/entities-and-relations/)

### 2.2.  working with device attribute

- **server-side:** attributes are reported and mange by the server-side application.
- **client-side:** attributes are reported and managed by the device application;
- **shared:**  attributes are reported and managed by the server-side application, visible to the device application;

### 2.3. working with telemetry data

- **collect** data from devices using MQTT, CoAP or HTTP protocols.
- **store** timeseries data in Cassandra (efficient, scalable and fault-tolerant NoSQL database).
- **query** latest timeseries data values or all data within the specified time interval.
- **subscribe** to data updates using websockets (for visualization or real-time analytics).
- **visualize** timeseries data using configurable and highly customizable widgets and dashboards.
- **filter and analyze** data using flexible Rule Engine (/docs/user-guide/rule-engine/).
- **generate alarms** based on collected data.
- **forward** data to external systems using Rule Nodes (e.g. Kafka or RabbitMQ Rule Nodes).

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/20200829212029.png)

【**Event**】

- **Connect event** - triggered when device connects to ThingsBoard. Meaningful in case of session based transports like MQTT. Will be also triggered for HTTP transport, but in this case it will be triggered on each HTTP request;
- **Disconnect event** - triggered when device disconnects from ThingsBoard. Meaningful in case of session based transports like MQTT. Will be also triggered for HTTP transport, but in this case it will be triggered on each HTTP request;
- **Activity event** - triggered when device pushes telemetry, attribute update or rpc command;
- **Inactivity event** - triggered when device was inactive for a certain period of time. Please note that this event may be triggered even without disconnect event from device. Typically means that there was no activity events triggered for a while;

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/mqtt/  

