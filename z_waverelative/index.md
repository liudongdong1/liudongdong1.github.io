# Z_WaveRelative


## 1. Introduce

- a wireless home automation protocol;
- an alliance of companies;
- all z-ware hardware uses sigma chips;
- z-wave has been a closed protocol;
  - devices and protocol managed by sigma and z-wave alliance;
  - certified devices undergo a certification scheme to ensure compatibility;
  - open source projects needed to rely on reverse engineering the protocol;
- approx 600 companies and 2200 certified devices;
- introduce in 2015 and backward compatible with the original standard;
- improved devices with lower battery comsumption, coupled with improved protocol functionality;
  - data rate 100kb/s;
  - lower power consumption for better battery life;

## 2. z-wave protocol features

- two way communications:
  - guarantees delivery of data or notification of failure;
- mesh networking:
  - extends the network outside the immediate range of the controller;
- immediate status updates:
  - reduced latency over a polled system;
- large number of devices on  a network
  - supports 232 physical devices;
- high security option

## 3. routing

- z-wave uses a  source routing mesh network
  - the sender( controller ) is responsible for defining the routes;
  - up to four routers can be traversed between the source and destination;
- the controller uses "explorer frames" to derive a route to the destination;

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/z_waverelative/  

