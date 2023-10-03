# Android_RxAndroidOp


### 1. RxBLE

#### Scanner

```java
private void scanBleDevices() {
    scanDisposable = rxBleClient.scanBleDevices(
        new ScanSettings.Builder()
        .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
        .setCallbackType(ScanSettings.CALLBACK_TYPE_ALL_MATCHES)
        .build(),
        new ScanFilter.Builder()
        //                            .setDeviceAddress("B4:99:4C:34:DC:8B")
        // add custom filters if needed
        .build()
    )
        .observeOn(AndroidSchedulers.mainThread())
        .doFinally(this::dispose)
        .subscribe(resultsAdapter::addScanResult, this::onScanFailure);
}
```

#### Connection

```java
bleDevice = SampleApplication.getRxBleClient(this).getBleDevice(macAddress);  //获取设备
// How to listen for connection state changes
// Note: it is meant for UI updates only — one should not observeConnectionStateChanges() with BLE connection logic
stateDisposable = bleDevice.observeConnectionStateChanges()
    .observeOn(AndroidSchedulers.mainThread())
    .subscribe(
        rxBleConnection -> {
            // All GATT operations are done through the rxBleConnection.
        },
        throwable -> {
            // Handle an error here.
        }
    );
```

#### MTu

```java
public void onSetMtu() {
    final Disposable disposable = bleDevice.establishConnection(false)
        .flatMapSingle(rxBleConnection -> rxBleConnection.requestMtu(72))
        .take(1) // Disconnect automatically after discovery
        .observeOn(AndroidSchedulers.mainThread())
        .doFinally(this::updateUI)
        .subscribe(this::onMtuReceived, this::onConnectionFailure);
    mtuDisposable.add(disposable);
}
```

#### DiscoverService

```java
public void onConnectToggleClick() {
    final Disposable disposable = bleDevice.establishConnection(false)
        .flatMapSingle(RxBleConnection::discoverServices)
        .take(1) // Disconnect automatically after discovery
        .observeOn(AndroidSchedulers.mainThread())
        .doFinally(this::updateUI)
        .subscribe(adapter::swapScanResult, this::onConnectionFailure);
    servicesDisposable.add(disposable);
    updateUI();
}
```

```java
void swapScanResult(RxBleDeviceServices services) {
    data.clear();
    for (BluetoothGattService service : services.getBluetoothGattServices()) {
        // Add service
        data.add(new AdapterItem(AdapterItem.SERVICE, getServiceType(service), service.getUuid()));
        final List<BluetoothGattCharacteristic> characteristics = service.getCharacteristics();
        for (BluetoothGattCharacteristic characteristic : characteristics) {
            data.add(new AdapterItem(AdapterItem.CHARACTERISTIC, describeProperties(characteristic), characteristic.getUuid()));
        }
    }
    notifyDataSetChanged();
}
```

#### RSSiRead

```java
connectionDisposable = bleDevice.establishConnection(false)
    .doFinally(this::clearSubscription)
    .flatMap(rxBleConnection -> // Set desired interval.
             Observable.interval(2, SECONDS).flatMapSingle(sequence -> rxBleConnection.readRssi()))
    .observeOn(AndroidSchedulers.mainThread())
    .subscribe(this::updateRssi, this::onConnectionFailure);
```

```java
device.establishConnection(false)
    .flatMap(rxBleConnection -> rxBleConnection.readCharacteristic(characteristicUUID))
    .subscribe(
    characteristicValue -> {
        // Read characteristic value.
    },
    throwable -> {
        // Handle an error here.
    }
);
```

```java
device.establishConnection(false)
    .flatMap(rxBleConnection -> rxBleConnection.writeCharacteristic(characteristicUUID, bytesToWrite))
    .subscribe(
    characteristicValue -> {
        // Characteristic value confirmed.
    },
    throwable -> {
        // Handle an error here.
    }
);
```

```java
device.establishConnection(false)
    .flatMap(rxBleConnection -> Observable.combineLatest(
        rxBleConnection.readCharacteristic(firstUUID),
        rxBleConnection.readCharacteristic(secondUUID),
        YourModelCombiningTwoValues::new
    ))
    .subscribe(
    model -> {
        // Process your model.
    },
    throwable -> {
        // Handle an error here.
    }
);
```



#### LongWrite

```java
final RxBleClient rxBleClient = SampleApplication.getRxBleClient(this);
disposable = rxBleClient.getBleDevice(DUMMY_DEVICE_ADDRESS) // get our assumed device
    .establishConnection(false) // establish the connection
    .flatMap(rxBleConnection -> Observable.combineLatest(
        // after establishing the connection lets setup the notifications
        rxBleConnection.setupNotification(DEVICE_CALLBACK_0),
        rxBleConnection.setupNotification(DEVICE_CALLBACK_1),
        Pair::new
    ), (rxBleConnection, callbackObservablePair) -> { // after the setup lets start the long write
        Observable<byte[]> deviceCallback0 = callbackObservablePair.first;
        Observable<byte[]> deviceCallback1 = callbackObservablePair.second;
        return rxBleConnection.createNewLongWriteBuilder() // create a new long write builder
            .setBytes(bytesToWrite) // REQUIRED - set the bytes to write
            /*
                             * REQUIRED - To perform a write you need to specify to which characteristic you want to write. You can do it
                             * either by calling {@link LongWriteOperationBuilder#setCharacteristicUuid(UUID)} or
                             * {@link LongWriteOperationBuilder#setCharacteristic(BluetoothGattCharacteristic)}
                             */
            .setCharacteristicUuid(WRITE_CHARACTERISTIC) // set the UUID of the characteristic to write
            // .setCharacteristic( /* some BluetoothGattCharacteristic */ ) // alternative to setCharacteristicUuid()
            /*
                             * If you want to send batches with length other than default.
                             * Default value is 20 bytes if MTU was not negotiated. If the MTU was negotiated prior to the Long Write
                             * Operation execution then the batch size default is the new MTU.
                             */
            // .setMaxBatchSize( /* your batch size */ )
            /*
                              Inform the Long Write when we want to send the next batch of data. If not set the operation will try to write
                              the next batch of data as soon as the Android will call `BluetoothGattCallback.onCharacteristicWrite()` but
                              we want to postpone it until also DC0 and DC1 will emit.
                             */
            .setWriteOperationAckStrategy(new RxBleConnection.WriteOperationAckStrategy() {
                @Override
                public ObservableSource<Boolean> apply(Observable<Boolean> booleanObservable) {
                    return Observable.zip(
                        // so we zip three observables
                        deviceCallback0, // DEVICE_CALLBACK_0
                        deviceCallback1, // DEVICE_CALLBACK_1
                        booleanObservable, /* previous batch of data was sent - we do not care if value emitted from
                                            the booleanObservable is TRUE or FALSE. But the value will be TRUE unless the previously sent
                                            data batch was the final one */
                        (callback0, callback1, aBoolean) -> aBoolean // value of the returned Boolean is not important
                    );
                }
            })
            .build();
    })
    .flatMap(observable -> observable)
    .take(1) // after the successful write we are no longer interested in the connection so it will be released
    .subscribe(
    bytes -> {
        // react
    },
    throwable -> {
        // handle error
    }
);
}
```

### 2. Rx2AndroidNetworking

#### .1. 一个接口请求

```java
Rx2AndroidNetworking.get("http://c.m.163.com/nc/article/list/T1371543208049/0-20.html")
    .addHeaders("User-Agent","Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:0.9.4)")
    .build()
    .getObjectObservable(CBAbean.class)   // 解析对应类
    .subscribeOn(Schedulers.io())
    .observeOn(AndroidSchedulers.mainThread())
    .subscribe(new Consumer<CBAbean>() {
        @Override
        public void accept(CBAbean cbAbean) throws Exception {

            List<CBAbean.T1371543208049Bean> beans = cbAbean.getT1371543208049();
            list.addAll(beans);
            myAdapter.notifyDataSetChanged();

        }
    }, new Consumer<Throwable>() {
        @Override
        public void accept(Throwable throwable) throws Exception {
            Log.d(TAG, "accept: "+throwable.getMessage());
        }
    });     
```

#### .2. 俩个接口请求

```java
Rx2AndroidNetworking
    .get("http://www.tngou.net/api/food/list")
    .addQueryParameter("rows", 1 + "")
    .build()
    .getObjectObservable(FoodList.class) // 发起获取食品列表的请求，并解析到FootList
    .subscribeOn(Schedulers.io())        // 在io线程进行网络请求
    .observeOn(AndroidSchedulers.mainThread()) // 在主线程处理获取食品列表的请求结果
    .doOnNext(new Consumer<FoodList>() {
        @Override
        public void accept(@NonNull FoodList foodList) throws Exception {
            // 先根据获取食品列表的响应结果做一些操作
            Log.e(TAG, "accept: doOnNext :" + foodList.toString());
            mRxOperatorsText.append("accept: doOnNext :" + foodList.toString()+"\n");
        }
    })
    .observeOn(Schedulers.io()) // 回到 io 线程去处理获取食品详情的请求
    .flatMap(new Function<FoodList, ObservableSource<FoodDetail>>() {  // flatMap 作用
        @Override
        public ObservableSource<FoodDetail> apply(@NonNull FoodList foodList) throws Exception {
            if (foodList != null && foodList.getTngou() != null && foodList.getTngou().size() > 0) {
                return Rx2AndroidNetworking.post("http://www.tngou.net/api/food/show")
                    .addBodyParameter("id", foodList.getTngou().get(0).getId() + "")
                    .build()
                    .getObjectObservable(FoodDetail.class);
            }
            return null;

        }
    })
    .observeOn(AndroidSchedulers.mainThread())
    .subscribe(new Consumer<FoodDetail>() {
        @Override
        public void accept(@NonNull FoodDetail foodDetail) throws Exception {
            Log.e(TAG, "accept: success ：" + foodDetail.toString());
            mRxOperatorsText.append("accept: success ：" + foodDetail.toString()+"\n");
        }
    }, new Consumer<Throwable>() {
        @Override
        public void accept(@NonNull Throwable throwable) throws Exception {
            Log.e(TAG, "accept: error :" + throwable.getMessage());
            mRxOperatorsText.append("accept: error :" + throwable.getMessage()+"\n");
        }
    });

}
```

### 3. Rxjava+ OkHttp3

- https://juejin.cn/post/6844903495426899981#heading-2


---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/android_rxandroidop/  

