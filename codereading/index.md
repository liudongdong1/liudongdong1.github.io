# CodeReading


##### 1. Connect

```java
  public void connect(String address, int port, boolean useTLS) throws OctaneSdkException {
        LLRPIoHandlerAdapterImpl llrpio = new LLRPIoHandlerAdapterImpl();
        llrpio.setKeepAliveAck(false);
        llrpio.setKeepAliveForward(true);
        if (this.readerConnection == null) {
            this.readerConnection = new LLRPConnector(this, address, llrpio);
        }

        llrpio.setConnection(this.readerConnection);
        ((LLRPConnector)this.readerConnection).setPort(port);

        try {
            ((LLRPConnector)this.readerConnection).connect((long)this.connectTimeout, useTLS);
        } catch (LLRPConnectionAttemptFailedException var6) {
            this.readerConnection = null;
            throw new OctaneSdkException("Error connecting to the reader (" + address + ") : " + var6.getMessage());
        } catch (RuntimeIoException var7) {
            this.readerConnection = null;
            throw new OctaneSdkException("Error connecting to the reader (" + address + ") : " + var7.getMessage());
        }
}
```

##### 2. GetFeatures

```java
GET_READER_CAPABILITIES_RESPONSE getReaderCapabilities() throws OctaneSdkException {
        GET_READER_CAPABILITIES get = new GET_READER_CAPABILITIES();
        GetReaderCapabilitiesRequestedData data = new GetReaderCapabilitiesRequestedData(0);
        get.setRequestedData(data);
        get.setMessageID(this.getUniqueMessageID());

        try {
            LLRPMessage response = this.readerConnection.transact(get, (long)this.messageTimeout);
            GET_READER_CAPABILITIES_RESPONSE gresp = (GET_READER_CAPABILITIES_RESPONSE)response;
            StatusCode status = gresp.getLLRPStatus().getStatusCode();
            if (status.equals(new StatusCode("M_Success"))) {
                GeneralDeviceCapabilities dev_cap = gresp.getGeneralDeviceCapabilities();
                if (dev_cap != null && !dev_cap.getDeviceManufacturerName().equals(new UnsignedInteger(25882))) {
                    throw new OctaneSdkException("OctaneSdk must use Impinj model Reader, not " + dev_cap.getDeviceManufacturerName().toString());
                } else {
                    return gresp;
                }
            } else {
                throw new OctaneSdkException("OctaneSdk exception. [status]=[" + status.toXMLString() + "]");
            }
        } catch (Exception var7) {
            throw new OctaneSdkException("OctaneSdk exception: " + var7.getMessage());
        }
    }
```

##### 3. startRoSpec

```java
  void startRoSpec() throws OctaneSdkException {
        START_ROSPEC msg = new START_ROSPEC();
        UnsignedInteger roSpecId = new UnsignedInteger(14150);
        msg.setMessageID(this.getUniqueMessageID());
        msg.setROSpecID(roSpecId);
        ERROR_MESSAGE msgErr = null;
        LLRPMessage rsp = null;

        try {
            rsp = this.readerConnection.transact(msg, (long)this.messageTimeout);
        } catch (TimeoutException var9) {
            throw new OctaneSdkException("OctaneSdk timeout exception: " + var9.getMessage());
        }

        String msgType = "START_ROSPEC";
        this.checkForNullReply(msgType, rsp, (ERROR_MESSAGE)msgErr);
        StatusCode status = ((START_ROSPEC_RESPONSE)rsp).getLLRPStatus().getStatusCode();
        if (!status.equals(new StatusCode("M_Success"))) {
            try {
                String llrMessage = rsp.toXMLString();
                throw new OctaneSdkException("OctaneSdk exception: " + llrMessage);
            } catch (InvalidLLRPMessageException var8) { 
                throw new OctaneSdkException("OctaneSdk exception: " + var8.getMessage());
            }
        }
    }
```

##### 4. FilterOp

```java
C1G2Filter[] getC1G2Filters(FilterSettings filterSettings) throws OctaneSdkException {
        C1G2Filter[] filters = null;
        if (filterSettings.getMode() == TagFilterMode.UseTagSelectFilters) {
            List<TagSelectFilter> tagSelectFilterList = filterSettings.getTagSelectFilterList();
            int maxNumSelectFiltersPerQuery = this.readerCapabilities.getMaxNumSelectFiltersPerQuery();   //Todo 这里再哪设置
            int tagSelectFilterListSize = tagSelectFilterList.size();
            if (tagSelectFilterListSize > maxNumSelectFiltersPerQuery) {
                throw new OctaneSdkException("Error parsing tag select filter list. " + String.format("The tag select filter list has %d filters, ", tagSelectFilterListSize) + String.format("which is more than the maximum supported by the reader which is %d", maxNumSelectFiltersPerQuery));
            }

            filters = new C1G2Filter[tagSelectFilterListSize];

            for(int i = 0; i < tagSelectFilterList.size(); ++i) {
                TagSelectFilter tagSelectFilter = (TagSelectFilter)tagSelectFilterList.get(i);
                filters[i] = new C1G2Filter();
                filters[i].setT(new C1G2TruncateAction(1));
                C1G2TagInventoryMask c1g2TagInventoryMaskFilter = new C1G2TagInventoryMask();
                c1g2TagInventoryMaskFilter.setMB(this.twoBitFieldFromInt(tagSelectFilter.getMemoryBank().getValue()));
                UnsignedShort bitPointerFilter = new UnsignedShort(tagSelectFilter.getBitPointer());
                c1g2TagInventoryMaskFilter.setPointer(bitPointerFilter);
                c1g2TagInventoryMaskFilter.setTagMask(new BitArray_HEX(tagSelectFilter.getTagMask()));
                filters[i].setC1G2TagInventoryMask(c1g2TagInventoryMaskFilter);
                if (tagSelectFilter.getBitCount() > 0L) {
                    filters[i].getC1G2TagInventoryMask().setTagMask(this.truncateTagMask(filters[i].getC1G2TagInventoryMask().getTagMask(), (int)tagSelectFilter.getBitCount()));
                }

                TagFilterStateUnawareAction matchingAction = tagSelectFilter.getMatchingAction();
                TagFilterStateUnawareAction nonMatchingAction = tagSelectFilter.getNonMatchingAction();
                C1G2TagInventoryStateUnawareFilterAction c1g2TagInventoryStateUnawareFilterAction = new C1G2TagInventoryStateUnawareFilterAction();
                C1G2StateUnawareAction c1g2StateUnawareAction = new C1G2StateUnawareAction();
                c1g2StateUnawareAction.set(TagFilterStateUnawareAction.convertToC1G2StateUnawareAction(tagSelectFilter.getMatchingAction(), tagSelectFilter.getNonMatchingAction()));
                c1g2TagInventoryStateUnawareFilterAction.setAction(c1g2StateUnawareAction);
                filters[i].setC1G2TagInventoryStateUnawareFilterAction(c1g2TagInventoryStateUnawareFilterAction);
            }
        } else {
            C1G2TagInventoryMask c1g2TagInventoryMask;
            UnsignedShort bitPointerFilter;
            C1G2StateUnawareAction c1g2tateUnawareAction;
            if (filterSettings.getMode() != TagFilterMode.Filter1AndFilter2 && filterSettings.getMode() != TagFilterMode.Filter1OrFilter2) {
                if (filterSettings.getMode() != TagFilterMode.None) {
                    filters = new C1G2Filter[]{new C1G2Filter()};
                    c1g2TagInventoryMask = new C1G2TagInventoryMask();
                    C1G2TagInventoryStateUnawareFilterAction c1g2TagInventoryStateUnawareFilterAction = new C1G2TagInventoryStateUnawareFilterAction();
                    filters[0].setC1G2TagInventoryMask(c1g2TagInventoryMask);
                    filters[0].setC1G2TagInventoryStateUnawareFilterAction(c1g2TagInventoryStateUnawareFilterAction);
                    TagFilter enabledFilter;
                    if (filterSettings.getMode() == TagFilterMode.OnlyFilter1) {
                        enabledFilter = filterSettings.getTagFilter1();
                    } else {
                        enabledFilter = filterSettings.getTagFilter2();
                    }

                    c1g2TagInventoryMask.setMB(this.twoBitFieldFromInt(enabledFilter.getMemoryBank().getValue()));
                    bitPointerFilter = new UnsignedShort(enabledFilter.getBitPointer());
                    c1g2TagInventoryMask.setPointer(bitPointerFilter);
                    BitArray_HEX b = new BitArray_HEX(enabledFilter.getTagMask());
                    c1g2TagInventoryMask.setTagMask(b);
                    filters[0].setC1G2TagInventoryMask(c1g2TagInventoryMask);
                    filters[0].setT(new C1G2TruncateAction(1));
                    if (enabledFilter.getBitCount() > 0L) {
                        filters[0].getC1G2TagInventoryMask().setTagMask(this.truncateTagMask(filters[0].getC1G2TagInventoryMask().getTagMask(), (int)enabledFilter.getBitCount()));
                    }

                    if (enabledFilter.getFilterOp() == TagFilterOp.Match) {
                        c1g2tateUnawareAction = new C1G2StateUnawareAction();
                        c1g2tateUnawareAction.set(0);
                        c1g2TagInventoryStateUnawareFilterAction.setAction(c1g2tateUnawareAction);
                        filters[0].setC1G2TagInventoryStateUnawareFilterAction(c1g2TagInventoryStateUnawareFilterAction);
                    } else {
                        c1g2tateUnawareAction = new C1G2StateUnawareAction();
                        c1g2tateUnawareAction.set(4);
                        c1g2TagInventoryStateUnawareFilterAction.setAction(c1g2tateUnawareAction);
                        filters[0].setC1G2TagInventoryStateUnawareFilterAction(c1g2TagInventoryStateUnawareFilterAction);
                    }
                }
            } else {
                filters = new C1G2Filter[]{new C1G2Filter(), new C1G2Filter()};
                c1g2TagInventoryMask = new C1G2TagInventoryMask();
                c1g2TagInventoryMask.setMB(this.twoBitFieldFromInt(filterSettings.getTagFilter1().getMemoryBank().getValue()));
                UnsignedShort bitPointerFilter1 = new UnsignedShort(filterSettings.getTagFilter1().getBitPointer());
                c1g2TagInventoryMask.setPointer(bitPointerFilter1);
                c1g2TagInventoryMask.setTagMask(new BitArray_HEX(filterSettings.getTagFilter1().getTagMask()));
                filters[0].setT(new C1G2TruncateAction(1));
                filters[0].setC1G2TagInventoryMask(c1g2TagInventoryMask);
                C1G2TagInventoryMask c1g2TagInventoryMaskFilter2 = new C1G2TagInventoryMask();
                c1g2TagInventoryMaskFilter2.setMB(this.twoBitFieldFromInt(filterSettings.getTagFilter2().getMemoryBank().getValue()));
                bitPointerFilter = new UnsignedShort(filterSettings.getTagFilter2().getBitPointer());
                c1g2TagInventoryMaskFilter2.setPointer(bitPointerFilter);
                c1g2TagInventoryMaskFilter2.setTagMask(new BitArray_HEX(filterSettings.getTagFilter2().getTagMask()));
                filters[1].setT(new C1G2TruncateAction(1));
                filters[1].setC1G2TagInventoryMask(c1g2TagInventoryMaskFilter2);
                if (filterSettings.getTagFilter1().getBitCount() > 0L) {
                    filters[0].getC1G2TagInventoryMask().setTagMask(this.truncateTagMask(filters[0].getC1G2TagInventoryMask().getTagMask(), (int)filterSettings.getTagFilter1().getBitCount()));
                }

                if (filterSettings.getTagFilter2().getBitCount() > 0L) {
                    filters[1].getC1G2TagInventoryMask().setTagMask(this.truncateTagMask(filters[1].getC1G2TagInventoryMask().getTagMask(), (int)filterSettings.getTagFilter2().getBitCount()));
                }

                C1G2TagInventoryStateUnawareFilterAction c1g2TagInventoryStateUnawareFilterAction = new C1G2TagInventoryStateUnawareFilterAction();
                if (filterSettings.getTagFilter1().getFilterOp() == TagFilterOp.Match) {
                    c1g2tateUnawareAction = new C1G2StateUnawareAction();
                    c1g2tateUnawareAction.set(0);
                    c1g2TagInventoryStateUnawareFilterAction.setAction(c1g2tateUnawareAction);
                    filters[0].setC1G2TagInventoryStateUnawareFilterAction(c1g2TagInventoryStateUnawareFilterAction);
                } else {
                    c1g2tateUnawareAction = new C1G2StateUnawareAction();
                    c1g2tateUnawareAction.set(4);
                    c1g2TagInventoryStateUnawareFilterAction.setAction(c1g2tateUnawareAction);
                    filters[0].setC1G2TagInventoryStateUnawareFilterAction(c1g2TagInventoryStateUnawareFilterAction);
                }

                C1G2TagInventoryStateUnawareFilterAction c1g2TagInventoryStateUnawareFilterActionFilter2 = new C1G2TagInventoryStateUnawareFilterAction();
                C1G2StateUnawareAction c1g2tateUnawareAction;
                if (filterSettings.getMode() == TagFilterMode.Filter1AndFilter2) {
                    if (filterSettings.getTagFilter2().getFilterOp() == TagFilterOp.Match) {
                        c1g2tateUnawareAction = new C1G2StateUnawareAction();
                        c1g2tateUnawareAction.set(2);
                        c1g2TagInventoryStateUnawareFilterActionFilter2.setAction(c1g2tateUnawareAction);
                        filters[1].setC1G2TagInventoryStateUnawareFilterAction(c1g2TagInventoryStateUnawareFilterActionFilter2);
                    } else {
                        c1g2tateUnawareAction = new C1G2StateUnawareAction();
                        c1g2tateUnawareAction.set(3);
                        c1g2TagInventoryStateUnawareFilterActionFilter2.setAction(c1g2tateUnawareAction);
                        filters[1].setC1G2TagInventoryStateUnawareFilterAction(c1g2TagInventoryStateUnawareFilterActionFilter2);
                    }
                } else if (filterSettings.getMode() == TagFilterMode.Filter1OrFilter2) {
                    if (filterSettings.getTagFilter2().getFilterOp() == TagFilterOp.Match) {
                        c1g2tateUnawareAction = new C1G2StateUnawareAction();
                        c1g2tateUnawareAction.set(1);
                        c1g2TagInventoryStateUnawareFilterActionFilter2.setAction(c1g2tateUnawareAction);
                        filters[1].setC1G2TagInventoryStateUnawareFilterAction(c1g2TagInventoryStateUnawareFilterActionFilter2);
                    } else {
                        c1g2tateUnawareAction = new C1G2StateUnawareAction();
                        c1g2tateUnawareAction.set(2);
                        c1g2TagInventoryStateUnawareFilterActionFilter2.setAction(c1g2tateUnawareAction);
                        filters[1].setC1G2TagInventoryStateUnawareFilterAction(c1g2TagInventoryStateUnawareFilterActionFilter2);
                    }
                }
            }
        }

        return filters;
    }
```

```java
if (enabledFilter.getBitCount() > 0L) {
                        filters[0].getC1G2TagInventoryMask().setTagMask(this.truncateTagMask(filters[0].getC1G2TagInventoryMask().getTagMask(), (int)enabledFilter.getBitCount()));
}
c1g2TagInventoryMaskFilter2.setTagMask(new BitArray_HEX(filterSettings.getTagFilter2().getTagMask()));

 BitArray_HEX truncateTagMask(BitArray_HEX mask, int len) throws OctaneSdkException {
        BitArray_HEX newMask = mask;
        if (len > mask.size()) {
            throw new OctaneSdkException("Error setting the tag mask. The value specified for BitCount is greater than the data provided.");
        } else if (mask.size() % 8 != 0) {
            throw new OctaneSdkException("Error setting the tag mask. The tag mask bit length must be divisible by 8. From the LLRP Spec: Integer numbers SHALL be encoded in network byte order with the most significant byte of the integer being sent first (big-Endian). Bit arrays are aligned to the most significant bit. Bit arrays are padded to an octet boundary. Pad bits SHALL be ignored by the Reader and Client.");
        } else {
            if (len < mask.size()) {
                newMask = new BitArray_HEX(len);

                for(int i = 0; i < len; ++i) {
                    if (mask.get(i).toBoolean()) {
                        newMask.set(i);
                    }
                }
            }

            return newMask;
        }
    }
```

```java
C1G2TagInventoryStateUnawareFilterAction c1g2TagInventoryStateUnawareFilterActionFilter2 = new C1G2TagInventoryStateUnawareFilterAction();
                C1G2StateUnawareAction c1g2tateUnawareAction;
                if (filterSettings.getMode() == TagFilterMode.Filter1AndFilter2) {
                    if (filterSettings.getTagFilter2().getFilterOp() == TagFilterOp.Match) {
                        c1g2tateUnawareAction = new C1G2StateUnawareAction();
                        c1g2tateUnawareAction.set(2);
                        c1g2TagInventoryStateUnawareFilterActionFilter2.setAction(c1g2tateUnawareAction);
                        filters[1].setC1G2TagInventoryStateUnawareFilterAction(c1g2TagInventoryStateUnawareFilterActionFilter2);
                    } else {
                        c1g2tateUnawareAction = new C1G2StateUnawareAction();
                        c1g2tateUnawareAction.set(3);
                        c1g2TagInventoryStateUnawareFilterActionFilter2.setAction(c1g2tateUnawareAction);
                        filters[1].setC1G2TagInventoryStateUnawareFilterAction(c1g2TagInventoryStateUnawareFilterActionFilter2);
                    }
                } else if (filterSettings.getMode() == TagFilterMode.Filter1OrFilter2) {
                    if (filterSettings.getTagFilter2().getFilterOp() == TagFilterOp.Match) {
                        c1g2tateUnawareAction = new C1G2StateUnawareAction();
                        c1g2tateUnawareAction.set(1);
                        c1g2TagInventoryStateUnawareFilterActionFilter2.setAction(c1g2tateUnawareAction);
                        filters[1].setC1G2TagInventoryStateUnawareFilterAction(c1g2TagInventoryStateUnawareFilterActionFilter2);
                    } else {
                        c1g2tateUnawareAction = new C1G2StateUnawareAction();
                        c1g2tateUnawareAction.set(2);
                        c1g2TagInventoryStateUnawareFilterActionFilter2.setAction(c1g2tateUnawareAction);
                        filters[1].setC1G2TagInventoryStateUnawareFilterAction(c1g2TagInventoryStateUnawareFilterActionFilter2);
                    }
                }
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/codereading/  

