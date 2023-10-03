# Pytorch_LearningRate


> a. 有序调整：等间隔调整(Step)，按需调整学习率(MultiStep)，指数衰减调整(Exponential)和 余弦退火CosineAnnealing。
> b. 自适应调整：自适应调整学习率 ReduceLROnPlateau。
> c. 自定义调整：自定义调整学习率 LambdaLR。

```python
#得到当前学习率
lr = next(iter(optimizer.param_groups))['lr'] 
#multiple learning rates for different layers.
all_lr = []
for param_group in optimizer.param_groups:
    all_lr.append(param_group['lr'])
    
 #学习率衰减
#Reduce learning rate when validation accuarcy plateau.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)
for t in range(0, 80):
    train(...); val(...)
    scheduler.step(val_acc)
#Cosine annealing learning rate.    
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
#Reduce learning rate by 10 at given epochs.
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
for t in range(0, 80):
    scheduler.step()    
    train(...); val(...)
#Learning rate warmup by 10 epochs.
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: t / 10)
for t in range(0, 10):
    scheduler.step()
    train(...); val(...)
```

#### 1. 针对不同的层

```python
model = torchvision.models.resnet101(pretrained=True)
large_lr_layers = list(map(id,model.fc.parameters()))
small_lr_layers = filter(lambda p:id(p) not in large_lr_layers,model.parameters())
optimizer = torch.optim.SGD([
            {"params":large_lr_layers},
            {"params":small_lr_layers,"lr":1e-4}
            ],lr = 1e-2,momenum=0.9)
```

#### 2. 等间隔调整学习率 StepLR

```python
torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
```

> - `step_size(int)`- 学习率`下降间隔数`，若为 30，则会在 30、 60、 90…个 step 时，将学习率调整为 `lr*gamma`。
> - gamma(float)- 学习率调整倍数，默认为 0.1 倍，即下降 10 倍。
> - last_epoch(int)- 上一个 epoch 数，这个变量用来`指示学习率是否需要调整`。当last_epoch 符合设定的间隔时，就会对学习率进行调整。当为-1 时，学习率设置为初始值。
>
> 调整倍数为 gamma 倍，调整间隔为 step_size。间隔单位是step。需要注意的是， step 通常是指 epoch，不要弄成 iteration 了。

#### 3. 按需调整学习率 MultiStepLR

```python
torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
```

> - milestones(list)- 一个 list，`每一个元素代表何时调整学习率`，` list 元素必须是递增的`。如 milestones=[30,80,120]
> - gamma(float)- 学习率调整倍数，默认为 0.1 倍，即下降 10 倍。
>
> 按设定的间隔调整学习率。这个方法适合后期调试使用，观察 loss 曲线，为每个实验定制学习率调整时机。

#### 4. 指数衰减调整学习率 ExponentialLR

```python
torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
```

> gamma- 学习率调整倍数的底，指数为 epoch，即 gamma**epoch

#### 5. 余弦退火调整学习率 CosineAnnealingLR

```python
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
```

> - T_max(int)- 一次学习率周期的迭代次数，即` T_max 个 epoch 之后重新设置学习率`。
> - eta_min(float)- 最小学习率，即在一个周期中，`学习率最小会下降到 eta_min`，默认值为 0。
>
> 以余弦函数为周期，并在每个周期最大值时重新设置学习率。以初始学习率为最大学习率，以 2 ∗ T m a x 2*Tmax2∗Tmax 为周期，在一个周期内先下降，后上升。

```python
epochs = 60
optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=1e-4) 
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max = (epochs // 9) + 1)
for epoch in range(epochs):
    scheduler.step(epoch)
```

#### 6. 自适应调整学习率 ReduceLROnPlateau

```python
torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
```

> - mode(str)- 模式选择，有 min 和 max 两种模式， `min 表示当指标不再降低(如监测loss)`，` max 表示当指标不再升高(如监测 accuracy)`。
> - factor(float)- 学习率调整倍数(等同于其它方法的 gamma)，即学习率更新为 `lr = lr * factor`
> - patience(int)- `忍受该指标多少个 step 不变化，当忍无可忍时，调整学习率`。
> - verbose(bool)- `是否打印学习率信息`， print(‘Epoch {:5d}: reducing learning rate of group {} to {:.4e}.’.format(epoch, i, new_lr))
> - threshold_mode(str)- 选择判断指标是否达最优的模式，有两种模式， rel 和 abs。
>   当 threshold_mode == rel，并且 mode == max 时， dynamic_threshold = best * ( 1 +threshold )；
>   当 threshold_mode == rel，并且 mode == min 时， dynamic_threshold = best * ( 1 -threshold )；
>   当 threshold_mode == abs，并且 mode== max 时， dynamic_threshold = best + threshold ；
>   当 threshold_mode == rel，并且 mode == max 时， dynamic_threshold = best - threshold；
> - threshold(float)- 配合 threshold_mode 使用。
> - `cooldown(int)- “冷却时间“`，当调整学习率之后，让学习率调整策略冷静一下，让模型再训练一段时间，再重启监测模式。
> - min_lr(float or list)- `学习率下限`，可为 float，或者 list，当有多个参数组时，可用 list 进行设置。
> - eps(float)- `学习率衰减的最小值`，当学习率变化小于 eps 时，则不调整学习率。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'max',verbose=1,patience=3)
for epoch in range(10):
    train(...)
    val_acc = validate(...)
    # 降低学习率需要在给出 val_acc 之后
    scheduler.step(val_acc)
```

#### 7. 自定义调整学习率 LambdaLR

![image-20210531084712294](https://gitee.com/github-25970295/blogpictureV2/raw/master/image-20210531084712294.png)

```python
torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
```

> - lr_lambda(function or list)- 一个`计算学习率调整倍数的函数`，输入通常为 step，当有多个参数组时，设为 list。

#### 8. 手动设置

```python
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(60):        
    lr = 30e-5
    if epoch > 25:
        lr = 15e-5
    if epoch > 30:
        lr = 7.5e-5
    if epoch > 35:
        lr = 3e-5
    if epoch > 40:
        lr = 1e-5
    adjust_learning_rate(optimizer, lr)
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/pytorch_learningrate/  

