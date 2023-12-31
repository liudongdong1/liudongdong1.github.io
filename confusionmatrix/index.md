# confusionmatrix


```matlab
clc;
clear;
close all;

classname = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','\heartsuit','\Delta',''};
num_class=29;
%统计矩阵，每一行对应一个label， 每一列代表预测值
fuse_matrix=[
];

fuse_matrix=fuse_matrix/60;
imagesc(fuse_matrix)

colormap(parula);
colorbar;

% 
% textStrings = num2str(fuse_matrix(:),'%0.2f');
% textStrings = strtrim(cellstr(textStrings));
% 
% for i = 1:length(textStrings)
%     if isequal(textStrings(i),{'0.00'})
%         textStrings(i) = {''};
%     end
% end
% 
% [x,y] = meshgrid(1:num_class); 
% hStrings = text(x(:),y(:),textStrings(:), 'HorizontalAlignment','center');
% midValue = mean(get(gca,'CLim')); 
% textColors = repmat(matshow(:) > midValue,1,3); 
% set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors


titax.TickLabelInterpreter='latex';

% title('ConfusionMatrix','FontName','Times New Roman','FontWeight','Bold','FontSize',16) %添加图形标题
set(gca,'xtick',1:29)
set(gca,'xticklabel',classname,'XTickLabelRotation',0,'FontName','Times New Roman','FontWeight','Bold','FontSize',15)
set(gca,'ytick',1:29)
set(gca,'yticklabel',classname)
set(gca,'YTickLabelRotation',30,'FontName','Times New Roman','FontWeight','Bold','FontSize',15)
```

![](https://lddpicture.oss-cn-beijing.aliyuncs.com/picture/image-20210727181202843.png)

---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/confusionmatrix/  

