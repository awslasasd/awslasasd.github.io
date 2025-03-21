---
comments: true
---

# OneDrive

## 使用方法

### 备份任意文件夹

首先复制自己的onedrive的文件地址

```
"F:\Onedrive"
```

然后以管理员模式打开命令提示符，输入命令

```
mklink /d "F:\Onedrive"\mysite F:\mysite 
```

![image-20241225190337117](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412251903150.png)

即建立连接，`mklink /d`是建立映射，`"F:\Onedrive"\mysite`是想要保存进的文件夹 `F:\mysite`是想要备份的文件夹

![image-20241225190439935](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412251904039.png)

此时已经存在想要备份的文件了

如果想保存在Ondrive的文件夹里面，则可以

```
mklink /d "F:\Onedrive\Test"\awslasasd E:\awslasasd 
```



!!! note "备份一整个盘"
	查阅后发现微软官方给的回复是不可以，只能备份文件夹，没有用命令行去尝试，大家可以自行尝试<br>



## 一些解决方案

### 两个OneDrive

如图，win11系统侧边栏有两个OneDrive

![image-20241225185202612](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412251852758.png)

控制面板打开注册表编辑器，找到Desktop/NameSpace

![image-20241225185444542](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412251854591.png)

记得备份注册表！！！

删除{018D5C66-4533-4307-9B53-224DE2ED1FE6}立即生效

![image-20241225185757848](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412251857001.png)