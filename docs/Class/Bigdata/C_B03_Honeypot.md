# Honeypot

##  Hfish

!!! note "相关资料"
    [CSDN-HFish](https://blog.csdn.net/qq_49422880/article/details/121937941)<br>
    [CentOS7](https://www.cnblogs.com/tanghaorong/p/13210794.html)<br>
    [官方手册](https://hfish.net/#/highinteractive)<br>
    [HFish](https://blog.csdn.net/imtech/article/details/129688386)<br>

   蜜罐 技术本质上是一种对攻击方进行欺骗的技术，通过布置一些作为诱饵的主机、网络服务 或者信息，诱使攻击方对它们实施攻击，从而可以对攻击行为进行捕获 和分析，了解攻击方所使用的工具与方法，推测攻击意图和动机，能够让防御方清晰地了解他们所面对的安全威胁，并通过技术和管理手段来增强实际系统的安全防护能力。

 HFish是一款社区型免费蜜罐，侧重企业安全场景，从内网失陷检测、外网威胁感知、威胁情报生产三个场景出发，为用户提供可独立操作且实用的功能，通过安全、敏捷、可靠的中低交互蜜罐增加用户在失陷感知和威胁情报领域的能力。

HFish采用B/S架构，系统由管理端和节点端组成，管理端用来生成和管理节点端，并接收、分析和展示节点端回传的数据，节点端接受管理端的控制并负责构建蜜罐服务。

在HFish中，**管理端**只用于**数据的分析和展示**，**节点端**进行**虚拟蜜罐**，最后由**蜜罐来承受攻击**。

![](https://i-blog.csdnimg.cn/blog_migrate/731f257585fb6dd960edd2efd929a1c4.png)

特点：安全可靠、功能丰富、开放透明、快捷管理





本次安装在Docker中部署，首先确认Docker已经安装

```
docker version
```

在Docker中运行新的容器`hfish`，这里使用的镜像是`threatbook/hfish-server`

```
docker run -itd --name hfish \
-v /usr/share/hfish:/usr/share/hfish \
--network host \
--privileged=true \
threatbook/hfish-server:latest
```

接下来需要配置阿里云ESC安全组的入方向设置，如下图所示

![image-20250518083717334](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505180837473.png)

然后可以登录控制台首先，来查看攻击等相关信息

```
登陆地址：https://[server]:4433/web/
初始用户名：admin
初始密码：HFish2021
```

![image-20250518083835591](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505180838755.png)



查看扫描信息

![image-20250526153907492](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505261539640.png)

我们去shodan查看第一个IP信息

![image-20250526154822373](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505261548546.png)



考虑到SSH端口开放且有已知漏洞（CVE-2025-20465），这可能是一个蜜罐系统，用于吸引和分析潜在的网络攻击。
由于该IP地址是通过ISP动态分配的，这种类型的IP地址通常用于临时的网络连接或测试目的。















------

下面是另一种github源蜜罐

```
sudo docker pull imdevops/hfish
```

![image-20250514153440215](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505141534343.png)



```
sudo docker run -d --name hfish \
  -p 21:21 -p 2222:22 -p 23:23 -p 69:69 -p 3306:3306 -p 5900:5900 -p 6379:6379 -p 8080:8080 -p 8081:8081 -p 8989:8989 -p 9000:9000 -p 9001:9001 -p 9200:9200 -p 11211:11211 \
  -p 80:80 -p 443:443 -p 4433:4433 -p 7879:7879 \
  --restart=always imdevops/hfish:latest
```

![image-20250514153716484](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505141537566.png)

```
sudo docker ps -a
```

![image-20250514153711092](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505141537202.png)

```
sudo systemctl start firewalld
```



```
sudo firewall-cmd --zone=public --add-port=21/tcp --permanent
sudo firewall-cmd --zone=public --add-port=22/tcp --permanent
sudo firewall-cmd --zone=public --add-port=23/tcp --permanent
sudo firewall-cmd --zone=public --add-port=80/tcp --permanent
sudo firewall-cmd --zone=public --add-port=443/tcp --permanent
sudo firewall-cmd --zone=public --add-port=3306/tcp --permanent
sudo firewall-cmd --zone=public --add-port=6379/tcp --permanent
sudo firewall-cmd --zone=public --add-port=9000/tcp --permanent
sudo firewall-cmd --zone=public --add-port=9001/tcp --permanent
sudo firewall-cmd --reload
```

![image-20250514153550899](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505141535976.png)



```
sudo netstat -tuln | grep -E '21|22|23|80|443|3306|6379'
```

![image-20250514153609472](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505141536561.png)



```
sudo docker exec -it c26f73155ab6 /bin/sh
```

![image-20250514153908933](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505141539986.png)





```
cat config.ini
```

查看配置日志

![image-20250514153954719](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505141539866.png)



```
sudo docker logs c26f73155ab6
```

![image-20250514153758033](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505141537108.png)


之后就可以查看攻击日志了



