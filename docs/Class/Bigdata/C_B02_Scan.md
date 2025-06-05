# 扫描工具

!!! note "相关资料"
    [CSDN-Shodan](https://blog.csdn.net/xt350488/article/details/144145956)<br>
    [CSDN-Nmap](https://blog.csdn.net/2302_82189125/article/details/135961736)<br>
    [Nmap密码爆破](https://blog.csdn.net/daocaokafei/article/details/117968132)<br>
    [开启23端口](https://jingyan.baidu.com/article/cbcede07fc663c42f40b4da7.html)<br>
    [Telnet连接](https://blog.csdn.net/apollon_krj/article/details/104185564)<br>
    [p0f](https://blog.csdn.net/qq_24305079/article/details/145810716)<br>
    [TCP三次握手](https://zhuanlan.zhihu.com/p/108504297)<br>


##  Shodan

Shodan是一款功能强大的网络搜索引擎，专门用于发现和分析互联网上的各种设备和服务，主要功能包括：  


1. **查找设备**：搜索摄像头、路由器、服务器等联网设备。  
2. **发现服务**：识别开放的端口（如 FTP、SSH、HTTP）及运行的服务。
3. **漏洞检测**：识别设备的已知安全风险。  
4. **数据可视化**：通过地图和图表展示设备分布与漏洞。  
5. **搜索过滤**：利用关键词、端口号、服务类型等多种条件进行搜索过滤  
6. **API 集成**：通过代码自动获取数据。   

Shodan的指令如下
```
shodan -h
```

![操作指令](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505041035573.png)

我们可以通过搜索语法限定搜索范围来实现IP搜索。如我们想限定国家(country:CN)和限定城市(city:Beijing)来搜索

可以借助Shodan的API来查询数据，具体操作如下


```
shodan init <API_KEY>
shodan search --limit 10 --fields ip_str country:cn city:beijing
```

![命令行查询](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505041035784.png)

得到以下的结果

![](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505041035369.png)

这里我们去查看第一个IP的系统详细信息，如地理位置，开放端口以及漏洞信息等

![image-20250504104343530](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505041043582.png)



除此之外，还可以利用其网页端进行查询，在添加限定词后，得到如下结果

![](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505041035284.png)

由于在上面的调用API搜索结果中，我们限制了只给出10条结果，因此我们可以在网站上去搜索对应IP来验证结果是否正确

去搜素上面API找到的第一个IP`123.57.161.194`,可以看到其正是China Beijing的IP。

![image-20250504103840431](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505041038506.png)


## Zenmap

Zenmap是nmap的用户图形界面，接下来我们去扫描一下`172.20.10.0/24`内的

![image-20250516201426442](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505162014522.png)

扫描后得到的结果如下

可以看到，找到了两个IP的Host，其拓补结构如下

![image-20250516202342319](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505162023384.png)

同时，也可以查看扫描到的主机的开放端口

![image-20250516202445495](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505162024531.png)

## Nmap

Nmap是一款用于网络发现和安全审计的开源工具，允许用户在网络上执行主机发现、端口扫描、服务识别和版本检测等操作，以帮助评估网络的安全性、发现主机和服务、以及识别潜在的安全威胁。

1. 主机发现： Nmap可以扫描整个网络，查找活动的主机。
2. 端口扫描： Nmap 可以扫描目标主机的开放端口，包括 TCP扫描、UDP扫描、SYN/ACK扫描等。
3. 操作系统检测： Nmap 能够尝试检测目标主机的操作系统类型和版本。
4. 漏洞扫描：通过结合Nmap的漏洞扫描脚本（例如Nmap NSE脚本），可以对目标系统进行漏洞扫描，并识别系统中存在的安全漏洞和弱点

### Nmap指令

```
nmap <IP>      #扫描指定IP
nmap <IP> <IP> #扫描多个IP
nmap -sV <IP>  #扫描指定IP并显示服务版本
nmap -O <IP>            #扫描指定IP并显示操作系统信息
nmap --script=vuln <IP> #扫描指定IP并显示漏洞信息
```


这里在VMware上装了windows7xp系统，然后利用Nmap进行扫描，windows7xp的IP地址为`192.168.114.132`

![image-20250504172436201](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505041724264.png)

扫描结果如下

![image-20250504172511114](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505041725145.png)

查看xp的版本信息

```
nmap -O -sV 192.168.114.132
```

![image-20250504201853970](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505042019102.png)

并查看可能存在的漏洞

```
nmap --script=vuln 192.168.114.132
```

![image-20250504201926016](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505042019087.png)


### Nmap扫描实践

接下来去扫描一下实际的网址，我们找一个北京的IP`47.94.250.43`来看一下

![image-20250516194805571](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505161948802.png)


从网页中可以看到其开放的端口以及可能存在的漏洞，接下来用Nmap来看一下相关信息

```
nmap -O -sV 47.94.250.43
```

![image-20250516200514355](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505162005413.png)





### Telnet密码爆破

#### 开启23端口

打开windows7系统的控制面板，进入程序，找到`打开或关闭Windows功能`，选择将Telnet服务器打开

![image-20250504220416157](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505042204204.png)

`win+r`，输入`services.msc`回去进入服务,将`Telnet`启动类型改为**自动**

![image-20250504220642007](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505042206037.png)

#### 添加用户进TelnetClients

`win+r`，输入`lusrmgr.msc`,打开组，选择`TelnetClients`，点击属性，添加账户即可

![image-20250504221012903](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505042210943.png)

#### 实践

我们使用一个nmap脚本来爆破telnet密码

1. 查看nmap关于telnet相关脚本

```
ls /usr/share/nmap/scripts/ | grep telnet
```

2. 查看到关于该脚本的官方文档信息

```
nmap --script-help=telnet-brute
```

![image-20250504222102172](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505042221261.png)

3. 运行telnet密码爆破脚本

根据文档给出的示例，可以运行以下命令

```
nmap -p 23 --script telnet-brute --script-args userdb=user.lst,passdb=pwd.lst,telnet-brute.timeout=8s 192.168.114.132
```

得到结果如下图所示，可以看到找到了两个账号密码

![image-20250504221822124](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505042218160.png)


此外,如果想要快速暴力破解，也可以利用`hydra`指令

```
hydra 192.168.114.132 -L user.lst -P pwd.lst telnet
```

得到的结果与上面利用nmap的telnet-brute脚本结果相同

![image-20250504222649959](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505042226997.png)

## p0f

P0f是一种被动操作系统指纹识别工具，它不向目标系统发送任何的数据，只是被动地接受来自目标系统的数据进行分析。

工作原理：被动地拦截原始的TCP数据包中的数据，就能收集到很多有用的信息：TCP SYN 和SYN/ACK数据包能反映TCP的链接参数，并且不同的TCP协议栈在协商这些参数的表现不同，从而可以确定远程主机的操作系统类型、版本、设备类型等信息。

优点：P0f不增加任何直接或间接的网络负载，没有名称搜索、没有秘密探测、没有ARIN查询。同时P0f在网络分析方面功能强大，可以用它来分析NAT、负载均衡、应用代理等。

### 工作原理

- **被动监听**：p0f通过监听网络接口捕获流经的网络数据包。它不会主动发送任何探测包，而是分析经过本地网卡的流量。
- **分析TCP/IP数据包**：p0f分析捕获的IPv4/IPv6头、TCP头、TCP握手以及应用层数据。不同的操作系统在处理TCP/IP协议时会表现出不同的特征，例如TCP选项、窗口大小、TTL值等。
- **指纹匹配**：p0f将捕获的数据包特征与内置的指纹数据库进行匹配。指纹数据库包含了各种操作系统在不同网络场景下的行为模式。

### 查看帮助信息

```
p0f -h
```

![image-20250504211655156](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505042116226.png)


> p0f -i iface # 监听指定网络接口

```
p0f -i eth0 'port 443' 
```

![image-20250504212633665](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505042126697.png)

>client: 客户端 IP 地址和端口。
os: 操作系统识别为 Linux 2.2.x-3.x。
dist: 距离（跳数）为 0，表示客户端在同一网络中。
params: 通用参数。
raw_sig: 原始签名信息。
link: 链路类型为以太网或调制解调器。
raw_mtu: 原始 MTU 值为 1500。
server: 服务器 IP 地址和端口。
os: 操作系统无法识别。
dist: 距离为 0。
params: 没有特定参数。
raw_sig: 原始签名信息。

!!! note "raw_sig组成"
    IPv4: 初始TTL值：TOS: MSS: TCP 选项: DF标志: ID增量: IP长度


>IP 层版本和首部长度：表示 IPv4 和相应的首部长度。
>TTL 和 TTL 修正值：TTL 是数据包在网络中存活的时间，初始 TTL 值为 64。
>TOS：服务类型字段，默认为 0，没有特别设置。
>MSS：TCP 连接的最大段大小，通常为 1460 字节，这是以太网标准的最大数据段。
>选项：mss=32 和 7，进一步说明 MSS 的设定和窗口大小。
>TCP 选项：包括 MSS、SACK、时间戳、NOP 和窗口缩放，这些选项通常用于提高 TCP 传输的可靠性。
>DF 标志：Don't Fragment，表示数据包不能被分片。
>ID 增量：ID 字段是递增的。
>IP 选项：没有特别的 IP 选项设置。

### TCP三次握手

TCP 提供面向有连接的通信传输，面向有连接是指在传送数据之前必须先建立连接，数据传送完成后要释放连接。在TCP/IP协议中，TCP协议提供可靠的连接服务，连接是通过三次握手进行初始化的。

所谓三次握手(Three-way Handshake)，是指建立一个 TCP 连接时，需要客户端和服务器总共发送3个报文。

三次握手的目的是连接服务器指定端口，建立 TCP 连接，并同步连接双方的序列号和确认号，交换 TCP 窗口大小信息。在 socket 编程中，客户端执行 connect() 时。将触发三次握手。

三次握手过程的示意图

![](https://pic1.zhimg.com/v2-8ce8c897b4d5e7397b25eb4d4b31d7fc_1440w.jpg)

- 第一次握手：

客户端将TCP报文标志位SYN置为1，随机产生一个序号值seq=J，保存在TCP首部的序列号(Sequence Number)字段里，指明客户端打算连接的服务器的端口，并将该数据包发送给服务器端，发送完毕后，客户端进入SYN_SENT状态，等待服务器端确认。


- 第二次握手：

服务器端收到数据包后由标志位SYN=1知道客户端请求建立连接，服务器端将TCP报文标志位SYN和ACK都置为1，ack=J+1，随机产生一个序号值seq=K，并将该数据包发送给客户端以确认连接请求，服务器端进入SYN_RCVD状态。


- 第三次握手：

客户端收到确认后，检查ack是否为J+1，ACK是否为1，如果正确则将标志位ACK置为1，ack=K+1，并将该数据包发送给服务器端，服务器端检查ack是否为K+1，ACK是否为1，如果正确则连接建立成功，客户端和服务器端进入ESTABLISHED状态，完成三次握手，随后客户端与服务器端之间可以开始传输数据了。

#### 实践

![image-20250520103645291](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505201036343.png)



1. **第一次握手（SYN）**:
   - 客户端（192.168.114.131:43594）向服务器（223.109.146.132:443）发送一个SYN包，请求建立连接。
   - 这个包中，客户端指定了其最大段大小（MSS）为1460字节，并设置了SYN标志。
2. **第二次握手（SYN-ACK）**:
   - 服务器收到SYN包后，回复一个SYN-ACK包，表示同意建立连接，并发送自己的SYN请求。
   - 这个包中，服务器也指定了其最大段大小（MSS）为1460字节，并设置了SYN和ACK标志。

由于`p0f`是一个被动的操作系统指纹库，前两次握手的 SYN 和 SYN-ACK 数据包具有明确的标志位（SYN 和 SYN-ACK），这些标志位是 p0f 识别和分析的关键特征，而第三次包含 ACK 标志位，且通常不携带额外的特征信息，因此无法识别第三次握手。



### 特殊情况

**host change**

当一个服务的请求从一个后端服务器转移到另一个服务器时，就会观察到Host Change(从kali主页面切换到Google主页)。

![image-20250520103722196](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505201037228.png)



**uptime**

系统运行时间（uptime）是操作系统自上次启动以来已经运行的时间

当使用 p0f 对网络中的主机进行扫描时，它会尝试从网络流量中提取各种信息。如果目标主机允许 p0f 获取其系统运行时间信息，就会出现 uptime

![image-20250520103730221](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505201037252.png)


  1. **系统运行时间（uptime）**
    `uptime = 39 days 10 hrs 23 min (modulo 49 days)`：表示目标主机（服务器）的系统已经连续运行了`39`天`10`小时`23`分钟。这里的modulo 49 days表示这个运行时间是取自一个 49 天的周期内的值。

  2. **原始频率（raw_freq）**
    `raw_freq = 976.74 Hz`：这是与目标主机的处理器频率校准相关的值。它表示处理器频率校准值为`976.74`赫兹。

  