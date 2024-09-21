---
typora-copy-images-to: ..\..\picture
---

# Windows和Ubuntu双系统更改开机默认启动顺序

!!!bug Attention 
    如果是联想拯救者，请不要插着PD快充开机，否则会导致cfg文件重置<br>

## 1.打开终端进入/boot/grub/目录

`cd /boot/grub`

![image-20240922012318580](../../picture/image-20240922012318580.png)

## 2.编辑grub.cfg文件

`sudo gedit grub.cfg`输入电脑密码开始编辑

![image-20240922012347032](../../picture/image-20240922012347032.png)

!!!note 备份文件
    如果担心编辑出错，可以先保存文件再编辑<br>
    `mv grub.cfg grub.cfg.back` <br>

# 3.找到windows的位置

![image-20240922012356884](../../picture/image-20240922012356884.png)

# 4.粘贴到ubuntu之前

![image-20240922012405741](../../picture/image-20240922012405741.png)

# 5.保存，重启



# 6.美化措施

可以自行命名，修改结果会在boot界面显示

![image-20240922012423231](../../picture/image-20240922012423231.png)
