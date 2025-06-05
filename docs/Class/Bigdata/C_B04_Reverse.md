# 附加题

!!! note "相关资料"
    [CSDN-UPX魔改壳](https://blog.csdn.net/liKeQing1027520/article/details/142188160?spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-142188160-blog-144810681.235%5Ev43%5Epc_blog_bottom_relevance_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-142188160-blog-144810681.235%5Ev43%5Epc_blog_bottom_relevance_base3&utm_relevant_index=8)<br>
    [x32dbg去壳](https://blog.csdn.net/weixin_46287316/article/details/109669066)<br>
    [原题](https://tkazer.github.io/2025/03/06/GHCTF2025WP/index.html)<br>

## 去壳

先用查壳软件查看，发现加了UPX壳

![](./asserts/C2.1.png)

从他的提示中可以看到`UPX -> Markus & Laszlo ver. [ LIVV ] <- from file. [ ! Modified ! ] ( sign like UPX packer )` 这个UPX壳是`!Modified`，被修改过，因此直接去壳无法实现，效果如下图所示。

![image-20250521110353145](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505211103173.png)



接下来用010Editor发现UPX头被改了，如下图所示(更改前后对比)

![image-20250521110228448](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505211102614.png)



### 去壳方法一

直接 `upx -d <路径>`

![image-20250521110405553](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505211104573.png)

再去检查，发现为`.text`，即去壳成功

![image-20250521110419307](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505211104338.png)

### 去壳方法二

在修改后，用x32dbg打开，根据esp定律去壳。

先运行，找到`pushad`,关注`esp`的值，然后单步运行，发现`esp`值发生改变

![image-20250521112536500](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505211125625.png)

设置硬件断点，然后运行，之后会自动停下来， 往上可以看到一个popad的指令，该指令将寄存器的值进行还原。找到下面一个jmp指令所要跳转的位置，就是程序的OEP

![image-20250521112741328](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505211127369.png)

然后使用插件`Scylla`去壳即可得到下面的文件，`_SCY`文件即为目标文件

![image-20250521111745699](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505211117720.png)


## 逆向

用IDA Pro打开，查看其`main`函数如下

![image-20250521154656683](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505211547823.png)

分别去查看其子函数，并对其进行重命名，得到结果如下

![image-20250521154753118](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505211547253.png)


其中每个函数的含义如下：

1. **`InputPrompt`**
    ➡️ 格式化输出提示信息（类似 `printf`）。

2. **`scan`**
    ➡️ 从标准输入读取用户输入（类似 `scanf`）。

3. **`CreateRandNumber`**
    ➡️ 使用固定随机种子生成密钥数组 `dword_4043D8[8]`，用于后续加密。

4. **`Remap_TEA`**
    ➡️ 用 TEA 变体算法对输入的 32 字节 flag 进行加密处理（每 8 字节为一组），加密后与内置密文比较验证正确性。

在分析完后，进行逆向操作。


通过debugger模式，设置断点，查看数组`dword_4143D8[]`的值如下

```
0x64,0x96,0x50,0x16,0x69,0xFF,0xBE,0x60
```

其对应的10进制数值为

```
100, 150, 80, 22, 105, 255, 190, 96
```

![image-20250521154838796](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505211548138.png)



数组`byte_414060`的定义如下，即为

```
byte_414060[32] = {
    0xDC, 0x45, 0x1E, 0x03, 0x89, 0xE9, 0x76, 0x27,
    0x47, 0x48, 0x23, 0x01, 0x70, 0xD2, 0xCE, 0x64,
    0xDA, 0x7F, 0x46, 0x33, 0xB1, 0x03, 0x49, 0xA3,
    0x27, 0x00, 0xD1, 0x2C, 0x37, 0xB3, 0xBD, 0x75
};
```

![image-20250521160140504](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505211601557.png)


对加密函数进行反汇编，其中，加密函数如下：

```C++
int __cdecl Remap_TEA(int a1)
{
  unsigned int v1;
  unsigned int v2;
  int result;
  int n4_1;
  unsigned int v5;
  unsigned int v6;
  unsigned int v7;
  unsigned int v8;
  unsigned int v9;
  unsigned int v10;
  unsigned int v11;
  unsigned int v12;
  unsigned int v13;
  unsigned int v14;
  unsigned int v15;
  unsigned int v16;
  unsigned int v17;
  int i;
  int n4;
  _DWORD v20[4];
  _DWORD IamTheKeyYouKnow[5];

  strcpy((char *)IamTheKeyYouKnow, "IamTheKeyYouKnow");
  for ( i = 0; i < 15; ++i )
    *((_BYTE *)IamTheKeyYouKnow + i) ^= LOBYTE(dword_4043D8[i % 8]);
  memcpy(v20, IamTheKeyYouKnow, sizeof(v20));
  n4 = 4;
  do
  {
    v17 = *(_DWORD *)(a1 + 8 * (4 - n4) + 4);
    v16 = *(_DWORD *)(a1 + 8 * (4 - n4)) + ((v20[1] + (v17 >> 5)) ^ (v17 + 1579382783) ^ (v20[0] + 16 * v17));
    v15 = v17 + ((v20[3] + (v16 >> 5)) ^ (v16 + 1579382783) ^ (v20[2] + 16 * v16));
    v14 = v16 + ((v20[1] + (v15 >> 5)) ^ (v15 - 1136201730) ^ (v20[0] + 16 * v15));
    v13 = v15 + ((v20[3] + (v14 >> 5)) ^ (v14 - 1136201730) ^ (v20[2] + 16 * v14));
    v12 = v13
        + ((v20[3] + ((v14 + ((v20[1] + (v13 >> 5)) ^ (v13 + 443181053) ^ (v20[0] + 16 * v13))) >> 5)) ^ (v14 + ((v20[1] + (v13 >> 5)) ^ (v13 + 443181053) ^ (v20[0] + 16 * v13)) + 443181053) ^ (v20[2] + 16 * (v14 + ((v20[1] + (v13 >> 5)) ^ (v13 + 443181053) ^ (v20[0] + 16 * v13)))));
    v1 = v14
       + ((v20[1] + (v13 >> 5)) ^ (v13 + 443181053) ^ (v20[0] + 16 * v13))
       + ((v20[1] + (v12 >> 5)) ^ (v12 + 2022563836) ^ (v20[0] + 16 * v12));
    v11 = v12 + ((v20[3] + (v1 >> 5)) ^ (v1 + 2022563836) ^ (v20[2] + 16 * v1));
    v10 = v1 + ((v20[1] + (v11 >> 5)) ^ (v11 - 693020677) ^ (v20[0] + 16 * v11));
    v9 = v11 + ((v20[3] + (v10 >> 5)) ^ (v10 - 693020677) ^ (v20[2] + 16 * v10));
    v8 = v10 + ((v20[1] + (v9 >> 5)) ^ (v9 + 886362106) ^ (v20[0] + 16 * v9));
    v7 = v9 + ((v20[3] + (v8 >> 5)) ^ (v8 + 886362106) ^ (v20[2] + 16 * v8));
    v6 = v8 + ((v20[1] + (v7 >> 5)) ^ (v7 - 1829222407) ^ (v20[0] + 16 * v7));
    v2 = v7 + ((v20[3] + (v6 >> 5)) ^ (v6 - 1829222407) ^ (v20[2] + 16 * v6));
    v5 = v2
       + ((v20[3] + ((v6 + ((v20[1] + (v2 >> 5)) ^ (v2 - 249839624) ^ (v20[0] + 16 * v2))) >> 5)) ^ (v6 + ((v20[1] + (v2 >> 5)) ^ (v2 - 249839624) ^ (v20[0] + 16 * v2)) - 249839624) ^ (v20[2] + 16 * (v6 + ((v20[1] + (v2 >> 5)) ^ (v2 - 249839624) ^ (v20[0] + 16 * v2)))));
    *(_DWORD *)(a1 + 8 * (4 - n4)) = (v6 + ((v20[1] + (v2 >> 5)) ^ (v2 - 249839624) ^ (v20[0] + 16 * v2))) ^ 0xF;
    *(_DWORD *)(a1 + 8 * (4 - n4) + 4) = v5 ^ 0xF;
    n4_1 = n4;
    result = --n4;
  }
  while ( n4_1 );
  return result;
}
```

因此，逆向代码如下

```
import struct

def u32(x):
    return x & 0xFFFFFFFF

CONSTANTS_C = [
    u32(1579382783),
    u32(-1136201730),
    u32(443181053),
    u32(2022563836),
    u32(-693020677),
    u32(886362106),
    u32(-1829222407),
    u32(-249839624)
]
def f_transform(val, k_part_cycle1, k_part_cycle0, delta_const):
    term1 = u32(k_part_cycle1 + (val >> 5))
    term2 = u32(val + delta_const)
    term3 = u32(k_part_cycle0 + u32(16 * val))
    return u32(term1 ^ term2 ^ term3)

dword_4143D8_bytes = [0x64, 0x96, 0x50, 0x16, 0x69, 0xFF, 0xBE, 0x60]
key_material_bytes = bytearray(b"IamTheKeyYouKnow")
for i in range(15):
    key_material_bytes[i] ^= dword_4143D8_bytes[i % 8]
v20_key = []
for i in range(4):
    v20_key.append(struct.unpack('<I', key_material_bytes[i*4 : i*4+4])[0])

byte_414060_hex = [
    0xDC, 0x45, 0x1E, 0x03, 0x89, 0xE9, 0x76, 0x27,
    0x47, 0x48, 0x23, 0x01, 0x70, 0xD2, 0xCE, 0x64,
    0xDA, 0x7F, 0x46, 0x33, 0xB1, 0x03, 0x49, 0xA3,
    0x27, 0x00, 0xD1, 0x2C, 0x37, 0xB3, 0xBD, 0x75
]
encrypted_dwords = []
for i in range(0, len(byte_414060_hex), 4):
    encrypted_dwords.append(struct.unpack('<I', bytes(byte_414060_hex[i:i+4]))[0])

def decrypt_block_pair(enc_x0_dword, enc_x1_dword, key_v20_parts, C_constants):
    val_L_final_stage = u32(enc_x0_dword ^ 0xF)
    val_R_final_stage = u32(enc_x1_dword ^ 0xF)
    v2 = u32(val_R_final_stage - f_transform(val_L_final_stage, key_v20_parts[3], key_v20_parts[2], C_constants[7]))
    v6 = u32(val_L_final_stage - f_transform(v2, key_v20_parts[1], key_v20_parts[0], C_constants[7]))
    v7 = u32(v2 - f_transform(v6, key_v20_parts[3], key_v20_parts[2], C_constants[6]))
    v8 = u32(v6 - f_transform(v7, key_v20_parts[1], key_v20_parts[0], C_constants[6]))
    v9 = u32(v7 - f_transform(v8, key_v20_parts[3], key_v20_parts[2], C_constants[5]))
    v10 = u32(v8 - f_transform(v9, key_v20_parts[1], key_v20_parts[0], C_constants[5]))
    v11 = u32(v9 - f_transform(v10, key_v20_parts[3], key_v20_parts[2], C_constants[4]))
    v1_val = u32(v10 - f_transform(v11, key_v20_parts[1], key_v20_parts[0], C_constants[4]))
    v12 = u32(v11 - f_transform(v1_val, key_v20_parts[3], key_v20_parts[2], C_constants[3]))
    v_temp_L_for_v12 = u32(v1_val - f_transform(v12, key_v20_parts[1], key_v20_parts[0], C_constants[3]))
    v13 = u32(v12 - f_transform(v_temp_L_for_v12, key_v20_parts[3], key_v20_parts[2], C_constants[2]))
    v14 = u32(v_temp_L_for_v12 - f_transform(v13, key_v20_parts[1], key_v20_parts[0], C_constants[2]))
    v15 = u32(v13 - f_transform(v14, key_v20_parts[3], key_v20_parts[2], C_constants[1]))
    v16 = u32(v14 - f_transform(v15, key_v20_parts[1], key_v20_parts[0], C_constants[1]))
    v17 = u32(v15 - f_transform(v16, key_v20_parts[3], key_v20_parts[2], C_constants[0]))
    original_x0 = u32(v16 - f_transform(v17, key_v20_parts[1], key_v20_parts[0], C_constants[0]))
    original_x1 = v17
    return original_x0, original_x1

decrypted_a1_dwords = []
for i in range(0, len(encrypted_dwords), 2):
    enc_x0 = encrypted_dwords[i]
    enc_x1 = encrypted_dwords[i+1]
    dec_x0, dec_x1 = decrypt_block_pair(enc_x0, enc_x1, v20_key, CONSTANTS_C)
    decrypted_a1_dwords.append(dec_x0)
    decrypted_a1_dwords.append(dec_x1)

decrypted_a1_bytes = bytearray()
for dword_val in decrypted_a1_dwords:
    decrypted_a1_bytes.extend(struct.pack('<I', dword_val))

print("派生密钥 (v20) 的DWORD表示 (十六进制):")
print([hex(k) for k in v20_key])
print("\n加密后的DWORD (来自 byte_414060):")
print([hex(d) for d in encrypted_dwords])
print("\n解密后的 a1 的DWORD表示 (十六进制):")
print([hex(d) for d in decrypted_a1_dwords])
print("\n解密后的 a1 以十六进制字节序列表示:")
print(' '.join(f'{b:02X}' for b in decrypted_a1_bytes))
print("\n解密后的 a1 以ASCLL表示:")
print(decrypted_a1_bytes.decode('utf-8'))
```

最后运行结果如下

![image-20250522141627794](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505221416877.png)

将密钥`NSSCTF{!!!Y0u_g3t_th3_s3cr3t!!!}`输入exe发现结果正确，逆向完成

![image-20250522141725188](https://zyysite.oss-cn-hangzhou.aliyuncs.com/202505221417211.png)

