---
title: "Tailscale 连接指南与使用详解"
categories: ["z_mixed"]
author: "BluHuang"
date: 2026-06-04T19:01:05+0800
lastmod: 2026-06-04
---

> 快速参考：状态判断、与 Clash 冲突解决、relay 问题处理。

## 1. 常用命令

```bash
tailscale status
```

```bash
tailscale ping 100.88.241.40
```

```bash
tailscale netcheck
```

```bash
tailscale down
```

```bash
tailscale up
```

```bash
sudo pkill -f Tailscale && sleep 3 && sudo tailscale up
```

```bash
sudo tailscale set --relay-server-port=40000
```

## 2. 状态解读

| 状态 | 含义 |
|------|------|
| `active; direct` | ✅ 点对点直连，速度最快 |
| `active; direct [IPv6地址]` | ✅ IPv6 直连，也很快 |
| `idle` | ✅ 空闲待机，有流量自动激活，无 relay 即正常 |
| `active; relay "hkg/lax/..."` | ❌ 走公共中继，速度慢，需处理 |

## 3. 出现 relay 怎么办？

### 3.1 优先尝试：重启 Mac（最有效）

清除残留虚拟网卡和路由表，恢复 IPv6 直连。

### 3.2 重启后仍 relay 则依次尝试：

**临时关闭所有 VPN（包括 Clash TUN 模式）**，然后重启 Tailscale

```bash
sudo pkill -f Tailscale && sleep 3 && sudo tailscale up
```

**临时关闭 macOS 防火墙（测试用）**

```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off
```

测试完立即重新开启：

```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on
```

**手动清理 Clash 残留的 utun 接口**（针对 0~7 编号）

```bash
sudo ifconfig utun0 down && sudo ifconfig utun0 destroy
```

**修改 Clash 配置（需同时使用 Clash TUN 时）**  
在配置文件中添加：

```yaml
tun:
  exclude-interface:
    - Tailscale
  route-exclude-address:
    - 100.64.0.0/10
```

### 3.3 最终方案：自建对等中继（Peer Relay）

```bash
sudo tailscale set --relay-server-port=40000
```

然后在路由器上做端口转发（UDP 40000 → Mac 内网 IP），并在 Tailscale 后台 ACL 中授权。  
之后状态会显示 `via peer-relay`（比公共 DERP 快）。

## 4. 与 Clash 共存的已知经验

- 当前环境：Clash TUN 开启 + Tailscale 走 IPv6 直连 → 可共存，无需关闭 Clash
- 若未来又出现 relay，先关 Clash 再重启 Tailscale 判断是否是 Clash 导致
- 最稳定组合：Clash 规则排除 Tailscale + 启用 IPv6

## 5. 一句话总结

**`tailscale status` 显示 `direct` 或 `idle`（无 relay）即正常；出现 relay 优先重启 Mac。**