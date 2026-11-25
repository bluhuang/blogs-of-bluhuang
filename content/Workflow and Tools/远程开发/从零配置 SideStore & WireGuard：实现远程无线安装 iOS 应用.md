---
title: "从零配置 SideStore & WireGuard：实现远程无线安装 iOS 应用"
categories: ["Workflow and Tools"]
author: "BluHuang"
date: 2026-07-30T20:08:42+0800
lastmod: 2026-06-11T09:25:19+0800
---

# SideStore + WireGuard 环境搭建与远程 iOS 开发安装配置指南

> **目标**：在 Mac 和 iPhone 之间配置完全无线、可续签的 iOS App 安装环境，使 OpenCode 远程构建 `.ipa`，并通过手机上的 SideStore 直接安装，无需 USB。

---

## 🧠 背景与原理

- **免费 Apple ID** 签名的应用只有 **7 天有效期**，到期需重新签名。
- **SideStore** 是一款侧载工具，能在 iPhone 上独立完成应用签名和续签（**无需电脑**）。
- **WireGuard** 用于建立虚拟局域网，让 SideStore 能与 Apple 的签名服务器通信。
- **Jitterbug** 生成配对文件（`.plist`），使 SideStore 无需电脑即可控制设备。
- **iLoader** 是一个图形化工具，用于把 SideStore 安装到 iPhone。

整个过程分为 **准备文件 → 安装 WireGuard → 生成配对文件 → 安装 SideStore → 导入配置 → 日常使用**。

---

## 📦 第一阶段：OpenCode 自动准备所有必要文件

> 🤖 **此阶段全部由 OpenCode 自动完成**，你只需发送一条指令。

### 🎯 要做什么？
让 OpenCode 在你的 Mac 上下载以下文件到指定目录：
- `SideStore.ipa` – SideStore 应用本身（需安装到 iPhone）
- `sidestore.conf` – WireGuard 的隧道配置文件
- `jitterbugpair` – 生成配对文件的命令行工具
- `iLoader.dmg` – 用于在 Mac 上把 SideStore 安装到 iPhone 的工具

### 🤔 为什么这样做？
这些文件下载源都在 GitHub，国内访问不稳定，让 OpenCode 用 Mac 终端下载更快、更可靠，且集中存放便于管理。

### 🤖 发给 OpenCode 的指令（一键复制）

```text
请帮我准备 SideStore + WireGuard 环境所需的所有文件，统一存放到 `～/Desktop/code/install-something/SideStore-Files/`。

步骤：
1. 创建工作目录：mkdir -p ～/Desktop/code/install-something/SideStore-Files
2. 进入目录：cd ～/Desktop/code/install-something/SideStore-Files
3. 下载 SideStore.ipa：curl -L -o SideStore.ipa https://github.com/SideStore/SideStore/releases/latest/download/SideStore.ipa
4. 下载 sidestore.conf：curl -L -o sidestore.conf https://github.com/SideStore/SideStore/raw/main/SideStore.conf
5. 下载 Jitterbug（配对工具）：curl -L -o jitterbug-macos.zip https://github.com/osy/Jitterbug/releases/download/1.5.0/jitterbug-macos.zip
6. 解压：unzip jitterbug-macos.zip -d jitterbug
7. 下载 iLoader（用于安装 SideStore）：curl -L -o iLoader.dmg https://github.com/nab138/iloader/releases/download/v1.4.0/iLoader.dmg
8. 完成后输出文件清单。

如果某一步因版本号变化而失败，请先获取最新版本号再重试。
```

### ✅ 预期结果
目录下出现 `SideStore.ipa`, `sidestore.conf`, `iLoader.dmg` 及 `jitterbug/` 文件夹。

---

## 📲 第二阶段：手动安装 WireGuard 到 iPhone

> 🧑‍💻 **此阶段必须手动完成**，因为 OpenCode 无法操作 App Store。

### 🎯 要做什么？
在 iPhone 上安装 WireGuard App（用于建立 VPN 隧道）。

### 🤔 为什么这样做？
- SideStore 续签需要与 Apple 服务器通信，WireGuard 提供稳定的本机 VPN 通道。
- WireGuard 是官方推荐且完全免费的工具。

### 🧑‍💻 手动操作步骤
1. 在 iPhone 上**使用非国区 Apple ID** 登录 App Store。
   > 💡 **为什么必须非国区账号？** WireGuard 在中国大陆 App Store 未上架。如果没有外区账号，可以用 **LocalDevVPN** 作为替代（但配置稍复杂）。
2. 搜索 “WireGuard” 并安装。
3. 打开 WireGuard，保持空状态（后面会导入配置文件）。

---

## 🔧 第三阶段：手动生成配对文件（需要一次 USB 连接）

> 🧑‍💻 **此阶段需要 USB 数据线连接 iPhone 和 Mac**，但只需做一次。

### 🎯 要做什么？
用数据线连接 iPhone 和 Mac，运行 `jitterbugpair` 生成配对文件（`.plist`）。

### 🤔 为什么这样做？
配对文件是 SideStore 实现“免电脑”操作的关键。它授权 SideStore 与手机通信，完成签名和安装。

### 🧑‍💻 手动操作步骤
1. 用数据线连接 iPhone 到 Mac。
2. 打开 Mac 的“终端”App，执行以下命令：
   ```bash
   cd ～/Desktop/code/install-something/SideStore-Files/jitterbug
   
   ./jitterbugpair
   
   ```
3. 在 iPhone 上点击“信任此电脑”，输入锁屏密码。
4. 终端会生成一个 `UDID.mobiledevicepairing` 文件。
5. **重要**：将该文件改名为 `UDID.plist`（仅改后缀名）。

---

## ⚙️ 第四阶段：手动安装 SideStore 到 iPhone

> 🧑‍💻 **此阶段需要你在 Mac 上操作 iLoader**，但无需 iPhone 连接（除了 USB 线仍在）。

### 🎯 要做什么？
在 Mac 上运行 iLoader，把 SideStore 安装到 iPhone。

### 🤔 为什么这样做？
iLoader 是最简单的图形化安装工具，能自动处理签名和设备识别。

### 🧑‍💻 手动操作步骤
1. 双击之前下载的 `iLoader.dmg`（路径：`～/Desktop/code/install-something/SideStore-Files/iLoader.dmg`），将 `iLoader.app` 拖入 `Applications` 文件夹。
2. 打开 iLoader（从“启动台”或“应用程序”文件夹）。
3. 用数据线保持 iPhone 连接，在 iLoader 中选择你的 iPhone。
4. 登录你的 Apple ID（免费即可）。
5. 点击 **“Install SideStore (Stable)”** 按钮，等待安装完成。
6. 在 iPhone 上：**设置 → 通用 → VPN 与设备管理** → 找到你的 Apple ID 描述文件 → 点击“信任”。
7. 进入 **设置 → 隐私与安全性 → 开发者模式** → 打开开关 → 重启手机。

> ⚠️ 如果 iLoader 提示“No device”，检查数据线连接或更换 USB 口。

---

## 🔐 第五阶段：手动导入配对文件和 WireGuard 配置

> 🧑‍💻 **此阶段完全在 iPhone 上操作**，无需电脑。

### 🎯 要做什么？
- 将配对文件发送到 iPhone，供 SideStore 使用。
- 将 `sidestore.conf` 导入 WireGuard。

### 🧑‍💻 手动操作步骤
1. **发送配对文件**：把 Mac 上的 `pairing.plist` 文件通过隔空投送（AirDrop）发送到 iPhone，保存在“文件”App 中。
2. **导入到 SideStore**：打开 SideStore → 会提示需要配对文件 → 选择你保存的 `pairing.plist` 文件。
3. **发送 WireGuard 配置**：把 Mac 上的 `sidestore.conf` 通过 AirDrop 发到 iPhone。
4. **导入到 WireGuard**：打开 WireGuard → 点击“导入隧道” → 选择 `sidestore.conf`。
5. **开启隧道**：打开隧道开关（状态变为绿色）。

---

## 🚀 第六阶段：手动刷新证书并验证

> 🧑‍💻 **此阶段在 iPhone 上完成**，只需几秒钟。

### 🎯 要做什么？
在 SideStore 中刷新证书，确保安装环境可用。

### 🧑‍💻 手动操作步骤
1. 打开 WireGuard，确保隧道已连接（绿色）。
2. 打开 SideStore，进入 “My Apps” 页面。
3. 点击 SideStore 卡片下方的 **“7 DAYS”** 按钮 → 输入 Apple ID 密码。
4. 刷新成功后，按钮会显示剩余 7 天或 “Never expires”。
5. 至此，环境配置完成！你现在可以通过 SideStore 安装任何 `.ipa` 文件。

---

## 📦 日常使用流程（完全无线）

### 🖥️ Mac 端：让 OpenCode 自动构建并上传到 iCloud

> 🤖 **每次代码更新后，只需发给 OpenCode 一条指令。**

```text
请构建 Packpack 的最新 Release 版本，生成 .ipa 并放到 iCloud 指定目录：
~/Library/Mobile Documents/com~apple~CloudDocs/interact/PackPackInterAct/
```

### 📱 iPhone 端：手动安装

> 🧑‍💻 **每次安装新版本只需几步。**

1. 打开 “文件” App → iCloud 云盘 → `interact/PackPackInterAct/` → 点击 `.ipa` 文件。
2. 点击分享按钮 → 选择 **SideStore**。
3. 输入 Apple ID 密码 → 等待安装完成。
4. **每 7 天**打开一次 SideStore，它会自动刷新所有应用签名（前提是 WireGuard 已连接）。

---

## ❓ 常见问题

| 问题 | 解决方法 |
|------|----------|
| WireGuard 隧道无法连接 | 检查是否使用非国区账号下载；尝试重启手机或重新导入 `sidestore.conf` |
| SideStore 安装 IPA 时提示 “无效的配对文件” | 重新生成配对文件并导入（见第三阶段） |
| 刷新证书时要求输入密码但一直失败 | 检查 Apple ID 密码是否正确，或开启网络代理 |
| 手机无法下载 iCloud 文件 | 确保 Mac 和手机使用同一 Apple ID，且 iCloud Drive 已开启 |
| 7 天后续签失败 | 打开 WireGuard 隧道，打开 SideStore 手动点击 “Refresh All” |
| 安装 IPA 时提示 “maximum number of apps signed” | 免费 Apple ID 最多同时安装 3 个侧载应用。删除一个不常用的 App（如其他测试应用）再重试 |

---

## 🎉 总结

通过上述步骤，你将获得：

- ✅ 完全无线、无需电脑的 iOS App 安装能力
- ✅ OpenCode 自动构建 `.ipa` 并同步到 iCloud
- ✅ 手机端一键安装，每 7 天自动续签
- ✅ 所有工具和配置文件都有明确用途，可重复使用