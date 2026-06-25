---
title: "mac远程安装开发好的ios app实践 - SideStore"
categories: ["z_mixed"]
author: "BluHuang"
date: 2026-06-25T21:31:11+0800
lastmod: 2026-06-25T21:31:11+0800
---

# SideStore 无线远程安装方案（Packpack 开发实践）

> **目标**：在家庭以外的任意地点，通过 SideStore 将 Mac 上由 OpenCode 自动构建的 `.ipa` 安装到 iPhone，无需 USB 线、无需同一 Wi-Fi，完全免费。

## 一、原理：SideStore 如何实现“无线远程安装”？

### 1.1 苹果的限制
- 免费 Apple ID 签名的应用有效期为 **7 天**，到期必须重新签名。
- 正常的 Xcode 无线调试要求 Mac 与 iPhone **同一本地网络**（Bonjour 广播），无法跨网络工作。
- iOS 系统禁止通过蜂窝数据直接安装侧载应用，必须检测到 Wi-Fi 连接。

### 1.2 SideStore 的解决思路
- **VPN 隧道**：利用 WireGuard / LocalDevVPN 在手机本地创建虚拟网络接口，模拟设备与 Apple 签名服务器的通信，实现**脱离电脑的独立续签**。
- **配对文件**：通过 `iLoader` 工具生成设备配对文件（`.plist`），授权 SideStore 与手机通信，无需依赖 Mac 常驻服务。
- **安装机制**：用户从 iCloud 下载 `.ipa` 后，SideStore 通过本地 VPN 完成签名并安装，**仅需手机连接任意 Wi-Fi**（不要求与 Mac 同网络）。

### 1.3 我们的工作流
```
OpenCode (Mac) → 自动构建 .ipa → 存入 iCloud 云盘
                ↓
你 (iPhone) → 从 iCloud 下载 .ipa → 分享到 SideStore → 安装
                ↓
           每隔 7 天打开 SideStore（LocalDevVPN 开启）自动续签
```

## 二、环境准备（一次性配置，需在家用 USB 线完成）

### 2.1 下载必要的工具
- **iLoader**：用于安装 SideStore 并生成配对文件。  
  [https://iloader.app](https://iloader.app)（Mac 版）
- **LocalDevVPN**：SideStore 推荐的 VPN 工具，用于独立续签。  
  从 App Store 下载（**需要非国区 Apple ID**，因为该应用在中国大陆未上架）。
- **SideStore.ipa**（可选，iLoader 会帮你安装）

### 2.2 使用 iLoader 安装 SideStore
1. 用 USB 数据线连接 iPhone 到 Mac，解锁并点击“信任”。
2. 打开 iLoader，登录你的 Apple ID（免费即可）。
3. 在设备列表中选择你的 iPhone，点击 **“Install SideStore”**（或 **“Install SideStore + LiveContainer”**，后者可突破 3 个应用限制，但非必需）。
4. 等待安装完成。此时手机上会出现 SideStore 图标。

### 2.3 配置 LocalDevVPN
1. 打开 LocalDevVPN，点击 **Connect**，允许添加 VPN 配置（需要输入锁屏密码）。
2. 保持 VPN 在后台运行（后续续签时会自动连接）。

### 2.4 首次刷新 SideStore 证书
1. 打开 SideStore，登录与 iLoader 相同的 Apple ID。
2. 进入 **“My Apps”** 页面，点击 SideStore 卡片下方的 **“7 DAYS”** 按钮。
3. 系统会要求允许 SideStore 执行本地网络操作，点击允许。
4. 回到 **设置 → 通用 → VPN 与设备管理**，再次信任你的 Apple ID 描述文件。
5. 打开 **设置 → 隐私与安全性 → 开发者模式**，开启并重启手机。
6. 重新打开 SideStore，再次点击 **“7 DAYS”**，确认无报错，显示剩余 7 天。

### 2.5 验证无线安装能力
- 断开 USB 线，将手机连接到任意 Wi‑Fi（可以不是家里 Wi‑Fi）。
- 准备一个测试 `.ipa` 文件（例如 Xcode 导出的 Ad Hoc IPA）。
- 在 iPhone 的“文件”App 中点击该 IPA，选择分享 → SideStore → 输入 Apple ID 密码，应能成功安装。

## 三、日常远程开发操作（OpenCode 自动化）

### 3.1 Mac 端：让 OpenCode 自动构建并上传 iCloud
创建脚本 `~/Desktop/code/PackPack/build_and_export_ipa.sh`，内容如下：

```bash
#!/bin/bash
set -euo pipefail

PROJECT_DIR="$HOME/Desktop/code/PackPack"
SCHEME="Packpack"
CONFIGURATION="Release"
ICLOUD_BASE="$HOME/Library/Mobile Documents/com~apple~CloudDocs/interact/PackPackInterAct/ipa"

cd "$PROJECT_DIR"

# 获取 commit 信息（用于文件夹命名）
COMMIT_MSG=$(git log -1 --pretty=%B | head -c 40 | tr -d '\n' | tr -c '[:alnum:]_-' '-')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEST_DIR="$ICLOUD_BASE/${TIMESTAMP}_${COMMIT_MSG}"
mkdir -p "$DEST_DIR"

echo "🔨 构建 Release..."
xcodebuild clean build -scheme "$SCHEME" -configuration "$CONFIGURATION" -derivedDataPath ./build -destination 'generic/platform=iOS' CODE_SIGN_STYLE=Automatic

APP_PATH=$(find ./build/Build/Products/Release-iphoneos -name "*.app" -type d | head -n 1)
if [ -z "$APP_PATH" ]; then
    echo "❌ 找不到 .app"
    exit 1
fi

# 打包 IPA
PAYLOAD="Payload"
rm -rf "$PAYLOAD"
mkdir "$PAYLOAD"
cp -R "$APP_PATH" "$PAYLOAD/"
zip -qr "Packpack.ipa" "$PAYLOAD"
rm -rf "$PAYLOAD"

mv "Packpack.ipa" "$DEST_DIR/"
echo "✅ IPA 已保存至: $DEST_DIR/Packpack.ipa"
```

**赋予执行权限**：
```bash
chmod +x ~/Desktop/code/PackPack/build_and_export_ipa.sh
```

**让 OpenCode 执行**：  
每次修改代码后，只需对 OpenCode 说：“执行 `build_and_export_ipa.sh` 打包并上传 iCloud”。它会自动完成构建、命名、上传。

### 3.2 iPhone 端：安装最新版本
1. 打开 **“文件”App** → iCloud 云盘 → `interact/PackPackInterAct/ipa/` → 选择最新日期的文件夹。
2. 点击 `Packpack.ipa` 文件。
3. 点击右上角 **分享** 按钮 → 选择 **SideStore**。
4. 输入你的 Apple ID 密码，等待安装完成。
5. 安装后，建议立即打开 SideStore，确认证书剩余时间（7 天）。

### 3.3 自动续签（无需电脑）
- 平时保持 **LocalDevVPN** 后台开启（可在设置中允许后台刷新）。
- 每隔几天打开一次 SideStore，它会自动刷新所有已安装应用的证书。
- 如果证书即将过期，SideStore 会推送通知提醒。

## 四、踩坑记录与解决方案

### 4.1 安装时报错：`Could not determine device's UDID`
- **原因**：配对文件失效或未正确生成。
- **解决**：
  1. 用 USB 线连接手机到 Mac。
  2. 打开 iLoader → 点击 **“Delete Stored Pairing”**（删除存储的配对）。
  3. 重新点击 **“Install SideStore”** 覆盖安装。
  4. 手机上重新信任描述文件，刷新证书。

### 4.2 报错：`Ensure Wi-Fi and LocalDevVPN are connected`
- **原因**：手机未连接 Wi‑Fi（即使蜂窝网络 + VPN 也不行）。
- **解决**：确保 iPhone 已连接任意 Wi‑Fi（可以不是家里的，咖啡馆、酒店均可）。**纯蜂窝无法安装**。

### 4.3 报错：`AFC was unable to manage files on the device`
- **原因**：配对文件权限过期或 USB 信任失效。
- **解决**：重新用 iLoader 安装一次 SideStore（无需删除 App），然后重启手机。

### 4.4 刷新证书时提示 `SideStore could not refresh` 但无详细错误
- **原因**：LocalDevVPN 未正确连接或 Anisette 服务器不可达。
- **解决**：
  - 确认 LocalDevVPN 显示“已连接”。
  - 尝试切换 SideStore 中的 Anisette 服务器（设置 → Anisette Servers → 选择 `https://macley.sideload.icu/`）。
  - 重启手机后重试。

### 4.5 应用安装后打开闪退
- **原因**：证书未正确签名或应用权限不足。
- **解决**：打开 SideStore → 点击该应用卡片的 **“7 DAYS”** 重新签名，然后信任描述文件。

### 4.6 iCloud 目录路径找不到
- **原因**：Mac 上 iCloud Drive 未开启或文件夹未同步。
- **解决**：在 Mac 系统设置中开启 iCloud Drive，并确保“桌面与文档文件夹”同步已开启（可选）。手机端“文件”App 中应能看到相同目录。

## 五、总结

| 方案 | 费用 | 远程无线安装 | 续签方式 | 推荐度 |
|------|------|--------------|----------|--------|
| **SideStore + LocalDevVPN** | 免费 | ✅（手机需 Wi‑Fi） | 手机独立续签 | ⭐⭐⭐⭐ |
| AltStore | 免费 | ❌（需同网络） | 依赖电脑 AltServer | ⭐⭐ |
| TestFlight (99美元) | 99美元/年 | ✅（蜂窝亦可） | 自动 | ⭐⭐⭐⭐⭐（稳定） |

**最终建议**：  
如果你不愿意每年支付 99 美元，SideStore 是免费方案中的最佳选择。按照本文配置后，你可以在任何有 Wi‑Fi 的地方远程安装最新版 App，且无需担心 7 天过期问题。如果未来需要更稳定的体验或发布到 App Store，再考虑付费账号。

**最后更新**：2026-06-14  
**适用系统**：iOS 26.0+ / macOS 15.7+（Intel 或 Apple Silicon 均可）