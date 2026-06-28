---
title: "iOS 远程 Mac 开发环境搭建指南"
categories: ["z_mixed"]
author: "BluHuang"
date: 2026-06-04T09:42:47+0800
lastmod: 2026-06-04
---

> 目标：让任何人能根据本指南，用 iPhone 远程连接家里的 Mac，通过终端或 Web 界面写代码。

## 1. 准备工作

- 一台 Mac（保持开机，连接家里 Wi-Fi）
- 一部 iPhone（可连接 5G 或其它 Wi-Fi）
- 两个设备登录同一个 Apple ID（非必需，但方便）
- Mac 上已经安装了 Homebrew（推荐，用来装软件）

## 2. Mac 端安装与配置

### 2.1 安装 Tailscale（组网工具）

1. 打开官网 [https://tailscale.com/download](https://tailscale.com/download) → 下载 macOS 版安装包
2. 安装后，启动 Tailscale，用你的账号（Google/Microsoft/GitHub）登录
3. 菜单栏出现 Tailscale 图标，确保状态为 `Connected`

### 2.2 开启 Mac 的远程登录
- 打开 **系统设置 → 通用 → 共享**
- 打开 **远程登录** 开关
- 下方“允许访问”建议设为“所有用户”，或者只允许你的 Mac 用户名

### 2.3 安装 tmux（防断连工具）

打开 Mac 终端，执行：

```
brew install tmux
```

### 2.4 安装 OpenCode CLI（可选，用于 AI 编码）

```
brew install anomalyco/tap/opencode
```

或者用官方脚本：
```
curl -fsSL https://opencode.ai/install | bash
```

安装后，如果提示 `command not found`，添加 PATH：

```
echo 'export PATH="$HOME/.opencode/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
```

## 3. iPhone 端安装与配置

### 3.1 安装 Tailscale
- 在非国区 App Store 搜索 “Tailscale” 并安装
- 打开 App，用 **同一个账号** 登录
- 确保顶部开关为绿色 `Connected`

### 3.2 安装 Termius（SSH 客户端）
- App Store 搜索 “Termius” 并安装（免费版足够）

### 3.3 配置 Termius 连接 Mac
1. 打开 Termius，点击 **New Host**
2. 填写以下信息：
    - **Address**：Mac 的 Tailscale IP（在 Mac 终端输入 `tailscale status` 查看，形如 `100.x.x.x`）
    - **Port**：`22`
    - **Username**：你的 Mac 登录用户名
    - **Password**：你的 Mac 登录密码
3. 点击右上角 **Save**

## 4. 每次远程连接的操作流程

### 4.1 连接前检查
- Mac 上 Tailscale 已启动且 `Connected`
- iPhone 上 Tailscale 已打开且 `Connected`

### 4.2 用 Termius SSH 连上 Mac
- 打开 Termius，点击你保存的主机 → Connect
- 输入密码（如果没保存密码）

### 4.3 防止 Mac 自动睡眠（可选）

在 SSH 终端中输入：

```
caffeinate
```

然后按 `Ctrl+Z` 暂停，再输入 `bg` 放到后台。

> 不运行 caffeinate 的话，Mac 可能在长时间无操作后睡眠，导致连接断开。

### 4.4 使用 tmux 工作（防止网络中断）

- **首次创建会话**：

```
tmux new -s 会话名
```

例如：`tmux new -s work`

- **以后每次连接，恢复会话**：

```
tmux attach -t 会话名
```

如果不记得会话名，先输入 `tmux ls` 查看。

### 4.5 在 tmux 里开始开发

- 正常执行 `opencode`、`git`、`npm` 等命令
- 需要图形界面时，在 Mac 终端运行：
```
opencode web --hostname 0.0.0.0 --port 3000
```

然后在手机 Safari 访问 `http://Mac的Tailscale IP:3000`

### 4.6 临时断开（不中断任务）

按 `Ctrl+B` 然后按 `D`，回到普通 SSH 界面。之后可以直接关闭 Termius。

### 4.7 重新连接恢复

再次 SSH 到 Mac，然后执行：

```
tmux attach -t 会话名
```

一切恢复如初，任务继续运行。

## 5. 常见问题

- **连接不上**：检查 Tailscale 是否都 Connected；检查 Mac 远程登录是否开启。
- **速度非常慢**：查看 `tailscale status` 是否显示 `relay "xxx"`。如果是，说明走中继，需要参考笔记二中的解决方案。
- **opencode: command not found**：重新执行添加 PATH 的命令。