---
title: "tmux 使用指南"
categories: ["z_mixed"]
author: "BluHuang"
date: 2026-07-25T11:33:29+0800
lastmod: 2026-07-25T11:33:29+0800
---

> 终端复用器，SSH 断开后会话不中断，任务继续运行。

## 1. 安装

```bash
brew install tmux
```

## 2. 会话管理（每条命令可单独复制）

```bash
tmux new -s 会话名
```

```bash
tmux ls
```

```bash
tmux attach -t 会话名
```

```bash
tmux attach
```

```bash
tmux kill-session -t 会话名
```

```bash
tmux kill-server
```

## 3. 会话内快捷键（先按 `Ctrl+B`，再按以下键）

| 快捷键 | 作用 |
|--------|------|
| `c` | 创建新窗口 |
| `n` | 下一个窗口 |
| `p` | 上一个窗口 |
| `"` | 水平分割面板（上下） |
| `%` | 垂直分割面板（左右） |
| `方向键` | 切换面板 |
| `x` | 关闭当前面板（按 y 确认） |
| `d` | 脱离会话（任务继续后台运行） |
| `[` | 进入滚动/复制模式（按 `q` 退出） |
| `?` | 显示帮助 |

## 4. 典型工作流

```bash
# 首次连接：创建会话
tmux new -s work

# 运行你的程序（如 opencode、vim、npm run dev）

# 临时断开（不中断任务）: Ctrl+B 然后按 D

# 下次连接：恢复会话
tmux attach -t work
```

## 5. 小技巧

- 建议使用短单词作为会话名：`w`、`c`、`o`
- 长时间任务配合 `caffeinate` 防止 Mac 睡眠（另开终端执行 `caffeinate`）
- 手机锁屏/切换 App 不影响 tmux 内任务