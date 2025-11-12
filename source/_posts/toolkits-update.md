---
title: toolkits update
top: false
cover: false
toc: true
mathjax: true
date: 2025-11-03 16:08:35
password:
summary:
tags:
categories:
---

有时候遇到版本问题，需要更换版本

以及思考一下版本的大小问题

以及思考的cuda version 12.7的尴尬版本问题


需要检查的问题：
- ubuntu版本检查命令： 
    `lsb_release -a`  这台是24.04
- 哪个包管理
    - `dpkg -l | grep cuda-` 查看是否dpkg安装
    - `apt list --installed | grep cuda-12-6` 查看是否apt安装

dpkg是底层的，apt也是调用dpkg，所以查到了也问题不大




