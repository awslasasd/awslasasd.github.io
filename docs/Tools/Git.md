---
comments: true
---

# Git

## 基本指令

### 新建、管理远程仓库

直接在GitHub官网建立仓库

### 克隆、更新本地仓库副本

`git init` 创建仓库

`git clone https://github.com/xxx/xxx.git` 克隆网上的项目

`git branch` 查看分支列表

`git checkout <切换的分支名>` 切换到目标分支

#### 更新冲突解决方案

`git stash` 暂存本地修改

`git pull` 更新

`git stash pop ` 恢复本地修改

- 后续会显示出冲突修改的文件，需要手动人工处理

#### 提交文件到缓存区

`git status ` 查看修改文件列表

`git add [文件] [文件] ` 提交指定文件到缓存区

`git add *` 提交全部文件到缓存区

#### 提交本地仓库

`git commit -m "[跟新日志]"`

`git log` 查看修改的信息以及作者

#### 上传远程仓库

`git push`

#### 创建分支

`git checkout -b <你的分支名>`