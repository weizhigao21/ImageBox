# 本地更新组件

来自用户提供的 `G:/code/py-updater/pyupdater/`，版本 1.0.0。
复制时提交：`f263754f731ec1d55bc3f5ae9fd38ae4f1445242`。
放在项目 src 下，避免发布后的程序依赖开发机器上的另一个工程目录。

像匣使用 config/check/apply/updater_main 核心 API；Qt 界面与线程生命周期由 imgtool 自己管理。
本地适配：更新资源须匹配产品前缀，下载校验长度、文件名和取消状态；修复备份中途失败时的恢复与 Windows 64 位进程句柄；将更新组件副本与冻结的主程序一起分发。
