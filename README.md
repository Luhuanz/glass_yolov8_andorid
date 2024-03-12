# glass_yolov8_andorid

玻璃破碎yolov8-seg-Ncnn实时监测及图像检测
①②③④⑤⑥⑦⑧⑨⑩

## &#x1F554;1. 训练自己的数据集

### ①  安装cuda,cudnn,torch, torchvison

- (1).首先要在设备管理器中查看你的显卡型号,然后去[NVIDIA 驱动](https://www.nvidia.cn/Download/index.aspx?lang=cn)下载驱动。
![驱动](imgs/驱动.png)
- （2）完成之后，在cmd中输入执行：``nvidia-smi``
   ![驱动验证](imgs/驱动验证.png)
   注：**图中的 CUDA Version是当前Driver版本能支持的最高的CUDA版本。**

- (3) 安装[cuda](https://developer.nvidia.com/cuda-toolkit-archive)，cudnn。(我的CUDA用的是11.8版本)
 请参考[CUDA安装教程（超详细）](https://blog.csdn.net/m0_45447650/article/details/123704930?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171023307216800197048699%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171023307216800197048699&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-123704930-null-null.142^v99^pc_search_result_base1&utm_term=%E5%AE%89%E8%A3%85cuda&spm=1018.2226.3001.4187)该教程安装。
(4)

  <!-- 下载官方文档与权重
  然后还需要安装ultralytics，目前YOLOv8核心代码都封装在这个依赖包里面，可通过以下命令安装

``pip install ultralytics`` -->
