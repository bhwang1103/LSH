# LSH for float data
注意：为了在实验过程中区分flann库的LSH_Index,这里的LSH命名为 **FLSH_INDEX**

## C++ code
* algorithm :FLSH_INDEX方法
* io :使用HDF5格式对实验数据进行输入输出
* util :FLSH_INDEX方法需要的支持头文件
* test :测试代码

## 实验方法（ VS2015 debug(64) ）
* 将C++ code 添加到项目属性管理器VC++目录的**包含目录**下
* 将dll文件添加到项目属性管理器VC++目录的**库目录下**
* 将 flann_cpp_s-gd.lib  flann_s-gd.lib flann-gd.lib 添加到**链接器**的附加依赖项
* 运行test代码

