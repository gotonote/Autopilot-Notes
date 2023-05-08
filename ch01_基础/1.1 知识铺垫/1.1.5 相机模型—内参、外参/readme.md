# 1.1.5 相机模型-内参、外参

针孔相机模型，包含四个坐标系：物理成像坐标系、像素坐标系、相机坐标系、世界坐标系。

相机参数包含：内参、外参、畸变参数

## 1.1.5.1 内参（Intrinsics）

<div align=center>
<img src="./imgs/1.1.5.1.jpg" width="600" height="300">
</div>
<div align=center>图1. 坐标系关系</div>

物理成像坐标系： $O'-x'-y'$

像素坐标系： $O-u-v$

相机坐标系： $O-x-y$

世界坐标系： $O-X-Y-Z$

在世界坐标系下的点 $P[X,Y,Z]^T$ ，通过相机坐标系下的光心 $O$ 投影到物理成像平面上的 $P'[X',Y',Z']^T$ ，对应到像素坐标系下的 $[u,v]^T$ 。

由相似三角形可以得到：

$$\frac Z f = -\frac X {X'}=-\frac Y {Y'}$$

带负号是因为小孔成像成的是倒像。为了简化模型，可以把物理成像平面看作为放到了相机的前方，这样可以直观的认为成立的正像（虚像），如下图2：

<div align=center>
<img src="./imgs/1.1.5.2.jpg" width="500" height="120">
</div>
<div align=center>图2. 小孔成像</div>

可以得到：

$$\begin{align} \frac Z f& = \frac X {X'}=\frac Y {Y'} \\ X'&=f\frac X Z \\ Y'&=f\frac Y Z \end{align}$$

从物理成像坐标系到像素坐标系之前，相差了一个**缩放**和**平移**。缩放是因为两个坐标系之前的表示的单位长度不一致，平移是因为两个坐标系的原点不一致。

假设，像素坐标在 $u$ 方向上缩放了 $\alpha$ 倍，在 $v$ 方向上缩放了 $\beta$ 倍，同时，原点平移了 $[c_x,c_y]^T$ 。那么点 $P'[X',Y',Z']^T$ 与像素坐标系下 $[u,v]^T$ 的关系为：

$$\begin{cases} u&=\alpha X'+c_x \\ v&=\beta Y'+c_y \\ \end{cases} \\ \\ \Downarrow \\ f_x = \alpha f \\ f_y= \beta f \\ \Downarrow \\ \begin{cases} u&=f_x \frac XZ + c_x \\ v&=f_y \frac YZ + c_y \\ \end{cases} \\$$

其中，变量的**单位**是 $f \rightarrow mm ; \alpha, \beta \rightarrow 像素/mm; f_x,f_y \rightarrow 像素$。将坐标进行归一化，写成矩阵形式，并对左侧像素坐标进行齐次化，方便后面的运算：

$$\begin{bmatrix} u \\ v\\ 1 \\ \end{bmatrix} = \frac 1Z \begin{bmatrix} f_x &0 &c_x \\ 0 &f_y &c_y\\ 0 &0 &1 \\ \end{bmatrix} \overset{\triangle}{=} \frac1Z \boldsymbol {KP} \\ \Downarrow \\ Z\begin{bmatrix} u \\ v\\ 1 \\ \end{bmatrix} = \begin{bmatrix} f_x &0 &c_x \\ 0 &f_y &c_y\\ 0 &0 &1 \\ \end{bmatrix} \overset{\triangle}{=} \boldsymbol {KP} \\$$

把中间的量组成的矩阵称为相机的**内参矩阵**（Camera Intrinsics） $\boldsymbol K$ 。


