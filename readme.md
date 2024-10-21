CPU GPU异步，所以可以算一下哪边是瓶颈
param
    调用的时候只传名，不传参？-先查一查编译能否自动优化
cudamemeset
半精度可以尝试一下
float4可以尝试一下，能够burst访存
bank conflict
wmma
https://zhuanlan.zhihu.com/p/659142274:bank conflict与TILEX
