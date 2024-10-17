CPU GPU异步，所以可以算一下哪边是瓶颈
malloc和free数量不一致
malloc和memcpy可以外提
    外提难度有点大啊
param
    调用的时候只传名，不传参？-先查一查编译能否自动优化
float16可以尝试一下
float4可以尝试一下，能够burst访存
bank conflict
wmma
https://zhuanlan.zhihu.com/p/659142274:bank conflict与TILEX
