1. 现在的P1_web.py，来自于官网，新增了random和uniform采样
    删去了取中位数部分
    random在下面dataset那一行和官网保持一致
    uniform需要batch那collate
    然后farthest算了

2. test.cu 增加了代码鲁棒性
    64*64 blk的cb和cbr没有实现convmax的技术，现在也没实现，但是在代码中增加了使用条件，同时调整maxpool在main函数中的numpoint
    ch ch_half ch_quarter

3. 目前来看随机采样128并推理，准确率达不到%50，但是不太想管了，开始写训练吧，按最简单的先写好
