OK，已经验收并暂存。接下来帮我做这几个任务：
1. main的'--profile'参数改名为'--dry-run'，该模式下只进行数据生成->h2d->kernel执行->d2h，不进行正确性校验
2. 把cpu_reference校验逻辑独立成两个函数：一个叫'sgemm_golden'，执行cpu ref的sgemm计算，可以用vector来返回；另一个叫'sgemm_verify'，用于校验比对正确性，返回一个true/false的bool值，如果比对失败，在函数内打印错误报告日志（如果错误的元素非常多，只打前8个错误的位置和值等信息），比对使用业界通用的混合容差公式，我记得是`|out-golden|<atol+rtol*|golden|`，这里的rtol和atol和数据类型有关，查一下FP32的值是多少应用进去。
3. 代码格式整改：所有的for/if等控制流，即使只是单行，也用花括号{}包裹；如果这个要求可以写进.clang-format的话可以补充，不能就算了。
4. profile.sh把输出目录从'report/'换成'ncu-rep/'，旧目录里的数据可以全删掉了
5. 试着用多线程技术优化'sgemm_golden'和'sgemm_verify'函数，如果较复杂的话可以独立出一个verify.cpp来实现，尽可能平衡性能收益和代码简洁性。
6. profile.sh里把ncu的命令的每个参数的含义和作用，用简单的注释附上去，方便我学习记忆。
7. build.sh的'--run'参数后也要支持传递'--dry-run'等main.cpp的参数，这方面请你以脚本怎么写比较简单怎么来，并附上usage，例如可以'./scripts/build.sh [--clean] --run "--dry-run [--param1 [val1]] [--param2 [val2]]"'这样来附加执行参数
8. 全部测通以后，再采集一次ncu报告，放在新的目录里，并确认这份新报告的v0/v1频率差距是比较小的（比较稳定），然后以此修正docs/下的几份文档。这次的报告名就用'-o sgemm.v0v1.0719'