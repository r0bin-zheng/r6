#!/bin/bash

# 生成一个日期时间（精确到秒） + “-” + 一个随机数作为id，并使用这个id创建一个文件夹
id=$(date +%Y%m%d%H%M%S)-$RANDOM
mkdir $id

# 执行文件后，输出重定向至id文件夹下的output.txt，并将id作为参数传递给python文件
python multi_tdeadp_main_2.py --n_obj=2 --n_var=2 --alg=0 --max_fe=250 --problem=dtlz1 --id=$id > $id/output.txt

# dtlz1
# python multi_tdeadp_main_2.py --n_obj=2 --n_var=2 --alg=0 --max_fe=250 --problem=dtlz1 > output1.txt
# python multi_tdeadp_main_2.py --n_obj=2 --n_var=2 --alg=1 --max_fe=250 --problem=dtlz1 > output2.txt
# python multi_tdeadp_main_2.py --n_obj=2 --n_var=2 --alg=2 --max_fe=250 --problem=dtlz1 > output3.txt

# python multi_tdeadp_main_2.py --n_obj=2 --n_var=3 --alg=0 --max_fe=250 --problem=dtlz1 > output4.txt
# python multi_tdeadp_main_2.py --n_obj=2 --n_var=3 --alg=1 --max_fe=250 --problem=dtlz1 > output5.txt
# python multi_tdeadp_main_2.py --n_obj=2 --n_var=3 --alg=2 --max_fe=250 --problem=dtlz1 > output6.txt

# python multi_tdeadp_main_2.py --n_obj=2 --n_var=5 --alg=0 --max_fe=250 --problem=dtlz1 > output7.txt
# python multi_tdeadp_main_2.py --n_obj=2 --n_var=5 --alg=1 --max_fe=250 --problem=dtlz1 > output8.txt
# python multi_tdeadp_main_2.py --n_obj=2 --n_var=5 --alg=2 --max_fe=250 --problem=dtlz1 > output9.txt

# python multi_tdeadp_main_2.py --n_obj=2 --n_var=10 --alg=0 --max_fe=500 --problem=dtlz1 > output10.txt
# python multi_tdeadp_main_2.py --n_obj=2 --n_var=10 --alg=1 --max_fe=500 --problem=dtlz1 > output11.txt
# python multi_tdeadp_main_2.py --n_obj=2 --n_var=10 --alg=2 --max_fe=500 --problem=dtlz1 > output12.txt

# python multi_tdeadp_main_2.py --n_obj=3 --n_var=2 --alg=0 --max_fe=250 --problem=dtlz1 > output13.txt
# python multi_tdeadp_main_2.py --n_obj=3 --n_var=2 --alg=1 --max_fe=250 --problem=dtlz1 > output14.txt
# python multi_tdeadp_main_2.py --n_obj=3 --n_var=2 --alg=2 --max_fe=250 --problem=dtlz1 > output15.txt

# python multi_tdeadp_main_2.py --n_obj=3 --n_var=3 --alg=0 --max_fe=250 --problem=dtlz1 > output16.txt
# python multi_tdeadp_main_2.py --n_obj=3 --n_var=3 --alg=1 --max_fe=250 --problem=dtlz1 > output17.txt
# python multi_tdeadp_main_2.py --n_obj=3 --n_var=3 --alg=2 --max_fe=250 --problem=dtlz1 > output18.txt

# python multi_tdeadp_main_2.py --n_obj=3 --n_var=5 --alg=0 --max_fe=250 --problem=dtlz1 > output19.txt
# python multi_tdeadp_main_2.py --n_obj=3 --n_var=5 --alg=1 --max_fe=250 --problem=dtlz1 > output20.txt
# python multi_tdeadp_main_2.py --n_obj=3 --n_var=5 --alg=2 --max_fe=250 --problem=dtlz1 > output21.txt

# python multi_tdeadp_main_2.py --n_obj=3 --n_var=10 --alg=0 --max_fe=500 --problem=dtlz1 > output22.txt
# python multi_tdeadp_main_2.py --n_obj=3 --n_var=10 --alg=1 --max_fe=500 --problem=dtlz1 > output23.txt
# python multi_tdeadp_main_2.py --n_obj=3 --n_var=10 --alg=2 --max_fe=500 --problem=dtlz1 > output24.txt

# python multi_tdeadp_main_2.py --n_obj=5 --n_var=2 --alg=0 --max_fe=250 --problem=dtlz1 > output25.txt
# python multi_tdeadp_main_2.py --n_obj=5 --n_var=2 --alg=1 --max_fe=250 --problem=dtlz1 > output26.txt
# python multi_tdeadp_main_2.py --n_obj=5 --n_var=2 --alg=2 --max_fe=250 --problem=dtlz1 > output27.txt

# python multi_tdeadp_main_2.py --n_obj=5 --n_var=3 --alg=0 --max_fe=250 --problem=dtlz1 > output28.txt
# python multi_tdeadp_main_2.py --n_obj=5 --n_var=3 --alg=1 --max_fe=250 --problem=dtlz1 > output29.txt
# python multi_tdeadp_main_2.py --n_obj=5 --n_var=3 --alg=2 --max_fe=250 --problem=dtlz1 > output30.txt

# python multi_tdeadp_main_2.py --n_obj=5 --n_var=5 --alg=0 --max_fe=250 --problem=dtlz1 > output31.txt
# python multi_tdeadp_main_2.py --n_obj=5 --n_var=5 --alg=1 --max_fe=250 --problem=dtlz1 > output32.txt
# python multi_tdeadp_main_2.py --n_obj=5 --n_var=5 --alg=2 --max_fe=250 --problem=dtlz1 > output33.txt

# python multi_tdeadp_main_2.py --n_obj=5 --n_var=10 --alg=0 --max_fe=500 --problem=dtlz1 > output34.txt
# python multi_tdeadp_main_2.py --n_obj=5 --n_var=10 --alg=1 --max_fe=500 --problem=dtlz1 > output35.txt
# python multi_tdeadp_main_2.py --n_obj=5 --n_var=10 --alg=2 --max_fe=500 --problem=dtlz1 > output36.txt

# # dtlz2
# python multi_tdeadp_main_2.py --n_obj=2 --n_var=2 --alg=0 --max_fe=250 --problem=dtlz2 > output37.txt
# python multi_tdeadp_main_2.py --n_obj=2 --n_var=2 --alg=1 --max_fe=250 --problem=dtlz2 > output38.txt
# python multi_tdeadp_main_2.py --n_obj=2 --n_var=2 --alg=2 --max_fe=250 --problem=dtlz2 > output39.txt

# python multi_tdeadp_main_2.py --n_obj=2 --n_var=3 --alg=0 --max_fe=250 --problem=dtlz2 > output40.txt
# python multi_tdeadp_main_2.py --n_obj=2 --n_var=3 --alg=1 --max_fe=250 --problem=dtlz2 > output41.txt
# python multi_tdeadp_main_2.py --n_obj=2 --n_var=3 --alg=2 --max_fe=250 --problem=dtlz2 > output42.txt

# python multi_tdeadp_main_2.py --n_obj=2 --n_var=5 --alg=0 --max_fe=250 --problem=dtlz2 > output43.txt
# python multi_tdeadp_main_2.py --n_obj=2 --n_var=5 --alg=1 --max_fe=250 --problem=dtlz2 > output44.txt
# python multi_tdeadp_main_2.py --n_obj=2 --n_var=5 --alg=2 --max_fe=250 --problem=dtlz2 > output45.txt

# python multi_tdeadp_main_2.py --n_obj=2 --n_var=10 --alg=0 --max_fe=500 --problem=dtlz2 > output46.txt
# python multi_tdeadp_main_2.py --n_obj=2 --n_var=10 --alg=1 --max_fe=500 --problem=dtlz2 > output47.txt
# python multi_tdeadp_main_2.py --n_obj=2 --n_var=10 --alg=2 --max_fe=500 --problem=dtlz2 > output48.txt

# python multi_tdeadp_main_2.py --n_obj=3 --n_var=2 --alg=0 --max_fe=250 --problem=dtlz2 > output49.txt
# python multi_tdeadp_main_2.py --n_obj=3 --n_var=2 --alg=1 --max_fe=250 --problem=dtlz2 > output50.txt
# python multi_tdeadp_main_2.py --n_obj=3 --n_var=2 --alg=2 --max_fe=250 --problem=dtlz2 > output51.txt

# python multi_tdeadp_main_2.py --n_obj=3 --n_var=3 --alg=0 --max_fe=250 --problem=dtlz2 > output52.txt
# python multi_tdeadp_main_2.py --n_obj=3 --n_var=3 --alg=1 --max_fe=250 --problem=dtlz2 > output53.txt
# python multi_tdeadp_main_2.py --n_obj=3 --n_var=3 --alg=2 --max_fe=250 --problem=dtlz2 > output54.txt

# python multi_tdeadp_main_2.py --n_obj=3 --n_var=5 --alg=0 --max_fe=250 --problem=dtlz2 > output55.txt
# python multi_tdeadp_main_2.py --n_obj=3 --n_var=5 --alg=1 --max_fe=250 --problem=dtlz2 > output56.txt
# python multi_tdeadp_main_2.py --n_obj=3 --n_var=5 --alg=2 --max_fe=250 --problem=dtlz2 > output57.txt

# python multi_tdeadp_main_2.py --n_obj=3 --n_var=10 --alg=0 --max_fe=500 --problem=dtlz2 > output58.txt
# python multi_tdeadp_main_2.py --n_obj=3 --n_var=10 --alg=1 --max_fe=500 --problem=dtlz2 > output59.txt
# python multi_tdeadp_main_2.py --n_obj=3 --n_var=10 --alg=2 --max_fe=500 --problem=dtlz2 > output60.txt

# python multi_tdeadp_main_2.py --n_obj=5 --n_var=2 --alg=0 --max_fe=250 --problem=dtlz2 > output61.txt
# python multi_tdeadp_main_2.py --n_obj=5 --n_var=2 --alg=1 --max_fe=250 --problem=dtlz2 > output62.txt
# python multi_tdeadp_main_2.py --n_obj=5 --n_var=2 --alg=2 --max_fe=250 --problem=dtlz2 > output63.txt

# python multi_tdeadp_main_2.py --n_obj=5 --n_var=3 --alg=0 --max_fe=250 --problem=dtlz2 > output64.txt
# python multi_tdeadp_main_2.py --n_obj=5 --n_var=3 --alg=1 --max_fe=250 --problem=dtlz2 > output65.txt
# python multi_tdeadp_main_2.py --n_obj=5 --n_var=3 --alg=2 --max_fe=250 --problem=dtlz2 > output66.txt

# python multi_tdeadp_main_2.py --n_obj=5 --n_var=5 --alg=0 --max_fe=250 --problem=dtlz2 > output67.txt
# python multi_tdeadp_main_2.py --n_obj=5 --n_var=5 --alg=1 --max_fe=250 --problem=dtlz2 > output68.txt
# python multi_tdeadp_main_2.py --n_obj=5 --n_var=5 --alg=2 --max_fe=250 --problem=dtlz2 > output69.txt

# python multi_tdeadp_main_2.py --n_obj=5 --n_var=10 --alg=0 --max_fe=500 --problem=dtlz2 > output70.txt
# python multi_tdeadp_main_2.py --n_obj=5 --n_var=10 --alg=1 --max_fe=500 --problem=dtlz2 > output71.txt
# python multi_tdeadp_main_2.py --n_obj=5 --n_var=10 --alg=2 --max_fe=500 --problem=dtlz2 > output72.txt
