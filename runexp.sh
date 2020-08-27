set -e
set -x
nexp=$(ls -1 /inputs | wc -l) 
echo "loading $nexp experiments" 
counter=1 
 for i in /inputs/* ; do 
     o=/results/$(basename ${i%.json}).npy.lz4
     if [ -f "$o" ] ; then
        echo [${counter}/${nexp}]: ${i} already done, skipping 
    else 
        echo -n [${counter}/${nexp}]": "
        python scripts/json_train.py  --results ${o} ${i} 
    fi 
    : $((counter++)) 
 done 
echo "runnign re-identification experiments" 
if [ ! -f /results/reid100_seq.txt ] ; then
    python    scripts/dynamic_aggregation.py  -b 1  -s 1 --running-mean -1 \
    --seq-up-to 10 --fold 10,100,10 --tests 1000  --no-frames /reid100 \
    --output /results/reid100_seq.txt
fi
if [ ! -f /results/reid100_frames.txt ] ; then
    python    scripts/dynamic_aggregation.py  -b 1  -s 1 --running-mean 1 \
    --seq-up-to 10 --fold 10,100,10 --tests 1000  --no-sequences /reid100 \
    --output /results/reid100_frames.txt
fi
if [ ! -f /results/mugs9_seq.txt ] ; then
    python    scripts/dynamic_aggregation.py  -b 1  -s 1 --running-mean -1 \
    --seq-up-to 5 --fold 5,9,1 --tests 1000  --no-sequences /reid9mugs \
    --output /results/mugs9_seq.txt
fi
if [ ! -f /results/mugs9_frames.txt ] ; then
    python    scripts/dynamic_aggregation.py  -b 1  -s 1 --running-mean 1 \
    --seq-up-to 5 --fold 5,9,1 --tests 1000  --no-frames /reid9mugs \
    --output /results/mugs9_frames.txt
fi
echo "completed re-identification experiments" 
 echo creating figures... 
 python scripts/plot_closedworld.py /results/reid100_seq.txt /results/reid100_frames.txt \
        --output /outputs/cwreid100
 python scripts/plot_closedworld.py /results/mugs9_seq.txt /results/mugs9_frames.txt \
        --output /outputs/cwmugs9
python scripts/plot_openworld.py -o /outputs/reid100 /results/online.npy.lz4 /results/active92.npy.lz4
python scripts/plot_openworld.py -o /outputs/reid100_devel /results/online_n30.npy.lz4 /results/active35_n30.npy.lz4
python scripts/plot_openworld.py -o /outputs/core /results/coreonline.npy.lz4 /results/coreactive84.npy.lz4
python scripts/plot_openworld.py -o /outputs/core_devel /results/coreonline_n30.npy.lz4 /results/coreactive60_n30.npy.lz4
python scripts/plot_unsupervised.py --rr $(cat /info/rr.txt) \
        --dr $(cat /info/dr.txt) --rd $(cat /info/rd.txt) \
        --dd $(cat /info/dd.txt) > /outputs/table1.tex
