echo 'Calling scripts!'

# # toggle this on if you want to delete old logs each time you run new experiments
# rm -rf ../logs



for i in {1..5}
do
    python main.py --data=mnist &
    python main.py --data=fmnist &

    python main.py --data=cifar10 --device=cuda:1
done


echo 'All experiments are finished!'