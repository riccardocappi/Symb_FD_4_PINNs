conda activate symb_fd

echo "Executing exp seed 0 4-4:15"

source experiments.sh 0 &
sleep 1m
# wait

echo "Executing exp seed 42 4-4:15"
source experiments.sh 42 &
# wait
sleep 1m

echo "Executing exp seed 12345 4-4:15"
source experiments.sh 12345 &


wait


echo "Executing exp seed 0 5-5:15"

source experiments_i80_5:00-5:15.sh 0 & 
sleep 1m

echo "Executing exp seed 42 5-5:15"
source experiments_i80_5:00-5:15.sh 42 &
sleep 1m

echo "Executing exp seed 12345 5-5:15"
source experiments_i80_5:00-5:15.sh 12345 &

wait


echo "Executing exp seed 0 5:15-5:30"

source experiments_i80_5:15-5:30.sh 0 & 
sleep 1m

echo "Executing exp seed 42 5:15-5:30"
source experiments_i80_5:15-5:30.sh 42 &
sleep 1m

echo "Executing exp seed 12345 5:15-5:30"
source experiments_i80_5:15-5:30.sh 12345