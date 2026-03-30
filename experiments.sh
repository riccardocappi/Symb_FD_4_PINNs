conda activate symb_fd


SEED=${1:? "Usage: ./run_experiments.sh <seed>"}

DATA_NAME="I80_075_seed${SEED}.pth"
STUDY_SUFFIX="scaled_data_seed${SEED}_I80"


python main.py \
  --config_path=./conf/pinn_fd.yml \
  --n_trials=100 \
  --study_name="FD_${STUDY_SUFFIX}" \
  --seed=${SEED} \
  --scale \
  --learn_fd \
  --data_name=${DATA_NAME} \
  --load_data &

sleep 1m



python main.py \
  --config_path=./conf/pinn.yml \
  --n_trials=100 \
  --study_name="${STUDY_SUFFIX}" \
  --seed=${SEED} \
  --scale \
  --data_name=${DATA_NAME} \
  --load_data &


sleep 1m


python main.py \
  --config_path=./conf/symb_fd.yml \
  --n_trials=100 \
  --study_name="${STUDY_SUFFIX}" \
  --seed=${SEED} \
  --scale \
  --data_name=${DATA_NAME} \
  --load_data & 

wait

python main.py \
  --config_path=./conf/punn.yml \
  --n_trials=100 \
  --study_name="${STUDY_SUFFIX}" \
  --seed=${SEED} \
  --scale \
  --data_name=${DATA_NAME} \
  --load_data &
 

sleep 1m

python main.py \
  --config_path=./conf/pinn.yml \
  --n_trials=100 \
  --study_name="${STUDY_SUFFIX}_end_to_end" \
  --seed=${SEED} \
  --scale \
  --data_name=${DATA_NAME} \
  --load_data \
  --end_to_end &


sleep 1m

python main.py \
  --config_path=./conf/symb_fd_poly.yml \
  --n_trials=100 \
  --study_name="${STUDY_SUFFIX}-poly-fit" \
  --seed=${SEED} \
  --scale \
  --data_name=${DATA_NAME} \
  --load_data \
  --poly_fit