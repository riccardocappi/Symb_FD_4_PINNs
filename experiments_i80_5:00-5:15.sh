conda activate symb_fd


SEED=${1:? "Usage: ./run_experiments.sh <seed>"}

DATA_NAME="I80_5_5:15pm_075_seed${SEED}.pth"
STUDY_SUFFIX="scaled_data_seed${SEED}_I80_5_5:15pm"


python main.py \
  --config_path=./conf/pinn_fd.yml \
  --n_trials=100 \
  --study_name="FD_${STUDY_SUFFIX}_3" \
  --seed=${SEED} \
  --scale \
  --learn_fd \
  --data_name=${DATA_NAME} \
  --load_data \
  --vel_path=./data/I80/NGSIM_US80_5_5:15_pm_Velocity_Data.txt \
  --dens_path=./data/I80/NGSIM_US80_5_5:15_pm_Density_Data.txt 

# sleep 1m



# python main.py \
#   --config_path=./conf/pinn.yml \
#   --n_trials=100 \
#   --study_name="${STUDY_SUFFIX}" \
#   --seed=${SEED} \
#   --scale \
#   --data_name=${DATA_NAME} \
#   --load_data \
#   --vel_path=./data/I80/NGSIM_US80_5_5:15_pm_Velocity_Data.txt \
#   --dens_path=./data/I80/NGSIM_US80_5_5:15_pm_Density_Data.txt &

# sleep 1m

# python main.py \
#   --config_path=./conf/symb_fd.yml \
#   --n_trials=100 \
#   --study_name="${STUDY_SUFFIX}" \
#   --seed=${SEED} \
#   --scale \
#   --data_name=${DATA_NAME} \
#   --load_data \
#   --vel_path=./data/I80/NGSIM_US80_5_5:15_pm_Velocity_Data.txt \
#   --dens_path=./data/I80/NGSIM_US80_5_5:15_pm_Density_Data.txt &


# sleep 1m

# python main.py \
#   --config_path=./conf/punn.yml \
#   --n_trials=100 \
#   --study_name="${STUDY_SUFFIX}" \
#   --seed=${SEED} \
#   --scale \
#   --data_name=${DATA_NAME} \
#   --load_data \
#   --vel_path=./data/I80/NGSIM_US80_5_5:15_pm_Velocity_Data.txt \
#   --dens_path=./data/I80/NGSIM_US80_5_5:15_pm_Density_Data.txt &


# python main.py \
#   --config_path=./conf/pinn.yml \
#   --n_trials=100 \
#   --study_name="${STUDY_SUFFIX}_end_to_end" \
#   --seed=${SEED} \
#   --scale \
#   --data_name=${DATA_NAME} \
#   --load_data \
#   --vel_path=./data/I80/NGSIM_US80_5_5:15_pm_Velocity_Data.txt \
#   --dens_path=./data/I80/NGSIM_US80_5_5:15_pm_Density_Data.txt \
#   --end_to_end

# python main.py \
#   --config_path=./conf/symb_fd_poly.yml \
#   --n_trials=100 \
#   --study_name="${STUDY_SUFFIX}-poly-fit" \
#   --seed=${SEED} \
#   --scale \
#   --data_name=${DATA_NAME} \
#   --load_data \
#   --poly_fit \
#   --vel_path=./data/I80/NGSIM_US80_5_5:15_pm_Velocity_Data.txt \
#   --dens_path=./data/I80/NGSIM_US80_5_5:15_pm_Density_Data.txt 
