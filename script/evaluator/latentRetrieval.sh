SLURM_ARRAY_TASK_ID=0

##############################################################################
# Load the required modules.
##############################################################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scope

##############################################################################
#Evaluate the retrieval performance using the sparse latent features.
##############################################################################

ENTRYPOINT="source.evaluator.latentRetrieval"

latentTopKPool=(32 64 128)
latentTopKPick=${latentTopKPool[$SLURM_ARRAY_TASK_ID]}

CMD="--dataset msMarco --embedding bgeBase"
CMD="$CMD --latentSize 196K --latentTopK $latentTopKPick"
CMD="$CMD --modelName bgeBase-196K-$latentTopKPick --modelDevice 0"
python3 -m $ENTRYPOINT $CMD

CMD="--dataset msMarco --embedding miniCPM"
CMD="$CMD --latentSize 196K --latentTopK $latentTopKPick"
CMD="$CMD --modelName miniCPM-196K-$latentTopKPick --modelDevice 0"
python3 -m $ENTRYPOINT $CMD
