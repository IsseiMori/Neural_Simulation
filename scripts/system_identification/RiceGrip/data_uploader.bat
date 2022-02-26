rem UPLOAD
rem kubectl cp tmp/FLEX_RiceGrip imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp


rem Download
kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/scripts/system_identification/RiceGrip/images_flex images_flex
kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/scripts/system_identification/RiceGrip/images_mpm images_mpm
kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/scripts/system_identification/RiceGrip/rollouts_flex rollouts_flex
kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/scripts/system_identification/RiceGrip/rollouts_mpm rollouts_mpm