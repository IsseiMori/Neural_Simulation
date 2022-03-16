rem UPLOAD
rem kubectl cp tmp/FLEX_RiceGrip imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp
rem kubectl cp tmp/FLEX_RiceGrip/predicted_data imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip

kubectl cp tmp/Finetune/RiceGrip/FLEX/full/data imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/full
kubectl cp tmp/Finetune/RiceGrip/MPM/full/data imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/full
kubectl cp tmp/Finetune/PressDown/FLEX/full/data imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/PressDown/FLEX/full
kubectl cp tmp/Finetune/PressDown/MPM/full/data imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/PressDown/MPM/full


rem Download
rem kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip/rollouts/test_300000 tmp/FLEX_RiceGrip/rollouts/test_300000
rem kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip/models/model_300000.pth tmp/FLEX_RiceGrip/models/model_300000.pth

rem kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip/mpm_data tmp/FLEX_RiceGrip/mpm_data
rem kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip/predicted_data tmp/FLEX_RiceGrip/predicted_data