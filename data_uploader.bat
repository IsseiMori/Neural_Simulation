rem UPLOAD
rem kubectl cp tmp/FLEX_RiceGrip imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp
rem kubectl cp tmp/FLEX_RiceGrip/predicted_data imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip
rem kubectl cp tmp/FLEX_RiceGrip/mpm_data imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip



rem Download
rem kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip/rollouts/test_500000 tmp/FLEX_RiceGrip/rollouts/test_500000

kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip/models/model_500000.pth tmp/FLEX_RiceGrip/models/model_500000.pth
kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip/mpm_data tmp/FLEX_RiceGrip/mpm_data
kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip/predicted_data tmp/FLEX_RiceGrip/predicted_data