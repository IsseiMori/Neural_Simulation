rem UPLOAD
rem kubectl cp tmp/FLEX_RiceGrip imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp
rem kubectl cp tmp/FLEX_RiceGrip/predicted_data imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip
rem kubectl cp tmp/FLEX_RiceGrip/mpm_data imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip

rem kubectl cp tmp/FLEX_RiceGrip/data/metadata.json imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip/data
rem kubectl cp tmp/FLEX_RiceGrip/data/rollouts/metadata.json imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip/data/rollouts

rem Download
kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip/rollouts/test_300000 tmp/FLEX_RiceGrip/rollouts/test_300000
kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip/models/model_300000.pth tmp/FLEX_RiceGrip/models/model_300000.pth

rem kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip/mpm_data tmp/FLEX_RiceGrip/mpm_data
rem kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip/predicted_data tmp/FLEX_RiceGrip/predicted_data