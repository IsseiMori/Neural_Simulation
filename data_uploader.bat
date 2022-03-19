rem UPLOAD
rem kubectl cp tmp/FLEX_RiceGrip imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp
rem kubectl cp tmp/FLEX_RiceGrip/predicted_data imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip

rem kubectl cp tmp/Finetune/RiceGrip/FLEX/partial/data imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/partial
rem kubectl cp tmp/Finetune/RiceGrip/MPM/partial/data imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/partial
rem kubectl cp tmp/Finetune/PressDown/FLEX/partial/data imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/PressDown/FLEX/partial
rem kubectl cp tmp/Finetune/PressDown/MPM/partial/data imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/PressDown/MPM/partial

rem kubectl cp tmp/Finetune/RiceGrip/MPM/reduced/full/data imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/reduced/full
rem kubectl cp tmp/Finetune/RiceGrip/MPM/reduced/partial/data imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/reduced/partial
rem kubectl cp tmp/Finetune/PressDown/MPM/reduced/full/data imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/PressDown/MPM/reduced/full
rem kubectl cp tmp/Finetune/PressDown/MPM/reduced/partial/data imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/PressDown/MPM/reduced/partial


rem kubectl cp tmp/Finetune/RiceGrip/MPM/reduced/full/data/metadata.json imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/reduced/full/data
rem kubectl cp tmp/Finetune/RiceGrip/MPM/reduced/partial/data/metadata.json imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/reduced/partial/data
rem kubectl cp tmp/Finetune/PressDown/MPM/reduced/full/data/metadata.json imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/PressDown/MPM/reduced/full/data
rem kubectl cp tmp/Finetune/PressDown/MPM/reduced/partial/data/metadata.json imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/PressDown/MPM/reduced/partial/data

rem kubectl cp tmp/Finetune/RiceGrip/MPM/reduced/full/data/metadata.json imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/reduced/full/data/rollouts
rem kubectl cp tmp/Finetune/RiceGrip/MPM/reduced/partial/data/metadata.json imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/reduced/partial/data/rollouts
rem kubectl cp tmp/Finetune/PressDown/MPM/reduced/full/data/metadata.json imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/PressDown/MPM/reduced/full/data/rollouts
rem kubectl cp tmp/Finetune/PressDown/MPM/reduced/partial/data/metadata.json imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/PressDown/MPM/reduced/partial/data/rollouts


rem Download
rem kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip/rollouts/test_300000 tmp/FLEX_RiceGrip/rollouts/test_300000
rem kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/FLEX_RiceGrip/models/model_300000.pth tmp/FLEX_RiceGrip/models/model_300000.pth

rem Download rollouts
kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/full/rollouts tmp/Finetune/RiceGrip/FLEX/full/rollouts
kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/partial/rollouts tmp/Finetune/RiceGrip/FLEX/partial/rollouts
kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/full/rollouts tmp/Finetune/RiceGrip/MPM/full/rollouts
kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/partial/rollouts tmp/Finetune/RiceGrip/MPM/partial/rollouts

kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/PressDown/FLEX/full/rollouts tmp/Finetune/PressDown/FLEX/full/rollouts
kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/PressDown/FLEX/partial/rollouts tmp/Finetune/PressDown/FLEX/partial/rollouts
kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/PressDown/MPM/full/rollouts tmp/Finetune/PressDown/MPM/full/rollouts
kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/PressDown/MPM/partial/rollouts tmp/Finetune/PressDown/MPM/partial/rollouts

kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/reduced/full/rollouts tmp/Finetune/RiceGrip/MPM/reduced/full/rollouts
kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/reduced/partial/rollouts tmp/Finetune/RiceGrip/MPM/reduced/partial/rollouts
kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/PressDown/MPM/reduced/full/rollouts tmp/Finetune/PressDown/MPM/reduced/full/rollouts
kubectl cp imori-pod-file-transfer:imori-fast-vol/Neural_Simulation/tmp/Finetune/PressDown/MPM/reduced/partial/rollouts tmp/Finetune/PressDown/MPM/reduced/partial/rollouts


kubectl logs imori-graphnet-job-press-down-flex-full--1-8kvhq; echo       
kubectl logs imori-graphnet-job-press-down-flex-partial--1-ph8cw; echo       
kubectl logs imori-graphnet-job-press-down-mpm-full--1-wjpdl; echo           
kubectl logs imori-graphnet-job-press-down-mpm-partial--1-67nn9; echo        
kubectl logs imori-graphnet-job-rice-grip-flex-full--1-dtmqx; echo           
kubectl logs imori-graphnet-job-rice-grip-flex-partial--1-89bc2; echo        
kubectl logs imori-graphnet-job-rice-grip-mpm-full--1-z9jcw; echo           
kubectl logs imori-graphnet-job-rice-grip-mpm-partial--1-7pb4d; echo
kubectl logs imori-graphnet-job-press-down-mpm-reduced-full--1-tz8w9; echo    
kubectl logs imori-graphnet-job-press-down-mpm-reduced-partial--1-2bhhx; echo         
kubectl logs imori-graphnet-job-rice-grip-mpm-reduced-full--1-8bdw9; echo
kubectl logs imori-graphnet-job-rice-grip-mpm-reduced-partial--1-xsbzz; echo
